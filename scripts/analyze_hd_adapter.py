"""Read-only diagnosis of the frozen train exploration; never calls a model."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from merit_qvh.core import file_sha256, write_json


def read_rows(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    result = {row["qid"]: row for row in rows}
    assert len(result) == len(rows), path
    return result


def compare(metrics, records, qids):
    """Source-clustered, query-weighted paired bootstrap, including ties."""
    qids = sorted(qids)
    delta = np.array([[metrics[q]["H1"][k] - metrics[q]["D0"][k] for k in ("mAP", "Hit1")] for q in qids])
    sources = sorted({records[q]["vid"].rsplit("_", 2)[0] for q in qids})
    sums = np.zeros((len(sources), 2))
    counts = np.zeros(len(sources))
    for q, d in zip(qids, delta, strict=True):
        i = sources.index(records[q]["vid"].rsplit("_", 2)[0])
        sums[i] += d
        counts[i] += 1
    rng = np.random.default_rng(42)
    draws = rng.integers(0, len(sources), size=(10000, len(sources)))
    ci = np.percentile(sums[draws].sum(1) / counts[draws].sum(1)[:, None], [2.5, 97.5], axis=0)
    return {
        "queries": len(qids),
        "sources": len(sources),
        "group_means": {
            g: {k: float(np.mean([metrics[q][g][k] for q in qids])) for k in ("mAP", "Hit1")}
            for g in ("D0", "H1")
        },
        "H1_minus_D0": {
            k: {
                "delta_pp": float(delta[:, i].mean()),
                "ci95_pp": ci[:, i].tolist(),
                "wins": int((delta[:, i] > 1e-9).sum()),
                "ties": int((np.abs(delta[:, i]) <= 1e-9).sum()),
                "losses": int((delta[:, i] < -1e-9).sum()),
            }
            for i, k in enumerate(("mAP", "Hit1"))
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--evaluator", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root, output = args.run_root.resolve(), args.output.resolve()
    if output == root or root in output.parents:
        raise ValueError("Write diagnostics outside the immutable experiment run")
    if output.exists():
        raise ValueError("Use a new output directory; keep previous diagnoses")
    sys.path.insert(0, str(args.evaluator))
    from standalone_eval.eval import compute_ap_from_tuple, mk_gt_scores

    truth_path = root / "data/ground_truth/explore.jsonl"
    paths = [truth_path, root / "protocol.json", root / "explore_report.json"]
    paths += sorted((root / "queries").glob("*.json"))
    paths += [root / f"explore_{g}_predictions.jsonl" for g in ("M0", "D0", "H1")]
    hashes = {str(p.relative_to(root)): file_sha256(p) for p in paths}
    truth = read_rows(truth_path)
    records = {q: json.loads((root / f"queries/{q}.json").read_text()) for q in truth}
    predictions = {g: read_rows(root / f"explore_{g}_predictions.jsonl") for g in ("M0", "D0", "H1")}
    official = json.loads((root / "explore_report.json").read_text())
    for rows in predictions.values():
        assert set(rows) == set(truth)
    metrics, details = {}, []
    for q, gt in truth.items():
        binary = (mk_gt_scores(gt) >= 4).astype(float)
        metrics[q] = {}
        for g, rows in predictions.items():
            scores = np.asarray(rows[q]["pred_saliency_scores"], dtype=float)
            assert scores.tolist() == records[q][g]["pred_saliency_scores"]
            assert rows[q]["vid"] == gt["vid"] == records[q]["vid"]
            assert len(scores) == len(binary) and np.isfinite(scores).all()
            metrics[q][g] = {
                "mAP": float(
                    100 * np.mean([compute_ap_from_tuple((0, w, binary[:, w], scores))[2] for w in range(3)])
                ),
                "Hit1": float(100 * binary[int(scores.argmax())].max()),
            }
        r = records[q]
        d, h = np.asarray(r["D0"]["pred_saliency_scores"]), np.asarray(r["H1"]["pred_saliency_scores"])
        ids = r["H1"].get("updated_clip_ids", [])
        positive = set(np.flatnonzero(binary.max(axis=1)))
        paired_ok = r["D0"]["status"] == r["H1"]["status"] == "ok"
        changed = np.flatnonzero(h != d).tolist() if paired_ok else []
        if paired_ok:
            assert set(changed) <= set(ids)
            frame_hashes = dict(
                zip(
                    [i for b in r["D0"]["batches"] for i in b["ids"]],
                    [i for b in r["D0"]["batches"] for i in b["frame_hashes"]],
                    strict=True,
                )
            )
            for b in r["H1"]["batches"]:
                assert b["frame_hashes"] == [frame_hashes[i] for i in b["ids"]]
        details.append(
            {
                "qid": q,
                "query": r["query"],
                "vid": r["vid"],
                "statuses": {g: r[g]["status"] for g in predictions},
                "metrics": metrics[q],
                "paired_success": paired_ok,
                "delta_mAP_pp": metrics[q]["H1"]["mAP"] - metrics[q]["D0"]["mAP"],
                "bins": len(d),
                "updated_bins": len(ids),
                "changed_bins": changed,
                "score_increases": int((h > d).sum()) if paired_ok else None,
                "score_decreases": int((h < d).sum()) if paired_ok else None,
                "verygood_any_annotator_bins": len(positive),
                "selected_positive_bins": len(set(ids) & positive) if paired_ok else None,
                "selected_positive_recall": len(set(ids) & positive) / len(positive)
                if paired_ok and positive
                else None,
                "D0_top_ties": int((d == d.max()).sum()),
                "H1_top_ties": int((h == h.max()).sum()),
                "selected_contains_D0_top": int(d.argmax()) in ids if paired_ok else None,
                "errors": {g: r[g].get("error") for g in predictions if r[g]["status"] != "ok"},
            }
        )
    for g in predictions:
        for name, key in (("mAP", "HL-mAP"), ("Hit1", "HL-Hit1")):
            assert (
                round(float(np.mean([x[g][name] for x in metrics.values()])), 2)
                == official["groups"][g]["metrics"]["HL-min-VeryGood"][key]
            )
    paired = [r for r in details if r["paired_success"]]
    excluded = [r for r in details if not r["paired_success"]]
    result = {
        "scope": "Post-hoc train exploration diagnosis, not confirmation or a causal ablation",
        "protocol_id": records[next(iter(records))]["protocol_id"],
        "bootstrap": {"seed": 42, "replicates": 10000, "cluster": "source video", "ci": "percentile 95%"},
        "full_denominator": compare(metrics, records, truth),
        "paired_success_diagnostic_only": compare(metrics, records, [r["qid"] for r in paired]),
        "decomposition_full_delta_pp": {
            "paired_success_contribution": sum(r["delta_mAP_pp"] for r in paired) / len(truth),
            "remaining_failure_records_contribution": sum(r["delta_mAP_pp"] for r in excluded) / len(truth),
        },
        "paired_success_structure": {
            "queries": len(paired),
            "changed_vectors": sum(bool(r["changed_bins"]) for r in paired),
            "unchanged_vectors": sum(not r["changed_bins"] for r in paired),
            "changed_bins": sum(len(r["changed_bins"]) for r in paired),
            "updated_bins": sum(r["updated_bins"] for r in paired),
            "total_bins": sum(r["bins"] for r in paired),
            "mean_positive_recall": float(
                np.mean(
                    [
                        r["selected_positive_recall"]
                        for r in paired
                        if r["selected_positive_recall"] is not None
                    ]
                )
            ),
            "no_positive_selected": sum(r["selected_positive_recall"] == 0 for r in paired),
            "positive_recall_denominator_queries": sum(
                r["selected_positive_recall"] is not None for r in paired
            ),
            "D0_tied_max_queries": sum(r["D0_top_ties"] > 1 for r in paired),
            "H1_tied_max_queries": sum(r["H1_top_ties"] > 1 for r in paired),
            "identical_D0_H1_images_verified": True,
        },
        "failures": [{k: r[k] for k in ("qid", "statuses", "delta_mAP_pp", "errors")} for r in excluded],
        "status_counts": {g: dict(Counter(r[g]["status"] for r in records.values())) for g in predictions},
        "input_sha256": hashes,
        "evaluator_sha256": {
            name: file_sha256(args.evaluator / "standalone_eval" / name) for name in ("eval.py", "utils.py")
        },
        "script_sha256": file_sha256(Path(__file__)),
    }
    assert hashes == {str(p.relative_to(root)): file_sha256(p) for p in paths}
    output.mkdir(parents=True)
    write_json(output / "analysis.json", result)
    write_json(output / "per_query.json", details)
    print(
        json.dumps(
            {k: v for k, v in result.items() if not k.endswith("sha256")}, ensure_ascii=False, indent=2
        )
    )


if __name__ == "__main__":
    main()
