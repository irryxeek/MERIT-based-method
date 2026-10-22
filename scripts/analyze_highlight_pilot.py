"""Offline paired diagnostics for the fixed twenty-query evidence pilot."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from merit_qvh.core import file_sha256, read_jsonl, write_json


def paired(metrics, records, left, right):
    ids = sorted(metrics)
    sources = sorted({records[q]["vid"].rsplit("_", 2)[0] for q in ids})
    delta = np.array([[metrics[q][left][k] - metrics[q][right][k] for k in ("mAP", "Hit1")] for q in ids])
    sums, counts = np.zeros((len(sources), 2)), np.zeros(len(sources))
    for q, d in zip(ids, delta, strict=True):
        index = sources.index(records[q]["vid"].rsplit("_", 2)[0])
        sums[index] += d
        counts[index] += 1
    samples = np.random.default_rng(42).integers(0, len(sources), (10000, len(sources)))
    ci = np.percentile(sums[samples].sum(1) / counts[samples].sum(1)[:, None], [2.5, 97.5], axis=0)
    return {
        k: {
            "delta_pp": float(delta[:, i].mean()),
            "ci95_pp": ci[:, i].tolist(),
            "wins": int((delta[:, i] > 1e-9).sum()),
            "ties": int((np.abs(delta[:, i]) <= 1e-9).sum()),
            "losses": int((delta[:, i] < -1e-9).sum()),
        }
        for i, k in enumerate(("mAP", "Hit1"))
    }


def analyze(root, output):
    if output.exists() or root == output or root in output.parents:
        raise ValueError("Use a new diagnostic directory outside the frozen run")
    report = json.loads((root / "report.json").read_text())
    truth = {r["qid"]: r for r in read_jsonl(root / "data/ground_truth/pilot.jsonl")}
    groups = ("D0", "E1", "E2", "E3")
    records = {q: json.loads((root / "queries" / (str(q) + ".json")).read_text()) for q in truth}
    paths = [root / "report.json", root / "protocol.json", root / "data/ground_truth/pilot.jsonl"]
    paths += sorted((root / "queries").glob("*.json"))
    paths += sorted((root / "memories").glob("*/*.json"))
    paths += sorted((root / "evaluator").rglob("*.py"))
    hashes = {str(p.relative_to(root)): file_sha256(p) for p in paths}
    sys.path.insert(0, str(root / "evaluator"))
    from standalone_eval.eval import compute_ap_from_tuple, mk_gt_scores

    metrics, details = {}, []
    for q, gt in truth.items():
        binary = (mk_gt_scores(gt) >= 4).astype(float)
        r = records[q]
        initial = np.asarray(r["D0"]["pred_saliency_scores"])
        metrics[q] = {}
        ties, changed = {}, {}
        selected = set(r.get("selected_bin_ids", []))
        for g in groups:
            scores = np.asarray(r[g]["pred_saliency_scores"])
            if len(scores) != len(binary) or not np.isfinite(scores).all():
                raise ValueError("Invalid dense score vector")
            metrics[q][g] = {
                "mAP": float(
                    100 * np.mean([compute_ap_from_tuple((0, i, binary[:, i], scores))[2] for i in range(3)])
                ),
                "Hit1": float(100 * binary[int(scores.argmax())].max()),
            }
            ties[g] = int((scores == scores.max()).sum())
            changed[g] = np.flatnonzero(scores != initial).tolist()
            if not set(changed[g]) <= selected:
                raise ValueError("Changes outside selected bins")
        acquisition = r.get("acquisition", {})
        evidence = acquisition.get("frames", [])
        before = evidence[: len(initial)]
        after = evidence[len(initial) :]
        if len({f["pts"] for f in after}) != len(after) or {f["pts"] for f in before} & {
            f["pts"] for f in after
        }:
            raise ValueError("New evidence repeats an already seen PTS")
        if len(after) > 48 or acquisition.get("used_views", 0) > 2 or len(r.get("comparisons", [])) > 3:
            raise ValueError("Acquisition/comparison budget exceeded")
        positives = set(np.flatnonzero(binary.max(axis=1)))
        details.append(
            {
                "qid": q,
                "vid": r["vid"],
                "query": r["query"],
                "metrics": metrics[q],
                "statuses": {g: r[g]["status"] for g in groups},
                "top_ties": ties,
                "changed_bins_vs_D0": changed,
                "selected_bins": len(selected),
                "bins": len(initial),
                "positive_bins": len(positives),
                "selected_positive_bins": len(selected & positives),
                "selected_positive_recall": len(selected & positives) / len(positives) if positives else None,
                "new_frames": len(after),
                "views": acquisition.get("used_views", 0),
                "needs": acquisition.get("planned_needs", []),
                "gaps": acquisition.get("gaps", []),
                "comparison_outcomes": [p["winner"] for p in r.get("comparisons", [])],
                "view_failures": [s for s in acquisition.get("steps", []) if s.get("status") == "failed"],
                "errors": r["errors"],
            }
        )
    for g in groups:
        for name, key in (("mAP", "HL-mAP"), ("Hit1", "HL-Hit1")):
            expected = report["groups"][g]["metrics"]["HL-min-VeryGood"][key]
            if round(float(np.mean([m[g][name] for m in metrics.values()])), 2) != expected:
                raise ValueError("Per-query metric disagrees with frozen official evaluator")
    cue_counts, references, edges, nodes, rejected, cross_edges = Counter(), 0, 0, 0, 0, 0
    for p in (root / "memories").glob("*/*.json"):
        chunk = json.loads(p.read_text())
        edges += len(chunk["links"])
        rejected += len(chunk.get("rejected_links", []))
        cross_edges += sum(
            edge["source"].split(":")[0] != edge["target"].split(":")[0] for edge in chunk["links"]
        )
        nodes += len(chunk["events"])
        for e in chunk["events"]:
            for v in e["values"]:
                cue_counts[(v["dimension"], v["status"])] += 1
                references += len(v["reference_event_ids"])
    result = {
        "queries": len(truth),
        "source_run": str(root),
        "protocol_id": report["protocol_id"],
        "bootstrap": {"seed": 42, "replicates": 10000, "cluster": "source video", "ci": "percentile95%"},
        "paired_differences": {
            a + "_minus_" + b: paired(metrics, records, a, b)
            for a, b in (("E1", "D0"), ("E2", "E1"), ("E3", "E2"), ("E3", "D0"))
        },
        "all_groups_ok": sum(all(r[g]["status"] == "ok" for g in groups) for r in records.values()),
        "completed_E3_without_fallback": sum(r["E3"]["status"] != "fallback" for r in records.values()),
        "completed_memories_with_omitted_links": sum(
            r.get("rejected_memory_links", 0) > 0 for r in records.values()
        ),
        "queries_with_views": sum(r["views"] > 0 for r in details),
        "queries_with_no_check_requested": sum(
            "acquisition" in r and not r["acquisition"]["planned_needs"] for r in records.values()
        ),
        "queries_with_all_planned_checks_structurally_resolved": sum(
            bool(r["needs"]) and not r["gaps"] for r in details
        ),
        "total_new_frames": sum(r["new_frames"] for r in details),
        "failed_views": sum(len(r["view_failures"]) for r in details),
        "top_tied_queries": {g: sum(r["top_ties"][g] > 1 for r in details) for g in groups},
        "changed_bins": {g: sum(len(r["changed_bins_vs_D0"][g]) for r in details) for g in groups},
        "mean_selected_positive_recall": float(
            np.mean(
                [r["selected_positive_recall"] for r in details if r["selected_positive_recall"] is not None]
            )
        ),
        "memory": {
            "scope": "all successfully built chunks, including queries that later fell back",
            "nodes": nodes,
            "accepted_edges": edges,
            "cross_window_edges": cross_edges,
            "rejected_edges": rejected,
            "reference_event_citations": references,
            "cue_statuses": {k[0] + ":" + k[1]: v for k, v in cue_counts.items()},
        },
        "source_sha256": hashes,
        "limitations": report["limitations"]
        + ["Bootstrap excludes model/systematic bias; this small explored subset is not confirmatory"],
    }
    for name, expected in hashes.items():
        if file_sha256(root / name) != expected:
            raise ValueError("Source changed during read-only analysis")
    write_json(output / "analysis.json", result)
    write_json(output / "per_query.json", details)
    print(json.dumps({k: result[k] for k in ("queries", "paired_differences", "memory")}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.run_root.resolve(), args.output.resolve())
