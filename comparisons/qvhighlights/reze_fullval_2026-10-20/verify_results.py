"""Verify published predictions and HD statistics on CPU, without model inference."""

import argparse
import hashlib
import importlib.metadata
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

BUNDLE = Path(__file__).resolve().parent
COMMON = BUNDLE.parent / "2026-10-15_194620/common"
ARMS = ("B0", "P_abs", "P_pair")


def read_json(path):
    return json.loads(path.read_text())


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def paired_interval(difference, groups, repeats, seed):
    """Match the frozen query-weighted source-video bootstrap."""
    _, inverse = np.unique(groups, return_inverse=True)
    sizes = np.bincount(inverse)
    sums = np.stack([np.bincount(inverse, weights=difference[:, col]) for col in range(2)], axis=1)
    rng = np.random.default_rng(seed)
    batches = []
    for begin in range(0, repeats, 250):
        indices = rng.integers(0, len(sizes), size=(min(250, repeats - begin), len(sizes)))
        batches.append(sums[indices].sum(axis=1) / sizes[indices].sum(axis=1, keepdims=True))
    bounds = np.percentile(np.concatenate(batches), [2.5, 97.5], axis=0)
    return {
        "delta_percentage_points": dict(zip(("mAP", "Hit1"), difference.mean(axis=0).tolist(), strict=True)),
        "ci95_percentage_points": dict(zip(("mAP", "Hit1"), bounds.T.tolist(), strict=True)),
        "query_map_win_tie_loss": [
            int(np.sum(difference[:, 0] > 1e-10)),
            int(np.sum(np.abs(difference[:, 0]) <= 1e-10)),
            int(np.sum(difference[:, 0] < -1e-10)),
        ],
    }


def verify():
    publication = read_json(BUNDLE / "publication.json")
    entries = publication["files"]
    require(len({item["path"] for item in entries}) == len(entries), "Duplicate publication path")
    for item in entries:
        path = BUNDLE / item["path"]
        require(sha256(path) == item["public_sha256"], f"Public hash mismatch: {item['path']}")
        require(path.stat().st_size == item["bytes"], f"Size mismatch: {item['path']}")
    source_hashes = {item["path"]: item["source_sha256"] for item in entries}

    evaluation = read_json(BUNDLE / "fullval/results/evaluation.json")
    plan = read_json(BUNDLE / "fullval/run/plan.json")
    completion = read_json(BUNDLE / "fullval/run/completion.json")
    provenance = evaluation["provenance"]
    labels_path = COMMON / "ground_truth_val.jsonl"
    require(sha256(labels_path) == provenance["labels_sha256"], "Validation labels changed")
    for name, digest in provenance["evaluator_sha256"].items():
        require(sha256(COMMON / "evaluator/standalone_eval" / name) == digest, "Evaluator changed")
    require(
        provenance["binding_sha256"] == source_hashes["fullval/run/binding.json"],
        "Original binding hash mismatch",
    )
    require(
        provenance["plan_sha256"] == source_hashes["fullval/run/plan.json"],
        "Original plan hash mismatch",
    )
    require(
        provenance["baseline_completion_sha256"] == source_hashes["references/reze_val_v2/completion.json"],
        "Original baseline completion hash mismatch",
    )
    require(
        completion["record_receipt_sha256"] == source_hashes["fullval/run/record_receipt.json"],
        "Original record receipt hash mismatch",
    )
    sys.path.insert(0, str(COMMON / "evaluator"))
    from standalone_eval.eval import compute_ap_from_tuple, eval_submission, mk_gt_scores

    labels = read_jsonl(labels_path)
    qids = [row["qid"] for row in labels]
    require(len(qids) == len(set(qids)) == evaluation["queries"] == 1550, "Invalid labels")
    require(len({row["vid"] for row in labels}) == plan["video_segments"] == 1519, "Segment count")
    groups = np.array([row["vid"].rsplit("_", 2)[0] for row in labels])
    require(len(set(groups)) == evaluation["source_videos"] == 474, "Source-video count")

    paths = {arm: BUNDLE / f"fullval/results/{arm}_predictions.jsonl" for arm in ARMS}
    expected = {arm: evaluation["results"][arm]["metrics"] for arm in ARMS}
    for method in ("clip_val", "vtg_gpt_val"):
        paths[method] = BUNDLE / f"references/{method}/predictions.jsonl"
        report = read_json(BUNDLE / f"references/{method}/evaluation_report.json")
        require(sha256(paths[method]) == report["predictions_sha256"], f"{method} source hash")
        require(report["label_sha256"] == provenance["labels_sha256"], f"{method} labels")
        expected[method] = report["metrics"]
    require(sha256(paths["B0"]) == plan["baseline_prediction_sha256"], "B0 cache changed")

    predictions, values, metrics = {}, {}, {}
    per_query = read_jsonl(BUNDLE / "fullval/results/per_query.jsonl")
    require([row["qid"] for row in per_query] == qids, "Per-query order mismatch")
    require([row["vid"] for row in per_query] == [row["vid"] for row in labels], "Per-query videos")
    for method, path in paths.items():
        rows = read_jsonl(path)
        require([row["qid"] for row in rows] == qids, f"{method}: incomplete or reordered queries")
        for row, gt in zip(rows, labels, strict=True):
            require(all(row[key] == gt[key] for key in ("qid", "vid", "query")), f"{method}: identity")
            scores = row["pred_saliency_scores"]
            require(len(scores) == int(gt["duration"] / 2), f"{method}: score length")
            require(np.isfinite(scores).all(), f"{method}: non-finite scores")
        predictions[method] = rows
        # VTG-GPT also archived MR candidates; this publication verifies HD only.
        hd_rows = [{key: row[key] for key in ("qid", "vid", "query", "pred_saliency_scores")} for row in rows]
        metrics[method] = eval_submission(hd_rows, labels, verbose=False, match_number=True)
        require(metrics[method] == expected[method], f"{method}: official metrics mismatch")
        if method not in ARMS:
            continue
        computed = []
        for row, gt in zip(rows, labels, strict=True):
            scores = np.asarray(row["pred_saliency_scores"])
            binary = (mk_gt_scores(gt) >= 4).astype(float)
            ap = np.mean(
                [compute_ap_from_tuple((0, worker, binary[:, worker], scores))[2] for worker in range(3)]
            )
            computed.append([100 * ap, 100 * binary[int(np.argmax(scores))].max()])
        values[method] = np.asarray(computed)
        stored = [[row[method]["mAP"], row[method]["Hit1"]] for row in per_query]
        np.testing.assert_allclose(values[method], stored, rtol=0, atol=1e-10)
        np.testing.assert_allclose(
            values[method].mean(axis=0),
            [evaluation["results"][method]["unrounded_VeryGood"][key] for key in ("mAP", "Hit1")],
            rtol=0,
            atol=1e-10,
        )
        tied = sum(row["pred_saliency_scores"].count(max(row["pred_saliency_scores"])) > 1 for row in rows)
        require(tied == evaluation["results"][method]["tied_max_queries"], f"{method}: ties")
        print(f"Verified {method}: 1550 queries, official HD and per-query scores", file=sys.stderr)

    comparisons = {}
    for left, right in (("P_abs", "B0"), ("P_pair", "B0"), ("P_pair", "P_abs")):
        key = f"{left}_minus_{right}"
        comparisons[key] = paired_interval(
            values[left] - values[right],
            groups,
            evaluation["bootstrap"]["resamples"],
            evaluation["bootstrap"]["seed"],
        )
        for field in ("delta_percentage_points", "ci95_percentage_points"):
            for metric in ("mAP", "Hit1"):
                np.testing.assert_allclose(
                    comparisons[key][field][metric],
                    evaluation["paired_comparisons"][key][field][metric],
                    rtol=0,
                    atol=1e-10,
                )
        require(
            comparisons[key]["query_map_win_tie_loss"]
            == evaluation["paired_comparisons"][key]["query_map_win_tie_loss"],
            f"{key}: win/tie/loss mismatch",
        )

    changed = Counter()
    for index, baseline in enumerate(predictions["B0"]):
        original = baseline["pred_saliency_scores"]
        candidates = set(sorted(range(len(original)), key=lambda i: (-original[i], i))[:5])
        for arm in ("P_abs", "P_pair"):
            scores = predictions[arm][index]["pred_saliency_scores"]
            require(sorted(original) == sorted(scores), f"{arm}: score multiset changed")
            require(
                all(
                    a == b
                    for i, (a, b) in enumerate(zip(original, scores, strict=True))
                    if i not in candidates
                ),
                f"{arm}: non-candidate score changed",
            )
            changed[arm] += scores != original
    require(dict(changed) == evaluation["actual_scores_changed"], "Changed-query counts mismatch")

    attempts = read_jsonl(BUNDLE / "fullval/run/attempts_0.jsonl")
    require({row["event"] for row in attempts} == {"reserved", "returned"}, "Unexpected attempt event")
    for event in ("reserved", "returned"):
        rows = [row for row in attempts if row["event"] == event]
        require([row["qid"] for row in rows] == qids, f"{event}: query ledger mismatch")
        require(all(row["requests"] == 40 for row in rows), f"{event}: per-query call count")
        require(sum(row["requests"] for row in rows) == 62000, f"{event}: total call count")
    returned = [row for row in attempts if row["event"] == "returned"]
    for key in ("requests", "input_tokens", "output_tokens"):
        require(
            sum(row[key] for row in returned) == evaluation["new_returned_attempt_cost"][key],
            f"Returned {key} mismatch",
        )
    require(completion["status"] == "complete" and completion["queries"] == 1550, "Incomplete run")
    require(completion["attempted_requests"] == evaluation["new_attempted_requests"] == 62000, "Calls")
    require(completion["worker_restarts"] == evaluation["worker_restarts"] == {"0": 0}, "Restarts")
    require(evaluation["fallbacks"] == {arm: [] for arm in ARMS}, "Unexpected recorded fallback")
    for arm in ARMS:
        require(evaluation["results"][arm]["fallback_queries"] == 0, "Unexpected fallback count")
    for key in ("input_tokens", "output_tokens"):
        require(
            sum(evaluation["results"][arm]["cost"][key] for arm in ("P_abs", "P_pair"))
            == evaluation["new_returned_attempt_cost"][key],
            f"Arm token totals disagree: {key}",
        )
    baseline_cost = evaluation["results"]["B0"]["cost"]
    baseline_tokens = baseline_cost["input_tokens"] + baseline_cost["output_tokens"]
    for arm in ARMS:
        result = evaluation["results"][arm]
        tokens = result["cost"]["input_tokens"] + result["cost"]["output_tokens"]
        cold = tokens if arm == "B0" else baseline_tokens + tokens
        require(cold == result["cold_total_tokens"], f"{arm}: cold tokens")
        np.testing.assert_allclose(cold / baseline_tokens, result["cold_total_token_ratio"])
    supervisor = read_json(BUNDLE / "fullval/run/supervisor.json")
    np.testing.assert_allclose(
        supervisor["allocated_gpu_seconds"] / 3600,
        [completion["allocated_gpu_hours"], evaluation["new_allocated_gpu_hours"]],
        rtol=0,
        atol=1e-10,
    )
    stage_a = read_json(BUNDLE / "stage_a/evaluation.json")
    require(
        stage_a["gate"]["decision"]
        == evaluation["prior_stage_A_decision"]
        == "stop_v1_no_qualifying_stage_A_arm",
        "Stage-A stop decision changed",
    )
    return {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "scope": "CPU publication verification; no new model inference",
        "public_files_verified": len(entries),
        "byte_identical_source_files": sum(
            item["source_sha256"] == item["public_sha256"] for item in entries
        ),
        "labels_and_official_evaluator_hashes_match": True,
        "all_five_fullval_predictions_have_1550_unique_matching_queries": True,
        "official_hd_metrics_recomputed": metrics,
        "reze_per_query_scores_and_all_three_paired_intervals_match": True,
        "recomputed_paired_comparisons": comparisons,
        "top_five_score_permutation_verified": True,
        "actual_scores_changed": dict(changed),
        "attempt_ledger_and_recorded_costs_consistent": True,
        "stage_a_stop_decision_preserved": True,
        "environment": {
            "python": sys.version.split()[0],
            "packages": {
                name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn")
            },
        },
        "verifier_sha256": sha256(Path(__file__)),
        "requirements_sha256": sha256(BUNDLE / "requirements-eval.txt"),
        "limitations": [
            "Original per-request logits, video pixels and complete raw records are not in this bundle.",
            "Source hash receipts identify original files, including files transformed for publication.",
            "Fallback counts and frame-evidence audits are historical records, not reaudited model outputs.",
            "Stage-A training metrics were not reevaluated; only its file hashes and stop decision were checked.",
            "Shared-batch timing does not measure independent arm latency.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write a new JSON receipt; refuse to overwrite")
    args = parser.parse_args()
    if args.output is not None:
        require(not args.output.exists(), "Output already exists; choose a new receipt path")
    result = verify()
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        with args.output.open("x") as stream:
            stream.write(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
