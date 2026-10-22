"""Validate complete disjoint shards and aggregate per-video official metrics."""

import argparse
import json
from collections import Counter
from pathlib import Path

from evaluate import bootstrap_mean
from shared import atomic_json, read_jsonl, sha256, utc_now, write_jsonl

METRIC_NAMES = (
    "AP50",
    "AP15",
    "summary_F1",
    "constant_AP50",
    "constant_AP15",
    "constant_summary_F1",
    "random_expected_AP50",
    "random_expected_AP15",
)
PROTOCOL_FIELDS = (
    "dataset",
    "query",
    "window_seconds",
    "frames_per_full_window",
    "sampling_fps",
    "readout",
    "timeline_policy",
    "model_path",
    "model_revision",
    "batch_size",
    "evaluator_revision",
    "evaluation_policy",
    "summary_policy",
    "labels_sha256",
    "baseline_code_sha256",
    "source_sha256",
)


def merge_details(expected_ids, shard_details):
    combined = [row for rows in shard_details for row in rows]
    ids = [row["vid"] for row in combined]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate video across shards")
    if len(expected_ids) != len(set(expected_ids)) or set(ids) != set(expected_ids):
        raise ValueError("Shard results do not cover the frozen video set exactly")
    by_id = {row["vid"]: row for row in combined}
    return [by_id[vid] for vid in expected_ids]


def aggregate_metrics(details):
    metrics = {name: bootstrap_mean([row[name] for row in details]) for name in METRIC_NAMES}
    for fraction in ("50", "15"):
        metrics[f"AP{fraction}_minus_random_expectation"] = bootstrap_mean(
            [row[f"AP{fraction}"] - row[f"random_expected_AP{fraction}"] for row in details]
        )
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    root = args.run.resolve()
    if (root / "complete.json").exists():
        return
    plan = json.loads((root / "plan.json").read_text())
    states = []
    for shard in plan["shards"]:
        run = root / shard["directory"]
        state = json.loads((run / "status.json").read_text()) if (run / "status.json").exists() else {}
        states.append(
            {
                "shard_index": shard["index"],
                "directory": shard["directory"],
                "state": state,
                "completed_videos": len(list((run / "predictions").glob("*.json"))),
                "expected_videos": len(shard["video_ids"]),
                "complete": (run / "complete.json").exists(),
                "stopped": (run / "stopped.json").exists(),
            }
        )
    if not all(s["complete"] for s in states):
        atomic_json(
            root / "status.json",
            {
                "status": "shard_needs_attention" if any(s["stopped"] for s in states) else "in_progress",
                "updated_at": utc_now(),
                "expected_queries": len(plan["selected_ids"]),
                "completed_queries": sum(s["completed_videos"] for s in states),
                "shards": states,
            },
        )
        return
    shard_details, shard_summaries, sources, evaluations, attempts, bindings = [], [], [], [], [], []
    for shard in plan["shards"]:
        run = root / shard["directory"]
        binding = json.loads((run / "binding.json").read_text())
        complete = json.loads((run / "complete.json").read_text())
        evaluation = json.loads((run / "evaluation.json").read_text())
        if complete["evaluation_sha256"] != sha256(run / "evaluation.json"):
            raise ValueError("Completed evaluation changed")
        if evaluation["binding_sha256"] != sha256(run / "binding.json"):
            raise ValueError("Evaluation binding changed")
        if evaluation["per_video_metrics_sha256"] != sha256(run / "per_video_metrics.jsonl"):
            raise ValueError("Per-video metrics changed")
        if evaluation["summary_predictions_sha256"] != sha256(run / "summary_predictions.jsonl"):
            raise ValueError("Summary predictions changed")
        if binding["selected_qids"] != shard["video_ids"] or binding["source_sha256"] != plan["source_sha256"]:
            raise ValueError("Shard membership or frozen source differs from full-run plan")
        for name, digest in binding["source_sha256"].items():
            if sha256(run / "source_snapshot" / name) != digest:
                raise ValueError("Frozen shard source changed")
        if bindings and any(binding[name] != bindings[0][name] for name in PROTOCOL_FIELDS):
            raise ValueError("Cannot aggregate shards with different inference/evaluation protocols")
        details = read_jsonl(run / "per_video_metrics.jsonl")
        summaries = read_jsonl(run / "summary_predictions.jsonl")
        merge_details(shard["video_ids"], [details])
        merge_details(shard["video_ids"], [summaries])
        if evaluation["evaluated_videos"] != len(details):
            raise ValueError("Shard evaluation coverage differs")
        shard_details.append(details)
        shard_summaries.append(summaries)
        evaluations.append(evaluation)
        bindings.append(binding)
        attempts.extend(json.loads((run / "worker_attempts.json").read_text()))
        sources.append(
            {
                "directory": shard["directory"],
                "binding_sha256": sha256(run / "binding.json"),
                "evaluation_sha256": sha256(run / "evaluation.json"),
            }
        )
    details = merge_details(plan["selected_ids"], shard_details)
    summaries = merge_details(plan["selected_ids"], shard_summaries)
    metrics = aggregate_metrics(details)
    cost = {key: sum(e["cost"][key] for e in evaluations) for key in evaluations[0]["cost"]}
    cost["inference_wall_seconds"] = max(a["started_epoch"] + a["seconds"] for a in attempts) - min(
        a["started_epoch"] for a in attempts
    )
    cost["parallel_gpu_limit"] = len(plan["shards"])
    diagnostic = {}
    for key in ("answers", "score_modes", "gt_boundary_tie_videos", "summary_length_differences"):
        counts = Counter()
        for evaluation in evaluations:
            counts.update(evaluation["diagnostics"][key])
        diagnostic[key] = dict(counts)
    diagnostic["constant_normalized_prediction_videos"] = sum(
        e["diagnostics"]["constant_normalized_prediction_videos"] for e in evaluations
    )
    write_jsonl(root / "per_video_metrics.jsonl", details)
    write_jsonl(root / "summary_predictions.jsonl", summaries)
    result = {
        "status": "complete",
        "finished_at": utc_now(),
        "scope": plan["scope"],
        "plan_sha256": sha256(root / "plan.json"),
        "evaluated_videos": len(details),
        "official_test_videos": plan["official_test_count"],
        "official_test_coverage": len(details) / plan["official_test_count"],
        "all_verified_media_selected": True,
        "excluded_status_counts": bindings[0]["excluded_status_counts"],
        "metrics_percent": metrics,
        "cost": cost,
        "diagnostics": diagnostic,
        "shards": sources,
        "evaluator_revision": bindings[0]["evaluator_revision"],
        "labels_sha256": bindings[0]["labels_sha256"],
        "per_video_metrics_sha256": sha256(root / "per_video_metrics.jsonl"),
        "summary_predictions_sha256": sha256(root / "summary_predictions.jsonl"),
        "not_full_test_result": len(details) != plan["official_test_count"],
        "not_original_REZE_paper_dataset_result": True,
    }
    atomic_json(root / "evaluation.json", result)
    lines = [
        "# REZE on MR.HiSum: all verified available test videos",
        "",
        f"Evaluated {len(details)}/{plan['official_test_count']} official test videos ({100 * result['official_test_coverage']:.2f}% coverage).",
        "Two disjoint GPU shards use the same Qwen3-VL-8B-Instruct REZE scorer and fixed query `Highlight of this video.`.",
        "",
        "| Method / control | mAP50 (%) | mAP15 (%) | Summary F1 (%) |",
        "|---|---:|---:|---:|",
        f"| REZE | {metrics['AP50']['mean']:.4f} | {metrics['AP15']['mean']:.4f} | {metrics['summary_F1']['mean']:.4f} |",
        f"| Constant scores | {metrics['constant_AP50']['mean']:.4f} | {metrics['constant_AP15']['mean']:.4f} | {metrics['constant_summary_F1']['mean']:.4f} |",
        f"| Uniform random ranking, exact expected AP | {metrics['random_expected_AP50']['mean']:.4f} | {metrics['random_expected_AP15']['mean']:.4f} | — |",
        "",
        "## Scope and evaluation",
        "",
        "The 100-video pilot is preserved separately. This run scores every verified video afresh, with no prompt or scoring changes.",
        "Pinned author helpers implement 5-second HD metrics and 15%-budget KTS/knapsack summary F1.",
        "All metrics and 10,000-draw video bootstrap intervals are recomputed from merged per-video results; shard averages are not averaged.",
        f"Excluded media: {result['excluded_status_counts']}. Missing videos are not assigned fabricated predictions.",
        "These are available-subset results, not complete official-test or original REZE MR.HiSum paper results.",
        "",
        "## Runtime",
        "",
        f"Formal requests: {cost['formal_requests']}; including smoke/recovery: {cost['charged_requests_including_smoke_and_failed_attempts']}.",
        f"Inference elapsed wall time: {cost['inference_wall_seconds'] / 3600:.2f} hours; cumulative GPU worker time: {cost['worker_seconds'] / 3600:.2f} GPU-hours.",
        "Preparation, native input checks and final CPU evaluation are outside inference wall time.",
        "",
        "Per-video metrics, confidence intervals, score diagnostics and source hashes are in `evaluation.json` and `per_video_metrics.jsonl`.",
        "",
    ]
    (root / "REPORT.md").write_text("\n".join(lines))
    atomic_json(
        root / "complete.json",
        {"status": "complete", "finished_at": utc_now(), "evaluation_sha256": sha256(root / "evaluation.json")},
    )
    atomic_json(
        root / "status.json",
        {
            "status": "complete",
            "updated_at": utc_now(),
            "completed_queries": len(details),
            "expected_queries": len(details),
            "metrics_percent": {k: metrics[k]["mean"] for k in ("AP50", "AP15", "summary_F1")},
        },
    )
    print(json.dumps({"evaluated_videos": len(details), "metrics_percent": metrics, "cost": cost}), flush=True)


if __name__ == "__main__":
    main()
