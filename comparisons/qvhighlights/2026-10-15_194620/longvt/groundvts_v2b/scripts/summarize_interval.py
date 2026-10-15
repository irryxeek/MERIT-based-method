#!/usr/bin/env python3
"""Official QVHighlights metrics, paired cluster bootstrap, and generation diagnostics."""

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
BASE_SCRIPTS = ROOT / "reproduction/qvhighlights/scripts"
sys.path.insert(0, str(BASE_SCRIPTS))
from evaluate_highlight import validate_predictions  # noqa: E402


THRESHOLDS = {"Fair": 2, "Good": 3, "VeryGood": 4}
TRANSIENT = {"timeout", "request_failure", "oom", "context_overflow"}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def traces_cost(traces: list[dict]) -> dict:
    round_rows = [round_row for trace in traces for round_row in trace.get("rounds", [])]
    elapsed = [row.get("elapsed_seconds", 0.0) for row in traces]
    sorted_elapsed = sorted(elapsed)
    p95 = sorted_elapsed[max(0, int(np.ceil(0.95 * len(sorted_elapsed))) - 1)] if sorted_elapsed else None
    prompt_tokens = sum((row.get("usage") or {}).get("prompt_tokens", 0) for row in round_rows)
    completion_tokens = sum((row.get("usage") or {}).get("completion_tokens", 0) for row in round_rows)
    executions = [call for trace in traces for call in trace.get("tool_calls", [])]
    calls_with_frames = [call for call in executions if call.get("success")]
    score_counts = [
        {str(score): trace["pred_saliency_scores"].count(float(score)) for score in range(5)}
        for trace in traces if trace.get("pred_saliency_scores")
    ]
    top_ties = []
    unique_counts = []
    for trace in traces:
        scores = trace.get("pred_saliency_scores", [])
        if scores:
            top_ties.append(sum(score == max(scores) for score in scores))
            unique_counts.append(len(set(scores)))
    status_counts = {status: sum(row.get("status") == status for row in traces) for status in sorted({row.get("status") for row in traces})}
    return {
        "samples": len(traces),
        "status_counts": status_counts,
        "valid_model_predictions": sum(bool(row.get("prediction_valid")) for row in traces),
        "fallback_count": sum(bool(row.get("fallback_applied")) for row in traces),
        "format_correction_used": sum(any(r.get("stage") == "format_correction" for r in row.get("rounds", [])) for row in traces),
        "all_zero_vectors": sum(bool(row.get("pred_saliency_scores")) and not any(row["pred_saliency_scores"]) for row in traces),
        "constant_vectors": sum(len(set(row.get("pred_saliency_scores", []))) <= 1 for row in traces),
        "mean_unique_score_count": float(np.mean(unique_counts)) if unique_counts else None,
        "mean_top_tie_clip_count": float(np.mean(top_ties)) if top_ties else None,
        "score_frequency_per_query": score_counts,
        "tool_calls": len(executions),
        "tool_successes": len(calls_with_frames),
        "samples_with_tool_calls": sum(bool(row.get("tool_calls")) for row in traces),
        "global_frames": sum(row.get("global_frame_count", 0) for row in traces),
        "local_frames": sum(row.get("local_frame_count", 0) for row in traces),
        "total_read_frames": sum(row.get("total_read_frame_count", 0) for row in traces),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_elapsed_seconds": sum(elapsed),
        "mean_elapsed_seconds": statistics.mean(elapsed) if elapsed else None,
        "median_elapsed_seconds": statistics.median(elapsed) if elapsed else None,
        "p95_elapsed_seconds": p95,
        "peak_gpu_memory_mib": {
            gpu: max((row.get("peak_gpu_memory_mib", {}).get(gpu, 0) for row in traces), default=0)
            for gpu in ("6", "7")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--global-dir", required=True)
    parser.add_argument("--tool-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--official-repo", default="/path/to/storage/longvt-repro/benchmark_sources/moment_detr")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()

    sys.path.insert(0, args.official_repo)
    from standalone_eval.eval import eval_highlight, mk_gt_scores
    from standalone_eval.utils import get_ap

    gt_rows = read_jsonl(Path(args.ground_truth))
    gt_by_qid = {row["qid"]: row for row in gt_rows}
    manifest = json.loads(Path(args.manifest).read_text())
    manifest_by_qid = {row["qid"]: row for row in manifest}
    global_dir, tool_dir = Path(args.global_dir), Path(args.tool_dir)
    gpreds, tpreds = read_jsonl(global_dir / "predictions.jsonl"), read_jsonl(tool_dir / "predictions.jsonl")
    gtraces, ttraces = read_jsonl(global_dir / "traces.jsonl"), read_jsonl(tool_dir / "traces.jsonl")
    for condition, preds, traces in (("global", gpreds, gtraces), ("tool", tpreds, ttraces)):
        if len(preds) != len(manifest) or len(traces) != len(manifest):
            raise ValueError(f"{condition}: full manifest coverage is incomplete")
        if {row["qid"] for row in preds} != set(manifest_by_qid) or {row["qid"] for row in traces} != set(manifest_by_qid):
            raise ValueError(f"{condition}: qid set mismatch")
        validate_predictions(preds, gt_rows)
        if any(row.get("status") in TRANSIENT for row in traces):
            raise ValueError(f"{condition}: unresolved infrastructure failure remains")

    g_by_qid = {row["qid"]: row for row in gpreds}
    t_by_qid = {row["qid"]: row for row in tpreds}
    gt_ordered = [gt_by_qid[row["qid"]] for row in manifest]
    g_ordered = [g_by_qid[row["qid"]] for row in manifest]
    t_ordered = [t_by_qid[row["qid"]] for row in manifest]
    official_metrics = {
        "global": eval_highlight(g_ordered, gt_ordered, verbose=False),
        "tool": eval_highlight(t_ordered, gt_ordered, verbose=False),
    }
    deltas = {
        threshold: {
            metric: official_metrics["tool"][threshold][metric] - official_metrics["global"][threshold][metric]
            for metric in ("HL-mAP", "HL-Hit1")
        }
        for threshold in official_metrics["global"]
    }

    per_query = []
    values = {mode: {name: {metric: [] for metric in ("HL-mAP", "HL-Hit1")} for name in THRESHOLDS} for mode in ("global", "tool")}
    for row in manifest:
        qid = row["qid"]
        gt_full = mk_gt_scores(gt_by_qid[qid])
        per_mode = {}
        for mode, prediction in (("global", g_by_qid[qid]), ("tool", t_by_qid[qid])):
            pred_scores = np.asarray(prediction["pred_saliency_scores"], dtype=float)
            argmax_idx = int(np.argmax(pred_scores))
            metric_map = {}
            for name, threshold in THRESHOLDS.items():
                binary = gt_full >= threshold
                ap = float(np.mean([
                    get_ap(binary[:, worker].astype(float), pred_scores)
                    for worker in range(binary.shape[1])
                ]))
                hit = float(bool(binary[argmax_idx].any()))
                values[mode][name]["HL-mAP"].append(ap)
                values[mode][name]["HL-Hit1"].append(hit)
                metric_map[f"HL-min-{name}"] = {"HL-mAP": ap, "HL-Hit1": hit}
            per_mode[mode] = metric_map
        per_query.append({
            "qid": qid,
            "source_video_id": row.get("source_video_id", row.get("youtube_id", row["vid"].split("_")[0])),
            "global": per_mode["global"], "tool": per_mode["tool"],
            "tool_minus_global": {
                threshold: {
                    metric: per_mode["tool"][threshold][metric] - per_mode["global"][threshold][metric]
                    for metric in ("HL-mAP", "HL-Hit1")
                }
                for threshold in per_mode["global"]
            },
        })

    video_ids = sorted({row["source_video_id"] for row in per_query})
    video_to_index = {video_id: idx for idx, video_id in enumerate(video_ids)}
    q_video_indices = np.asarray([video_to_index[row["source_video_id"]] for row in per_query])
    rng = np.random.RandomState(42)
    bootstrap_weights = np.zeros((args.bootstrap, len(per_query)), dtype=np.float64)
    for b in range(args.bootstrap):
        sampled = rng.randint(0, len(video_ids), size=len(video_ids))
        counts = np.bincount(sampled, minlength=len(video_ids))
        bootstrap_weights[b] = counts[q_video_indices]
    bootstrap = {}
    for name in THRESHOLDS:
        metric_key = f"HL-min-{name}"
        bootstrap[metric_key] = {}
        for metric in ("HL-mAP", "HL-Hit1"):
            diff = np.asarray(values["tool"][name][metric]) - np.asarray(values["global"][name][metric])
            raw_mean = float(np.mean(diff) * 100)
            replicates = (bootstrap_weights @ diff) / bootstrap_weights.sum(axis=1) * 100
            bootstrap[metric_key][metric] = {
                "mean_tool_minus_global": raw_mean,
                "ci95_cluster_source_video": [float(np.quantile(replicates, 0.025)), float(np.quantile(replicates, 0.975))],
                "bootstrap_replicates": args.bootstrap,
                "seed": 42,
                "cluster_count": len(video_ids),
                "sampling_unit": "source_video; query-weighted; duplicate sampled clusters retain repeated weight",
            }

    gtrace_by_qid = {row["qid"]: row for row in gtraces}
    ttrace_by_qid = {row["qid"]: row for row in ttraces}
    mismatched_frames = [
        qid for qid in manifest_by_qid
        if gtrace_by_qid[qid].get("global_frame_sha256") != ttrace_by_qid[qid].get("global_frame_sha256")
    ]
    native_valid = {
        mode: sum(bool(row.get("prediction_valid")) for row in traces)
        for mode, traces in (("global", gtraces), ("tool", ttraces))
    }
    result = {
        "experiment": "GroundVTS-inspired training-free interval inference adaptation",
        "query_count": len(manifest), "unique_source_videos": len(video_ids),
        "official_metrics": official_metrics, "tool_minus_global_official_metric_delta": deltas,
        "paired_cluster_bootstrap": bootstrap,
        "native_prediction_valid_count": native_valid,
        "frame_identity": {"identical_global_frames": len(manifest) - len(mismatched_frames), "mismatched_qids": mismatched_frames},
        "costs": {"global": traces_cost(gtraces), "tool": traces_cost(ttraces)},
        "notes": [
            "Official evaluator computes Hit@1 with numpy.argmax first-maximum semantics.",
            "Official validation was previously inspected in the frozen score-vector experiment.",
            "This interval output, positive-overlap mapping, and two-stage generation differ from the old score-vector protocol.",
            "Bootstrap uses per-query metrics from the official get_ap and mk_gt_scores routines; intervals are paired by source video.",
        ],
        "per_query": per_query,
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "paired_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    with (out_dir / "per_query.jsonl").open("w") as handle:
        for row in per_query:
            handle.write(json.dumps(row) + "\n")
    with (out_dir / "per_query.csv").open("w", newline="") as handle:
        fields = ["qid", "source_video_id"] + [
            f"{name}_{metric}_tool_minus_global" for name in THRESHOLDS for metric in ("HL-mAP", "HL-Hit1")
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in per_query:
            flat = {"qid": row["qid"], "source_video_id": row["source_video_id"]}
            for name in THRESHOLDS:
                for metric in ("HL-mAP", "HL-Hit1"):
                    flat[f"{name}_{metric}_tool_minus_global"] = row["tool_minus_global"][f"HL-min-{name}"][metric]
            writer.writerow(flat)
    print(json.dumps({
        "query_count": len(manifest), "official_metrics": official_metrics,
        "tool_minus_global": deltas, "mismatched_frames": len(mismatched_frames),
        "valid_predictions": native_valid,
    }, indent=2))


if __name__ == "__main__":
    main()
