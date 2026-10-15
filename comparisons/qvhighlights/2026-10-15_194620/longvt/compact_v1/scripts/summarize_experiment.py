#!/usr/bin/env python3
"""Summarize paired QVHighlights runs with official highlight metrics."""

import argparse
import json
import statistics
import sys
from pathlib import Path

from evaluate_highlight import validate_predictions


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def costs(traces: list[dict]) -> dict:
    prompt_tokens = completion_tokens = 0
    tool_calls = tool_executions = tool_successes = 0
    for trace in traces:
        for row in trace.get("rounds", []):
            usage = row.get("usage") or {}
            prompt_tokens += usage.get("prompt_tokens", 0)
            completion_tokens += usage.get("completion_tokens", 0)
        tool_calls += len(trace.get("tool_calls", []))
        tool_executions += sum(
            bool(call.get("executed")) and not call.get("budget_rejected")
            for call in trace.get("tool_calls", [])
        )
        tool_successes += sum(bool(call.get("success")) for call in trace.get("tool_calls", []))
    elapsed = [trace.get("elapsed_seconds", 0) for trace in traces]
    frames = [trace.get("total_read_frame_count", 0) for trace in traces]
    sorted_elapsed = sorted(elapsed)
    p95_index = max(0, (95 * len(sorted_elapsed) + 99) // 100 - 1) if sorted_elapsed else 0
    return {
        "samples": len(traces),
        "status_counts": {
            status: sum(x.get("status") == status for x in traces)
            for status in sorted({x.get("status") for x in traces})
        },
        "valid_predictions": sum(bool(x.get("prediction_valid")) for x in traces),
        "completion_rate": sum(bool(x.get("prediction_valid")) for x in traces) / len(traces) if traces else 0,
        "fallback_predictions": sum(bool(x.get("fallback_applied")) for x in traces),
        "tool_requests": tool_calls, "tool_executions": tool_executions, "tool_successes": tool_successes,
        "tool_success_rate": tool_successes / tool_executions if tool_executions else None,
        "samples_with_tool_request": sum(bool(x.get("tool_calls")) for x in traces),
        "max_tool_executions_per_sample": max((sum(
            bool(call.get("executed")) and not call.get("budget_rejected")
            for call in trace.get("tool_calls", [])
        ) for trace in traces), default=0),
        "global_frames": sum(x.get("global_frame_count", 0) for x in traces),
        "local_frames": sum(x.get("local_frame_count", 0) for x in traces),
        "total_read_frames": sum(frames), "mean_read_frames": statistics.mean(frames) if frames else 0,
        "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
        "total_seconds": sum(elapsed), "mean_seconds": statistics.mean(elapsed) if elapsed else 0,
        "median_seconds": statistics.median(elapsed) if elapsed else 0,
        "p95_seconds": sorted_elapsed[p95_index] if sorted_elapsed else 0,
        "peak_gpu_memory_mib": {
            gpu: max((trace.get("peak_gpu_memory_mib", {}).get(gpu, 0) for trace in traces), default=0)
            for gpu in ("6", "7")
        },
    }


def trace_cost(trace: dict) -> dict:
    prompt_tokens = completion_tokens = 0
    for row in trace.get("rounds", []):
        usage = row.get("usage") or {}
        prompt_tokens += usage.get("prompt_tokens", 0)
        completion_tokens += usage.get("completion_tokens", 0)
    return {
        "status": trace.get("status"),
        "prediction_valid": bool(trace.get("prediction_valid")),
        "fallback_applied": bool(trace.get("fallback_applied")),
        "global_frames": trace.get("global_frame_count", 0),
        "local_frames": trace.get("local_frame_count", 0),
        "total_read_frames": trace.get("total_read_frame_count", 0),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "elapsed_seconds": trace.get("elapsed_seconds", 0),
        "format_retry_count": sum(
            bool(row.get("is_format_retry")) for row in trace.get("rounds", [])
        ),
        "tool_calls": [
            {
                "success": bool(call.get("success")),
                "arguments": call.get("arguments"),
                "execution": call.get("execution"),
                "returned_frame_count": call.get("returned_frame_count", 0),
                "frame_timestamps": call.get("frame_timestamps", []),
                "returned_frames_added_to_next_request": bool(
                    call.get("returned_frames_added_to_next_request")
                ),
            }
            for call in trace.get("tool_calls", [])
        ],
    }


def metric_delta(tool_metrics: dict, global_metrics: dict) -> dict:
    return {
        threshold: {
            metric: tool_metrics[threshold][metric] - global_metrics[threshold][metric]
            for metric in ("HL-mAP", "HL-Hit1")
        }
        for threshold in global_metrics
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--global-dir", required=True)
    parser.add_argument("--tool-dir", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--official-repo", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    sys.path.insert(0, args.official_repo)
    from standalone_eval.eval import eval_highlight

    gt_all = {x["qid"]: x for x in load_jsonl(Path(args.ground_truth))}
    global_predictions = load_jsonl(Path(args.global_dir) / "predictions.jsonl")
    tool_predictions = load_jsonl(Path(args.tool_dir) / "predictions.jsonl")
    global_traces = load_jsonl(Path(args.global_dir) / "traces.jsonl")
    tool_traces = load_jsonl(Path(args.tool_dir) / "traces.jsonl")
    global_by_qid = {x["qid"]: x for x in global_predictions}
    tool_by_qid = {x["qid"]: x for x in tool_predictions}
    global_trace_by_qid = {x["qid"]: x for x in global_traces}
    tool_trace_by_qid = {x["qid"]: x for x in tool_traces}
    if set(global_by_qid) != set(tool_by_qid):
        raise ValueError("Global and tool prediction qids differ")
    qids = list(global_by_qid)
    if set(global_trace_by_qid) != set(qids) or set(tool_trace_by_qid) != set(qids):
        raise ValueError("Prediction and trace qids differ")
    ground_truth_all = list(gt_all.values())
    ground_truth = validate_predictions(global_predictions, ground_truth_all)
    validate_predictions(tool_predictions, ground_truth_all)
    common_success = [
        qid for qid in qids
        if global_trace_by_qid[qid].get("prediction_valid") and tool_trace_by_qid[qid].get("prediction_valid")
    ]
    all_global = eval_highlight(global_predictions, ground_truth, verbose=False)
    all_tool = eval_highlight(tool_predictions, ground_truth, verbose=False)
    common_global = eval_highlight(
        [global_by_qid[q] for q in common_success],
        [gt_all[q] for q in common_success], verbose=False,
    ) if common_success else None
    common_tool = eval_highlight(
        [tool_by_qid[q] for q in common_success],
        [gt_all[q] for q in common_success], verbose=False,
    ) if common_success else None
    frame_mismatch_qids = [
        qid for qid in qids
        if global_trace_by_qid[qid].get("global_frame_sha256")
        != tool_trace_by_qid[qid].get("global_frame_sha256")
    ]
    result = {
        "all_with_fixed_fallback": {"global": all_global, "tool": all_tool},
        "all_tool_minus_global": metric_delta(all_tool, all_global),
        "common_success_count": len(common_success),
        "common_success": {"global": common_global, "tool": common_tool},
        "common_success_tool_minus_global": (
            metric_delta(common_tool, common_global) if common_success else None
        ),
        "global_frame_pair_check": {
            "identical_count": len(qids) - len(frame_mismatch_qids),
            "mismatch_count": len(frame_mismatch_qids),
            "mismatch_qids": frame_mismatch_qids,
        },
        "costs": {"global": costs(global_traces), "tool": costs(tool_traces)},
        "per_query": [],
    }
    for qid in qids:
        gt = [gt_all[qid]]
        global_metrics = eval_highlight([global_by_qid[qid]], gt, verbose=False)
        tool_metrics = eval_highlight([tool_by_qid[qid]], gt, verbose=False)
        deltas = metric_delta(tool_metrics, global_metrics)
        result["per_query"].append({
            "qid": qid, "global": global_metrics, "tool": tool_metrics,
            "tool_minus_global": deltas,
            "global_trace": trace_cost(global_trace_by_qid[qid]),
            "tool_trace": trace_cost(tool_trace_by_qid[qid]),
        })
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    summary_keys = (
        "all_with_fixed_fallback", "all_tool_minus_global", "common_success_count",
        "common_success", "common_success_tool_minus_global", "global_frame_pair_check", "costs",
    )
    print(json.dumps({key: result[key] for key in summary_keys}, indent=2))


if __name__ == "__main__":
    main()
