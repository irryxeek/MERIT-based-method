#!/usr/bin/env python3
"""Audit paired QVHighlights predictions, score ties, and tool evidence coverage."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from evaluate_highlight import validate_predictions

THRESHOLDS = {"Fair": 2, "Good": 3, "VeryGood": 4}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def full_ground_truth(row: dict, clip_seconds: float) -> np.ndarray:
    scores = np.zeros((int(row["duration"] / clip_seconds), 3), dtype=float)
    scores[np.asarray(row["relevant_clip_ids"], dtype=int)] = np.asarray(row["saliency_scores"], dtype=float)
    return scores


def prediction_ties(prediction: dict, ground_truth: dict, clip_seconds: float) -> dict:
    scores = np.asarray(prediction["pred_saliency_scores"], dtype=float)
    top_indices = np.flatnonzero(scores == scores.max())
    gt_scores = full_ground_truth(ground_truth, clip_seconds)
    result = {
        "unique_score_count": int(len(np.unique(scores))),
        "constant_prediction": bool(len(np.unique(scores)) == 1),
        "top_score": float(scores.max()),
        "top_tie_count": int(len(top_indices)),
        "official_argmax_clip": int(top_indices[0]),
        "top_tied_clips": top_indices.tolist(),
        "hit1_tie_diagnostics": {},
    }
    for name, threshold in THRESHOLDS.items():
        positive = (gt_scores >= threshold).any(axis=1)
        tied_hits = positive[top_indices]
        result["hit1_tie_diagnostics"][name] = {
            "official_first_argmax_hit": bool(tied_hits[0]),
            "uniform_tie_break_expected_hit": float(tied_hits.mean()),
            "any_top_tied_clip_hits": bool(tied_hits.any()),
            "all_top_tied_clips_hit": bool(tied_hits.all()),
        }
    return result


def interval_clip_indices(start: float, end: float, num_clips: int, clip_seconds: float) -> set[int]:
    covered = set()
    for index in range(num_clips):
        clip_end = min((index + 1) * clip_seconds, num_clips * clip_seconds)
        midpoint = (index * clip_seconds + clip_end) / 2
        if start <= midpoint < end:
            covered.add(index)
    return covered


def tool_evidence(trace: dict, ground_truth: dict, clip_seconds: float) -> dict:
    num_clips = int(ground_truth["duration"] / clip_seconds)
    gt_scores = full_ground_truth(ground_truth, clip_seconds)
    covered = set()
    calls = []
    for call in trace.get("tool_calls", []):
        execution = call.get("execution") or {}
        start = execution.get("executed_start_time")
        end = execution.get("executed_end_time")
        call_covered = set()
        if call.get("success") and isinstance(start, (int, float)) and isinstance(end, (int, float)):
            call_covered = interval_clip_indices(float(start), float(end), num_clips, clip_seconds)
            covered.update(call_covered)
        calls.append({
            "success": bool(call.get("success")),
            "executed_start_time": start,
            "executed_end_time": end,
            "covered_clip_indices": sorted(call_covered),
            "returned_frame_count": call.get("returned_frame_count", 0),
            "returned_frames_added_to_next_request": bool(call.get("returned_frames_added_to_next_request")),
        })
    relevant = set(map(int, ground_truth["relevant_clip_ids"]))
    result = {
        "tool_requested": bool(trace.get("tool_calls")),
        "tool_call_count": len(trace.get("tool_calls", [])),
        "calls": calls,
        "covered_clip_indices": sorted(covered),
        "covered_relevant_clip_indices": sorted(covered & relevant),
        "covers_any_released_relevant_clip": bool(covered & relevant),
        "coverage_by_threshold": {},
    }
    for name, threshold in THRESHOLDS.items():
        positive = set(np.flatnonzero((gt_scores >= threshold).any(axis=1)).tolist())
        intersection = covered & positive
        result["coverage_by_threshold"][name] = {
            "covered_positive_clip_indices": sorted(intersection),
            "covers_any_positive_clip": bool(intersection),
            "positive_clip_recall": len(intersection) / len(positive) if positive else None,
        }
    return result


def aggregate_ties(rows: list[dict], mode: str) -> dict:
    tie_rows = [row[f"{mode}_ties"] for row in rows]
    aggregate = {
        "samples": len(tie_rows),
        "constant_predictions": sum(row["constant_prediction"] for row in tie_rows),
        "unique_top_predictions": sum(row["top_tie_count"] == 1 for row in tie_rows),
        "mean_unique_score_count": float(np.mean([row["unique_score_count"] for row in tie_rows])),
        "mean_top_tie_count": float(np.mean([row["top_tie_count"] for row in tie_rows])),
        "max_top_tie_count": max(row["top_tie_count"] for row in tie_rows),
        "hit1_tie_diagnostics": {},
    }
    for name in THRESHOLDS:
        diagnostics = [row["hit1_tie_diagnostics"][name] for row in tie_rows]
        aggregate["hit1_tie_diagnostics"][name] = {
            "official_first_argmax_percent": 100 * np.mean(
                [row["official_first_argmax_hit"] for row in diagnostics]
            ),
            "uniform_tie_break_expected_percent": 100 * np.mean(
                [row["uniform_tie_break_expected_hit"] for row in diagnostics]
            ),
            "any_top_tied_clip_percent": 100 * np.mean(
                [row["any_top_tied_clip_hits"] for row in diagnostics]
            ),
            "all_top_tied_clips_percent": 100 * np.mean(
                [row["all_top_tied_clips_hit"] for row in diagnostics]
            ),
        }
    return aggregate


def metric_direction_counts(rows: list[dict]) -> dict:
    output = {}
    for threshold in THRESHOLDS:
        key = f"HL-min-{threshold}"
        output[key] = {}
        for metric in ("HL-mAP", "HL-Hit1"):
            values = [row["tool_minus_global"][key][metric] for row in rows]
            output[key][metric] = {
                "tool_better": sum(value > 0 for value in values),
                "tied": sum(math.isclose(value, 0.0, abs_tol=1e-12) for value in values),
                "tool_worse": sum(value < 0 for value in values),
                "mean_delta": float(np.mean(values)),
            }
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = ["qid", "query", "tool_requested", "tool_covers_relevant", "global_constant", "tool_constant"]
    for threshold in THRESHOLDS:
        for metric in ("HL-mAP", "HL-Hit1"):
            fields.append(f"{threshold}_{metric}_tool_minus_global")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = {
                "qid": row["qid"],
                "query": row["query"],
                "tool_requested": row["tool_evidence"]["tool_requested"],
                "tool_covers_relevant": row["tool_evidence"]["covers_any_released_relevant_clip"],
                "global_constant": row["global_ties"]["constant_prediction"],
                "tool_constant": row["tool_ties"]["constant_prediction"],
            }
            for threshold in THRESHOLDS:
                key = f"HL-min-{threshold}"
                for metric in ("HL-mAP", "HL-Hit1"):
                    flat[f"{threshold}_{metric}_tool_minus_global"] = row["tool_minus_global"][key][metric]
            writer.writerow(flat)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--global-dir", required=True)
    parser.add_argument("--tool-dir", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--clip-seconds", type=float, default=2.0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--csv-output", required=True)
    args = parser.parse_args()

    summary = json.loads(Path(args.summary).read_text())
    ground_truth_rows = load_jsonl(Path(args.ground_truth))
    ground_truth = {row["qid"]: row for row in ground_truth_rows}
    global_predictions = load_jsonl(Path(args.global_dir) / "predictions.jsonl")
    tool_predictions = load_jsonl(Path(args.tool_dir) / "predictions.jsonl")
    tool_traces = load_jsonl(Path(args.tool_dir) / "traces.jsonl")
    validate_predictions(global_predictions, ground_truth_rows)
    validate_predictions(tool_predictions, ground_truth_rows)
    global_by_qid = {row["qid"]: row for row in global_predictions}
    tool_by_qid = {row["qid"]: row for row in tool_predictions}
    trace_by_qid = {row["qid"]: row for row in tool_traces}
    paired_by_qid = {row["qid"]: row for row in summary["per_query"]}
    qids = list(global_by_qid)
    expected = set(qids)
    for name, values in {
        "tool predictions": tool_by_qid,
        "tool traces": trace_by_qid,
        "paired summary": paired_by_qid,
    }.items():
        if set(values) != expected:
            raise ValueError(f"{name} qids differ from global predictions")

    rows = []
    for qid in qids:
        rows.append({
            "qid": qid,
            "query": ground_truth[qid]["query"],
            "global_ties": prediction_ties(global_by_qid[qid], ground_truth[qid], args.clip_seconds),
            "tool_ties": prediction_ties(tool_by_qid[qid], ground_truth[qid], args.clip_seconds),
            "tool_evidence": tool_evidence(trace_by_qid[qid], ground_truth[qid], args.clip_seconds),
            "tool_minus_global": paired_by_qid[qid]["tool_minus_global"],
        })

    requested = [row for row in rows if row["tool_evidence"]["tool_requested"]]
    evidence_summary = {
        "samples_with_tool_request": len(requested),
        "samples_covering_any_released_relevant_clip": sum(
            row["tool_evidence"]["covers_any_released_relevant_clip"] for row in requested
        ),
        "coverage_by_threshold": {},
    }
    for name in THRESHOLDS:
        eligible = [
            row for row in requested
            if row["tool_evidence"]["coverage_by_threshold"][name]["positive_clip_recall"] is not None
        ]
        evidence_summary["coverage_by_threshold"][name] = {
            "eligible_tool_request_samples": len(eligible),
            "samples_covering_any_positive_clip": sum(
                row["tool_evidence"]["coverage_by_threshold"][name]["covers_any_positive_clip"]
                for row in eligible
            ),
            "mean_positive_clip_recall": float(np.mean([
                row["tool_evidence"]["coverage_by_threshold"][name]["positive_clip_recall"]
                for row in eligible
            ])) if eligible else None,
        }

    result = {
        "notes": {
            "official_hit1_tie_rule": "numpy.argmax selects the first maximum-scored clip",
            "official_map_tie_rule": "sklearn precision_recall_curve groups equal-score clips by threshold",
            "tool_evidence_coverage_rule": "official clip midpoint lies in a successful executed crop interval",
            "tie_diagnostics_are_not_official_metrics": True,
        },
        "integrity": {
            "paired_samples": len(rows),
            "identical_prediction_vectors": sum(
                global_by_qid[qid]["pred_saliency_scores"] == tool_by_qid[qid]["pred_saliency_scores"]
                for qid in qids
            ),
        },
        "per_question_metric_directions": metric_direction_counts(rows),
        "ties": {"global": aggregate_ties(rows, "global"), "tool": aggregate_ties(rows, "tool")},
        "tool_evidence_coverage": evidence_summary,
        "per_question": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    write_csv(Path(args.csv_output), rows)
    print(json.dumps({key: value for key, value in result.items() if key != "per_question"}, indent=2))


if __name__ == "__main__":
    main()
