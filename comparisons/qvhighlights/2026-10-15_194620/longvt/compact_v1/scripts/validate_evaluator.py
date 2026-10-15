#!/usr/bin/env python3
"""Reproduce the official sample and run label-only evaluator direction checks."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-repo", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--synthetic-count", type=int, default=10)
    args = parser.parse_args()
    repo = Path(args.official_repo).resolve()
    sys.path.insert(0, str(repo))
    from standalone_eval.eval import eval_highlight, mk_gt_scores

    ground_truth = load_jsonl(Path(args.ground_truth))
    gt_by_qid = {row["qid"]: row for row in ground_truth}
    official_predictions = load_jsonl(repo / "standalone_eval" / "sample_val_preds.jsonl")
    official_gt = [gt_by_qid[row["qid"]] for row in official_predictions]
    length_mismatches = []
    for prediction, ground_truth_row in zip(official_predictions, official_gt, strict=True):
        expected = int(ground_truth_row["duration"] / 2)
        actual = len(prediction["pred_saliency_scores"])
        if actual != expected:
            length_mismatches.append({
                "qid": prediction["qid"], "vid": prediction["vid"],
                "expected": expected, "actual": actual,
            })
    official_metrics = eval_highlight(official_predictions, official_gt, verbose=False)
    expected_raw = json.loads((repo / "standalone_eval" / "sample_val_preds_metrics_raw.json").read_text())
    expected_highlight = {key: value for key, value in expected_raw.items() if key.startswith("HL-min-")}

    synthetic_gt = ground_truth[:args.synthetic_count]
    high_predictions = []
    reverse_predictions = []
    for row in synthetic_gt:
        mean_scores = np.mean(mk_gt_scores(row), axis=1)
        metadata = {key: row[key] for key in ("qid", "vid", "query")}
        high_predictions.append({**metadata, "pred_saliency_scores": mean_scores.tolist()})
        reverse_predictions.append({**metadata, "pred_saliency_scores": (-mean_scores).tolist()})
    high_metrics = eval_highlight(high_predictions, synthetic_gt, verbose=False)
    reverse_metrics = eval_highlight(reverse_predictions, synthetic_gt, verbose=False)
    direction_pass = all(
        high_metrics[key][metric] > reverse_metrics[key][metric]
        for key in high_metrics for metric in ("HL-mAP", "HL-Hit1")
    )
    result = {
        "official_sample": {
            "count": len(official_predictions), "metrics": official_metrics,
            "expected_highlight": expected_highlight, "exact_match": official_metrics == expected_highlight,
            "length_mismatches_accepted_by_official_evaluator": length_mismatches,
            "strict_adapter_policy": (
                "reject rather than use the official evaluator's silent truncate/zero-pad behavior"
            ),
        },
        "synthetic_label_only_direction_check": {
            "count": len(synthetic_gt), "ground_truth_mean_ranking": high_metrics,
            "reverse_ranking": reverse_metrics, "all_metrics_higher": direction_pass,
            "note": "Evaluator validation only; synthetic label-derived scores never enter model inference.",
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["official_sample"]["exact_match"] or not direction_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
