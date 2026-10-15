#!/usr/bin/env python3
"""Strict validation wrapper around the official QVHighlights evaluator."""

import argparse
import json
import math
import sys
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate_predictions(predictions: list[dict], ground_truth_all: list[dict]) -> list[dict]:
    gt_by_qid = {x["qid"]: x for x in ground_truth_all}
    if len({x["qid"] for x in predictions}) != len(predictions):
        raise ValueError("Duplicate qid in predictions")
    ground_truth = []
    for prediction in predictions:
        qid = prediction["qid"]
        if qid not in gt_by_qid:
            raise ValueError(f"Unknown qid {qid}")
        gt = gt_by_qid[qid]
        if prediction["vid"] != gt["vid"] or prediction["query"] != gt["query"]:
            raise ValueError(f"Metadata mismatch for qid {qid}")
        expected = int(gt["duration"] / 2)
        scores = prediction.get("pred_saliency_scores")
        if not isinstance(scores, list) or len(scores) != expected:
            actual = len(scores) if isinstance(scores, list) else type(scores)
            raise ValueError(f"qid {qid}: expected {expected} scores, got {actual}")
        if not all(not isinstance(x, bool) and isinstance(x, (int, float)) and math.isfinite(x) for x in scores):
            raise ValueError(f"qid {qid}: non-finite score")
        ground_truth.append(gt)
    return ground_truth


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--official-repo", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    sys.path.insert(0, args.official_repo)
    from standalone_eval.eval import eval_highlight

    predictions = load_jsonl(Path(args.predictions))
    ground_truth_all = load_jsonl(Path(args.ground_truth))
    ground_truth = validate_predictions(predictions, ground_truth_all)
    metrics = eval_highlight(predictions, ground_truth, verbose=False)
    Path(args.output).write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
