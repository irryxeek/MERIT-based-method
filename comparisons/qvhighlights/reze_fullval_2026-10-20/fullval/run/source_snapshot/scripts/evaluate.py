"""Strict coverage validation followed by the frozen official HD evaluator."""

import argparse
import json
import sys
from pathlib import Path

from common import ROOT, atomic_json, read_jsonl, sha256, utc_now, validate_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = json.loads((ROOT / "configs/protocol_v1.json").read_text())
    audit = json.loads((ROOT / "manifests/data_audit.json").read_text())
    for name, digest in audit["evaluator_hashes"].items():
        assert sha256(Path(config["evaluator_root"]) / "standalone_eval" / name) == digest
    assert sha256(config["labels_path"]) == audit["label_sha256"]
    manifest = read_jsonl(ROOT / "manifests" / ("smoke_inference.jsonl" if args.smoke else "val_inference.jsonl"))
    expected = {r["qid"]: r for r in manifest}
    predictions = read_jsonl(args.run / "predictions.jsonl")
    assert len(predictions) == len(expected)
    assert {r["qid"] for r in predictions} == set(expected)
    for pred in predictions:
        row = expected[pred["qid"]]
        assert pred["vid"] == row["vid"] and pred["query"] == row["query"]
        validate_scores(pred["pred_saliency_scores"], int(row["duration"] / 2))
    labels = [r for r in read_jsonl(config["labels_path"]) if r["qid"] in expected]
    sys.path.insert(0, config["evaluator_root"])
    from standalone_eval.eval import eval_submission

    # This experiment's prespecified primary task is HD.
    hd = [{k: p[k] for k in ["qid", "vid", "query", "pred_saliency_scores"]} for p in predictions]
    metrics = eval_submission(hd, labels, verbose=False, match_number=True)
    if (args.run / "raw_predictions.jsonl").exists():
        raw = read_jsonl(args.run / "raw_predictions.jsonl")
        raw_hd = [{k: p[k] for k in ["qid", "vid", "query", "pred_saliency_scores"]} for p in raw]
        raw_metrics = eval_submission(raw_hd, labels, verbose=False, match_number=True)
        assert metrics == raw_metrics, "Explicit tail padding changed official HD metrics"
    atomic_json(args.run / "metrics.json", metrics)
    report = {
        "evaluated_at": utc_now(),
        "queries": len(hd),
        "coverage": "full requested split",
        "split": "val_smoke" if args.smoke else "val",
        "metrics": metrics,
        "constant_score_queries": sum(len(set(p["pred_saliency_scores"])) == 1 for p in hd),
        "tied_max_queries": sum(p["pred_saliency_scores"].count(max(p["pred_saliency_scores"])) > 1 for p in hd),
        "predictions_sha256": sha256(args.run / "predictions.jsonl"),
        "label_sha256": audit["label_sha256"],
        "evaluator_hashes": audit["evaluator_hashes"],
    }
    atomic_json(args.run / "evaluation_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
