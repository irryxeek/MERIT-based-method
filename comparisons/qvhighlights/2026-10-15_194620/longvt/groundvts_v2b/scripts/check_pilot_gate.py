#!/usr/bin/env python3
"""Check the predeclared engineering gate without selecting on dev metrics."""

import argparse
import json
from pathlib import Path


TRANSIENT = {"timeout", "request_failure", "oom", "context_overflow"}


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def inspect(run_dir: Path, expected: int) -> dict:
    traces = rows(run_dir / "traces.jsonl")
    manifest = json.loads((run_dir / "manifest_snapshot.json").read_text())
    predictions = rows(run_dir / "predictions.jsonl")
    qids = {row["qid"] for row in manifest}
    trace_qids = [row["qid"] for row in traces]
    pred_qids = [row["qid"] for row in predictions]
    transient = [row for row in traces if row.get("status") in TRANSIENT]
    invalid_vectors = [
        row for row in traces
        if len(row.get("pred_saliency_scores", [])) != row["num_clips"]
        or any(score not in (0.0, 2.0, 3.0, 4.0) for score in row.get("pred_saliency_scores", []))
    ]
    valid_count = sum(bool(row.get("prediction_valid")) for row in traces)
    all_zero = sum(bool(row.get("prediction_valid")) and not any(row["pred_saliency_scores"]) for row in traces)
    missing_hashes = sum(not row.get("global_frame_sha256") for row in traces)
    result = {
        "run_dir": str(run_dir), "manifest_count": len(manifest), "trace_count": len(traces),
        "prediction_count": len(predictions), "unique_trace_qids": len(set(trace_qids)),
        "unique_prediction_qids": len(set(pred_qids)),
        "qid_coverage_exact": set(trace_qids) == qids and set(pred_qids) == qids,
        "expected_count_match": len(manifest) == expected,
        "native_prediction_valid_count": valid_count,
        "native_prediction_valid_rate": valid_count / len(manifest) if manifest else 0,
        "all_zero_valid_predictions": all_zero,
        "all_zero_rate": all_zero / valid_count if valid_count else None,
        "invalid_vectors": len(invalid_vectors), "transient_failures": len(transient),
        "global_frame_hash_missing": missing_hashes,
    }
    result["gate_pass"] = all([
        result["expected_count_match"], result["qid_coverage_exact"],
        result["native_prediction_valid_rate"] >= 0.95,
        result["invalid_vectors"] == 0, result["transient_failures"] == 0,
        result["global_frame_hash_missing"] == 0,
    ])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--global-dir", required=True)
    parser.add_argument("--tool-dir", required=True)
    parser.add_argument("--expected", type=int, default=100)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = {
        "global": inspect(Path(args.global_dir), args.expected),
        "tool": inspect(Path(args.tool_dir), args.expected),
    }
    result["gate_pass"] = result["global"]["gate_pass"] and result["tool"]["gate_pass"]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["gate_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
