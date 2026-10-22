"""CPU-only TVSum HD evaluation after every raw-score checkpoint validates."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from metrics import author_top5_map, summarize, two_second_scores, video_metrics
from shared import atomic_json, load_query_chunks, load_run, second_scores, sha256, utc_now, write_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    binding, rows, binding_hash = load_run(run)
    complete = json.loads((run / "inference_complete.json").read_text())
    if complete["binding_sha256"] != binding_hash or complete["queries"] != len(rows):
        raise ValueError("Inference completion belongs to another run or scope")
    if {p.stem for p in (run / "predictions").glob("*.json")} != set(binding["selected_qids"]):
        raise ValueError("Prediction ID set differs from fixed selection")
    if sha256(run / "media_verification.json") != binding["media_verification_sha256"]:
        raise ValueError("Frozen media timeline changed")
    for name, digest in binding["labels_sha256"].items():
        if sha256(run / "evaluation_inputs" / name) != digest:
            raise ValueError("Frozen evaluation inputs changed")
    labels = json.loads((run / "evaluation_inputs/tvsum_anno.json").read_text())
    metadata = json.loads((run / "evaluation_inputs/dataset_audit.json").read_text())["videos"]
    media = json.loads((run / "media_verification.json").read_text())
    details, predictions = [], {}
    for row in rows:
        vid = row["vid"]
        saved = json.loads((run / "predictions" / f"{vid}.json").read_text())
        segments = media[vid]["decoded_timeline"]["segments"]
        records, _ = load_query_chunks(run, row, binding_hash, segments, binding["batch_size"])
        curve = second_scores([r["pyes"] for r in records], segments, row["duration"], row["n_seconds"])
        if saved["binding_sha256"] != binding_hash or curve != saved["second_scores"]:
            raise ValueError("Saved curve differs from raw probability reconstruction")
        scores = two_second_scores(curve, metadata[vid]["n_2s_clips"])
        predictions[vid] = scores
        details.append(
            {
                "vid": vid,
                "category": metadata[vid]["category"],
                "umt_split": metadata[vid]["umt_split"],
                **video_metrics(scores, labels[vid]["anno"]),
            }
        )
    metrics = summarize(details)
    author = author_top5_map(predictions, labels)
    if abs(author - metrics["Top5_AP"]["mean"]) > 0.000501:
        raise ValueError("Independent metric differs from unmodified UMT evaluate method")
    ledgers = [json.loads(p.read_text()) for p in (run / "request_attempts").glob("*.json")]
    summaries = [json.loads((run / "query_summaries" / f"{r['vid']}.json").read_text()) for r in rows]
    attempts = json.loads((run / "worker_attempts.json").read_text())
    answers, modes = Counter(), Counter()
    for item in summaries:
        if item["binding_sha256"] != binding_hash:
            raise ValueError("Query cost summary belongs to another run")
        answers.update(item["generated_answers"])
        modes.update(item["score_modes"])
    if any(item["binding_sha256"] != binding_hash for item in ledgers):
        raise ValueError("Request ledger belongs to another run")
    cost = {
        "formal_requests": sum(q["model_requests"] for q in summaries),
        "charged_requests_including_smoke_and_failed_attempts": sum(r["requests"] for r in ledgers),
        "worker_seconds": sum(a["seconds"] for a in attempts),
        "gpu_worker_attempts": len(attempts),
        "formal_input_tokens": sum(q["input_tokens"] for q in summaries),
        "formal_frames": sum(q["frames"] for q in summaries),
        "formal_model_seconds": sum(q["inference_seconds"] for q in summaries),
        "formal_decode_seconds": sum(q["decode_preprocess_seconds"] for q in summaries),
    }
    if cost["formal_requests"] != binding["expected_model_requests"]:
        raise ValueError("Incomplete request coverage")
    write_jsonl(run / "per_video_metrics.jsonl", details)
    write_jsonl(run / "hd_predictions.jsonl", [{"vid": vid, "scores": score} for vid, score in predictions.items()])
    result = {
        "status": "complete",
        "finished_at": utc_now(),
        "binding_sha256": binding_hash,
        "scope": binding["scope"],
        "evaluated_videos": len(rows),
        "metrics_percent": metrics,
        "author_evaluator_rounded_percent": author,
        "author_metric_parity_passed": True,
        "cost": cost,
        "diagnostics": {
            "answers": dict(answers),
            "score_modes": dict(modes),
            "top5_boundary_tie_videos": sum(d["top5_boundary_tie"] for d in details),
            "constant_prediction_videos": sum(d["unique_scores"] == 1 for d in details),
        },
        "per_video_metrics_sha256": sha256(run / "per_video_metrics.jsonl"),
        "hd_predictions_sha256": sha256(run / "hd_predictions.jsonl"),
    }
    if not np.isfinite([v["mean"] for v in metrics.values()]).all():
        raise ValueError("Nonfinite result")
    atomic_json(run / "evaluation.json", result)
    atomic_json(
        run / "complete.json",
        {
            "status": "complete",
            "finished_at": utc_now(),
            "binding_sha256": binding_hash,
            "evaluation_sha256": sha256(run / "evaluation.json"),
        },
    )
    atomic_json(
        run / "status.json",
        {
            "status": "complete",
            "updated_at": utc_now(),
            "completed_queries": len(rows),
            "expected_queries": len(rows),
            "metrics_percent": {k: v["mean"] for k, v in metrics.items()},
        },
    )
    print(json.dumps({"evaluated": len(rows), "metrics": metrics, "cost": cost}))


if __name__ == "__main__":
    main()
