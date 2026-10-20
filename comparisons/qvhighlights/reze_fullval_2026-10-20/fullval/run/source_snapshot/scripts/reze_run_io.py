"""Immutable bindings and validation shared by REZE execution and collection."""

import json
from pathlib import Path

from common import ROOT, read_jsonl, sha256, validate_manifest, validate_scores
from reze_core import extract_pyes, highlight_scores


def load_bound_run(run):
    binding = json.loads((run / "binding.json").read_text())
    for name, digest in binding["source_sha256"].items():
        if sha256(ROOT / name) != digest or sha256(run / "source_snapshot" / name) != digest:
            raise ValueError(f"Frozen source changed: {name}")
    if sha256(run / "inference.jsonl") != binding["manifest_sha256"]:
        raise ValueError("Frozen inference manifest changed")
    rows = read_jsonl(run / "inference.jsonl")
    validate_manifest(rows)
    if any(set(r) != {"qid", "vid", "query", "duration", "video_path"} for r in rows):
        raise ValueError("Inference must not include captions or ground-truth annotations")
    return binding, rows


def validate_record(record, row, binding, binding_hash):
    if record["status"] != "complete" or record["binding_sha256"] != binding_hash:
        raise ValueError("Incomplete or mismatched checkpoint")
    pred, media = record["prediction"], record["media"]
    for key in ["qid", "vid", "query"]:
        if pred[key] != row[key]:
            raise ValueError(f"Checkpoint identity mismatch: {key}")
    if media["source_sha256"] != binding["video_sha256"][row["vid"]]:
        raise ValueError("Video bytes changed since CLIP baseline")
    for clip in record["clips"]:
        probability, mode = extract_pyes({item["id"]: item["logprob"] for item in clip["top20"]}, 9454, 2753)
        if probability != clip["pyes"] or mode != clip["score_mode"]:
            raise ValueError("Saved probability does not match raw Yes/No log probabilities")
    scores = pred["pred_saliency_scores"]
    validate_scores(scores, int(row["duration"] / 2))
    recomputed = highlight_scores([c["pyes"] for c in record["clips"]], media["segments"], row["duration"])
    if scores != recomputed:
        raise ValueError("Saved HD scores do not match the recorded clip probabilities")
    if not all(0 <= value <= 1 for value in scores):
        raise ValueError("HD scores outside [0, 1]")


def read_record(run, row, binding, binding_hash):
    path = Path(run) / "records" / f"{row['qid']}.json"
    record = json.loads(path.read_text())
    validate_record(record, row, binding, binding_hash)
    return record
