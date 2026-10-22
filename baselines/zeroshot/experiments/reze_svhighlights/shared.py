"""Frozen REZE components and strict SVHighlights checkpoint validation."""

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "vendor"))

from common import atomic_json as atomic_json  # noqa: E402
from common import read_jsonl, sha256  # noqa: E402
from common import utc_now as utc_now  # noqa: E402
from reze_core import extract_pyes, segments_to_curve  # noqa: E402
from reze_core import segments_for as segments_for  # noqa: E402


def svh_scores(probabilities, segments, duration, expected_count):
    """Keep REZE's second curve; retain SVH's final one-second bin when present."""
    curve = segments_to_curve(probabilities, segments, duration)
    values = [float(curve[start : start + 2].mean()) for start in range(0, len(curve), 2)]
    if len(values) != expected_count:
        raise ValueError(f"Media/grid mismatch: {len(values)} != {expected_count}")
    return values


def load_run(run):
    run = Path(run)
    binding = json.loads((run / "binding.json").read_text())
    if sha256(run / "inference.jsonl") != binding["inference_sha256"]:
        raise ValueError("Inference manifest changed")
    for name, digest in binding["source_sha256"].items():
        if sha256(HERE / name) != digest:
            raise ValueError(f"Source changed: {name}")
    rows = read_jsonl(run / "inference.jsonl")
    allowed = {
        "vid",
        "sport",
        "query",
        "duration",
        "video_path",
        "media_sha256",
        "media_size",
        "media_mtime_ns",
        "n_clips",
        "source_video_id",
    }
    if len(rows) != 7 or len({row["vid"] for row in rows}) != len(rows):
        raise ValueError("Pilot must contain exactly seven unique videos")
    for row in rows:
        if set(row) != allowed:
            raise ValueError("Unexpected inference fields")
        stat = Path(row["video_path"]).stat()
        if stat.st_size != row["media_size"] or stat.st_mtime_ns != row["media_mtime_ns"]:
            raise ValueError(f"Prepared media changed: {row['vid']}")
    return binding, rows, sha256(run / "binding.json")


def validate_chunk(data, binding_hash, row, start, segments):
    if data["binding_sha256"] != binding_hash or data["vid"] != row["vid"]:
        raise ValueError("Checkpoint belongs to another run")
    if data["start"] != start or data["segments"] != [list(s) for s in segments]:
        raise ValueError("Checkpoint is not aligned to the frozen windows")
    if len(data["records"]) != len(segments) or len(data["trace"]["model_rgb_sha256"]) != len(segments):
        raise ValueError("Missing checkpoint outputs or pixels")
    for record in data["records"]:
        p, mode = extract_pyes({v["id"]: v["logprob"] for v in record["top20"]}, 9454, 2753)
        if p != record["pyes"] or mode != record["score_mode"]:
            raise ValueError("Stored score disagrees with raw log probabilities")
        if not math.isfinite(p):
            raise ValueError("Nonfinite score")


def load_video_chunks(run, row, binding_hash, segments, batch_size):
    records, chunks = [], []
    for start in range(0, len(segments), batch_size):
        path = Path(run) / "chunks" / row["vid"] / f"{start:06d}.json"
        data = json.loads(path.read_text())
        selected = segments[start : start + batch_size]
        validate_chunk(data, binding_hash, row, start, selected)
        records.extend(data["records"])
        chunks.append(data)
    return records, chunks
