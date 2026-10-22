"""Frozen query-free MR.HiSum interface and REZE checkpoint validation."""

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "vendor"))

from common import atomic_json, read_jsonl, sha256, utc_now, write_jsonl  # noqa: E402, F401
from reze_core import extract_pyes, segments_for, segments_to_curve  # noqa: E402, F401

QUERY = "Highlight of this video."


def second_scores(probabilities, segments, duration, expected_count):
    if duration != expected_count or not isinstance(expected_count, int):
        raise ValueError("MR.HiSum metadata must define an integral one-second annotation grid")
    curve = segments_to_curve(probabilities, segments, duration)
    if len(curve) != expected_count:
        raise ValueError("REZE curve differs from the frozen annotation grid")
    return curve.tolist()


def load_run(run):
    run = Path(run)
    binding = json.loads((run / "binding.json").read_text())
    if sha256(run / "inference.jsonl") != binding["inference_sha256"]:
        raise ValueError("Inference manifest changed")
    for name, digest in binding["source_sha256"].items():
        if sha256(HERE / name) != digest:
            raise ValueError(f"Frozen source changed: {name}")
    rows = read_jsonl(run / "inference.jsonl")
    allowed = {
        "qid",
        "vid",
        "query",
        "duration",
        "video_path",
        "media_sha256",
        "media_size",
        "media_mtime_ns",
        "n_seconds",
    }
    if len(rows) != binding["n_queries"] or len({r["qid"] for r in rows}) != len(rows):
        raise ValueError("Query scope mismatch")
    if [r["qid"] for r in rows] != binding["selected_qids"]:
        raise ValueError("Fixed video selection changed")
    for row in rows:
        if set(row) != allowed or row["query"] != QUERY or row["qid"] != row["vid"]:
            raise ValueError("Unexpected inference fields or query")
        if row["duration"] != row["n_seconds"]:
            raise ValueError("Annotation grid changed")
        stat = Path(row["video_path"]).stat()
        if stat.st_size != row["media_size"] or stat.st_mtime_ns != row["media_mtime_ns"]:
            raise ValueError(f"Verified media changed: {row['vid']}")
    return binding, rows, sha256(run / "binding.json")


def validate_chunk(data, binding_hash, row, start, segments):
    if data["binding_sha256"] != binding_hash or data["qid"] != row["qid"]:
        raise ValueError("Checkpoint belongs to another query or run")
    if data["start"] != start or data["segments"] != [list(s) for s in segments]:
        raise ValueError("Checkpoint windows changed")
    if len(data["records"]) != len(segments) or len(data["trace"]["model_rgb_sha256"]) != len(segments):
        raise ValueError("Incomplete probability or pixel trace")
    for record in data["records"]:
        raw = {v["id"]: v["logprob"] for v in record["top20"]}
        if len(raw) != len(record["top20"]):
            raise ValueError("Duplicate raw probability tokens")
        p, mode = extract_pyes(raw, 9454, 2753)
        if p != record["pyes"] or mode != record["score_mode"] or not math.isfinite(p):
            raise ValueError("Score differs from raw log probabilities")


def load_query_chunks(run, row, binding_hash, segments, batch_size):
    records, chunks = [], []
    for start in range(0, len(segments), batch_size):
        data = json.loads((Path(run) / "chunks" / row["qid"] / f"{start:06d}.json").read_text())
        validate_chunk(data, binding_hash, row, start, segments[start : start + batch_size])
        records.extend(data["records"])
        chunks.append(data)
    return records, chunks
