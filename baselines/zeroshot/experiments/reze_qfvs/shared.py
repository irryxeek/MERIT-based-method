"""Frozen QFVS scope, REZE readout, and checkpoint checks."""

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "vendor"))

from common import atomic_json, read_jsonl, sha256, utc_now, write_jsonl  # noqa: E402, F401
from reze_core import extract_pyes, segments_for, segments_to_curve  # noqa: E402, F401


def query_text(concepts):
    aliases = {"Cupglass": "cup or glass", "Musicalinstrument": "musical instrument", "Petsanimal": "pet or animal"}
    if len(concepts) != 2:
        raise ValueError("QFVS release queries must have exactly two concepts")
    labels = [aliases.get(c, c.lower()) for c in concepts]
    return f"The video shows either {labels[0]} or {labels[1]}."


def qfvs_scores(probabilities, segments, duration, expected_count):
    """Retain REZE's second curve and average available seconds per native shot."""
    if expected_count not in {math.floor(duration / 5), math.ceil(duration / 5)}:
        raise ValueError("Native shot count is incompatible with media duration")
    curve = segments_to_curve(probabilities, segments, duration)
    scores = []
    for i in range(expected_count):
        part = curve[5 * i : 5 * (i + 1)]
        if len(part) == 0:
            raise ValueError("Native shot has no scored integer second")
        scores.append(float(part.mean()))
    return scores


def select_summary(scores, fraction=0.02):
    """Use a fixed 2% budget with explicit temporal tie-breaking."""
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Invalid shot scores")
    k = int(len(values) * fraction)
    if not 0 < k <= len(values):
        raise ValueError("Invalid summary budget")
    return np.argsort(-values, kind="stable")[:k].tolist()


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
        "query_name",
        "concepts",
        "query",
        "duration",
        "video_path",
        "media_sha256",
        "media_size",
        "media_mtime_ns",
        "n_shots",
    }
    if len(rows) != binding["n_queries"] or len({r["qid"] for r in rows}) != len(rows):
        raise ValueError("Query scope mismatch")
    checked = set()
    for row in rows:
        if set(row) != allowed or row["query"] != query_text(row["concepts"]):
            raise ValueError("Unexpected inference fields or query")
        if row["vid"] not in checked:
            stat = Path(row["video_path"]).stat()
            if stat.st_size != row["media_size"] or stat.st_mtime_ns != row["media_mtime_ns"]:
                raise ValueError(f"Verified media changed: {row['vid']}")
            checked.add(row["vid"])
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
