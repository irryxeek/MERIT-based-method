"""Protocol checks and deterministic temporal operations."""

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

INFERENCE_FIELDS = {"qid", "vid", "query", "duration", "video_path"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".writing-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def validate_manifest(rows):
    seen = set()
    for row in rows:
        if set(row) != INFERENCE_FIELDS:
            raise ValueError("Inference manifest contains unexpected or missing fields")
        if type(row["qid"]) is not int:
            raise ValueError("QVHighlights query IDs must be integers")
        if row["qid"] in seen:
            raise ValueError("Duplicate qid")
        seen.add(row["qid"])
        if not isinstance(row["query"], str) or not row["query"].strip():
            raise ValueError("Empty query")
        duration = row["duration"]
        if isinstance(duration, bool) or not isinstance(duration, (float, int)):
            raise ValueError("Invalid duration")
        if not math.isfinite(duration) or duration < 2:
            raise ValueError("Invalid duration")
        if not isinstance(row["vid"], str) or not row["vid"]:
            raise ValueError("Invalid video id")
        if Path(row["vid"]).name != row["vid"] or row["vid"] in (".", ".."):
            raise ValueError("Invalid video id")
    return rows


def windows(duration, seconds):
    if not math.isfinite(duration) or duration <= 0 or seconds <= 0:
        raise ValueError("Invalid temporal extent")
    return [(start, min(start + seconds, duration)) for start in range(0, math.ceil(duration), seconds)]


def neighbors(anchor_ids, count, radius):
    return sorted({j for i in anchor_ids for j in range(max(0, i - radius), min(count, i + radius + 1))})


def rank_clips(key_vectors, query_vector, excluded, top_k):
    import numpy as np

    vectors = np.asarray(key_vectors)
    query = np.asarray(query_vector)
    if vectors.ndim != 3 or query.ndim != 1 or vectors.shape[-1] != query.shape[-1]:
        raise ValueError("Embedding shape mismatch")
    if not np.isfinite(vectors).all() or not np.isfinite(query).all():
        raise ValueError("Nonfinite embeddings")
    norms = np.linalg.norm(vectors, axis=-1)
    query_norm = np.linalg.norm(query)
    if (norms <= 0).any() or query_norm <= 0:
        raise ValueError("Zero embedding")
    similarities = np.sum(vectors * query, axis=-1) / (norms * query_norm)
    scores = similarities.max(axis=1)
    # Temporal IDs remain distinct even when captions or embeddings are identical.
    candidates = sorted((i for i in range(len(scores)) if i not in excluded), key=lambda i: (-scores[i], i))
    return [(int(i), float(scores[i])) for i in candidates[:top_k]]


def parse_json(text):
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(text)


def validate_scores(value, expected):
    if not isinstance(value, dict) or set(value) != {"scores"}:
        raise ValueError("Expected object with only scores")
    scores = value["scores"]
    if not isinstance(scores, list) or len(scores) != expected:
        raise ValueError(f"Expected exactly {expected} ordered scores")
    if any(
        isinstance(s, bool) or not isinstance(s, (int, float)) or not math.isfinite(s) or not 0 <= s <= 100
        for s in scores
    ):
        raise ValueError("Scores must be finite numbers in [0, 100]")
    return [float(s) for s in scores]


def validate_keys(value):
    if not isinstance(value, dict) or set(value) != {"event", "object", "summary"}:
        raise ValueError("Expected event, object, summary")
    if any(not isinstance(v, str) or not v.strip() or len(v) > 1500 for v in value.values()):
        raise ValueError("Keys must be nonempty short strings")
    return {key: value[key].strip() for key in ("event", "object", "summary")}


def validate_filter(value, allowed_ids):
    if not isinstance(value, dict) or set(value) != {"evidence"} or not isinstance(value["evidence"], list):
        raise ValueError("Expected evidence list")
    for item in value["evidence"]:
        if not isinstance(item, dict) or set(item) != {"clip_ids", "text"}:
            raise ValueError("Invalid evidence item")
        ids = item["clip_ids"]
        if (
            not isinstance(ids, list)
            or not ids
            or any(type(i) is not int or i not in allowed_ids for i in ids)
        ):
            raise ValueError("Evidence refers to unsupported temporal IDs")
        if not isinstance(item["text"], str) or not item["text"].strip():
            raise ValueError("Empty evidence text")
    return value["evidence"]


def validate_decision(value):
    if not isinstance(value, dict) or set(value) != {"sufficient", "search_query"}:
        raise ValueError("Expected sufficient and search_query")
    if type(value["sufficient"]) is not bool or not isinstance(value["search_query"], str):
        raise ValueError("Invalid decision types")
    if not value["sufficient"] and not value["search_query"].strip():
        raise ValueError("Further retrieval needs a query")
    return value
