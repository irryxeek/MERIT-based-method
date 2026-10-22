"""Small, explicit helpers for frozen QVHighlights inference runs."""

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STORE = Path(os.environ.get("ZEROSHOT_ARTIFACT_ROOT", "artifacts/zeroshot"))


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def json_default(value):
    # The unmodified author's MR proposal code emits NumPy scalar timestamps.
    if type(value).__module__.startswith("numpy") and hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=json_default) + "\n")
    temp.replace(path)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False, default=json_default) + "\n")
    temp.replace(path)


def validate_manifest(rows):
    allowed = {"qid", "vid", "query", "duration", "video_path", "rephrased_query", "caption_path"}
    if len({r["qid"] for r in rows}) != len(rows):
        raise ValueError("Duplicate query IDs")
    for row in rows:
        if set(row) - allowed:
            raise ValueError(f"Unexpected inference fields: {set(row) - allowed}")
        if not isinstance(row["query"], str) or row["duration"] <= 0:
            raise ValueError("Invalid query or duration")


def validate_scores(scores, count):
    if len(scores) != count:
        raise ValueError(f"Expected {count} scores, received {len(scores)}")
    if not all(isinstance(s, (int, float)) and math.isfinite(s) for s in scores):
        raise ValueError("Non-finite or non-numeric score")


def pad_vtg_tail(scores, count):
    """Mirror the official evaluator's zero padding for the audited missing tail."""
    if len(scores) not in (count, count - 1):
        raise ValueError("Unexpected caption coverage; only the audited one-clip tail is allowed")
    validate_scores(scores, len(scores))
    if not scores or min(scores) < 0 or max(scores) > 1.00001:
        raise ValueError("VTG-GPT normalized scores must be finite and in [0, 1]")
    result = list(scores) + [0.0] * (count - len(scores))
    validate_scores(result, count)
    return result
