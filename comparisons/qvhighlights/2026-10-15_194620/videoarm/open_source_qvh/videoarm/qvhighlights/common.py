from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = ROOT / "reproduction/qvhighlights"
FIELDS = {
    "qid",
    "query",
    "vid",
    "duration",
    "media_path",
    "timeline_id",
    "num_clips",
    "split",
}


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def read_jsonl(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
    try:
        with os.fdopen(fd, "w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def source_id(vid):
    source, start, end = vid.rsplit("_", 2)
    if not source or not (math.isfinite(float(start)) and float(end) > float(start)):
        raise ValueError("invalid dataset video id")
    return source, float(start), float(end)


def finite(value):
    if (
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not math.isfinite(value)
    ):
        raise ValueError("expected a finite number")
    return float(value)


def load_manifest(path):
    rows = read_jsonl(path)
    seen = set()
    for row in rows:
        if set(row) != FIELDS:
            raise ValueError(
                f"inference field whitelist violation: {set(row) ^ FIELDS}"
            )
        if (
            isinstance(row["qid"], bool)
            or not isinstance(row["qid"], int)
            or row["qid"] in seen
        ):
            raise ValueError("duplicate/invalid qid")
        seen.add(row["qid"])
        d = finite(row["duration"])
        if (
            d <= 0
            or isinstance(row["num_clips"], bool)
            or not isinstance(row["num_clips"], int)
            or row["num_clips"] != int(d / 2)
        ):
            raise ValueError("duration/clip count mismatch")
        if not isinstance(row["query"], str) or not row["query"]:
            raise ValueError("missing query")
        source_id(row["vid"])
    return rows


def code_identity():
    paths = sorted((ROOT / "open_source_qvh").rglob("*.py"))
    paths += sorted((ARTIFACTS / "tools").glob("*.py"))
    return {str(p.relative_to(ROOT)): file_hash(p) for p in paths}
