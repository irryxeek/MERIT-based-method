"""Frozen inputs and durable checkpoints shared by the CPU and GPU paths."""

import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "experiments/reze_evidence"))
import protocol as legacy  # noqa: E402

digest = legacy.digest
sha256 = legacy.sha256
utc_now = legacy.utc_now


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("w") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def read_json(path):
    return json.loads(Path(path).read_text())


def fixed_twenty(rows):
    if len(rows) != 100:
        raise ValueError("Expected existing explore100")
    return sorted(rows, key=lambda r: hashlib.sha256(("revalue-v3:" + str(r["qid"])).encode()).hexdigest())[:20]


def parents(binding, row):
    b_run, p_run = (Path(binding[k]) for k in ("baseline_run", "pair_run"))
    for parent in (b_run, p_run):
        receipt = read_json(parent / "record_receipt.json")
        path = parent / "records" / f"{row['qid']}.json"
        if sha256(path) != receipt[str(row["qid"])]:
            raise ValueError("Parent record drift")
    baseline = legacy.read_baseline(b_run, row)
    pair = read_json(p_run / "records" / f"{row['qid']}.json")
    legacy.validate_pair_record(pair, row, read_json(p_run / "binding.json"), sha256(p_run / "binding.json"), baseline)
    if pair["arms"]["P_pair"]["status"] != "complete":
        raise ValueError("E0 must have a valid audited parent")
    return baseline, pair


def load_run(run):
    run = Path(run)
    binding = read_json(run / "binding.json")
    for name, expected in binding["source_sha256"].items():
        if sha256(ROOT / name) != expected or sha256(run / "source_snapshot" / name) != expected:
            raise ValueError(f"Source drift: {name}")
    for name, expected in binding["input_sha256"].items():
        if sha256(run / name) != expected:
            raise ValueError(f"Input drift: {name}")
    for name, expected in binding["parent_receipts"].items():
        if sha256(name) != expected:
            raise ValueError(f"Parent drift: {name}")
    rows = legacy.read_jsonl(run / "inference.jsonl")
    legacy.validate_rows(rows)
    if rows != fixed_twenty(legacy.read_jsonl(Path(binding["pair_run"]) / "inference.jsonl")):
        raise ValueError("Fixed 20 selection drift")
    return binding, rows, sha256(run / "binding.json")
