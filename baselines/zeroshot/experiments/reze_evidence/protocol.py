"""Bindings, budgets, and strict record replay; no annotations enter inference."""

import hashlib
import json
import math
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "scripts"))

from common import atomic_json, read_jsonl, sha256, utc_now, validate_scores, write_jsonl  # noqa: E402
from reze_core import extract_pyes, highlight_scores  # noqa: E402

__all__ = ["atomic_json", "read_jsonl", "sha256", "utc_now", "validate_scores", "write_jsonl"]

STORE = Path((os.environ.get("ZEROSHOT_ARTIFACT_ROOT", "artifacts/zeroshot") + "/reze_evidence_v1"))
PLAN = Path((os.environ.get("MERIT_ROOT", ".") + "/configs/reze_evidence_plan_v1.json"))
PYTHON = os.environ.get("REPRO_PYTHON", "python")
FIELDS = {"qid", "vid", "query", "duration", "video_path"}
BASE_SOURCES = [
    "scripts/common.py",
    "scripts/reze_core.py",
    "scripts/reze_media.py",
    "scripts/reze_backend.py",
    "configs/reze_protocol_v2.json",
    "manifests/reze_model_audit.json",
    "scripts/summarize.py",
]


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def validate_rows(rows):
    if not rows or len({row["qid"] for row in rows}) != len(rows):
        raise ValueError("Empty or duplicate inference identities")
    for row in rows:
        if set(row) != FIELDS or not isinstance(row["query"], str):
            raise ValueError("Inference manifest contains unauthorized fields")
        if not math.isfinite(row["duration"]) or row["duration"] < 2:
            raise ValueError("Invalid inference duration")


def source_hashes():
    names = BASE_SOURCES + [
        str(p.relative_to(PROJECT)) for p in sorted((PROJECT / "experiments/reze_evidence").glob("*.py"))
    ]
    return {name: sha256(PROJECT / name) for name in names}


def load_run(run, check_current=True):
    run = Path(run)
    binding = json.loads((run / "binding.json").read_text())
    for name, expected in binding["source_sha256"].items():
        if (check_current and sha256(PROJECT / name) != expected) or sha256(run / "source_snapshot" / name) != expected:
            raise ValueError(f"Frozen source changed: {name}")
    if sha256(run / "inference.jsonl") != binding["manifest_sha256"]:
        raise ValueError("Inference manifest drift")
    if sha256(run / "plan.json") != binding["plan_sha256"]:
        raise ValueError("Plan drift")
    if binding.get("parent_run"):
        parent = Path(binding["parent_run"])
        if sha256(parent / "completion.json") != binding["parent_completion_sha256"]:
            raise ValueError("Parent completion changed")
    rows = read_jsonl(run / "inference.jsonl")
    validate_rows(rows)
    return binding, rows


def verify_logprob(record, strict=False):
    logs = {item["id"]: item["logprob"] for item in record["top20"]}
    if strict and not {9454, 2753} <= set(logs):
        raise ValueError("Required Yes/No logprob missing")
    value, mode = extract_pyes(logs, 9454, 2753)
    if value != record["pyes"] or mode != record["score_mode"]:
        raise ValueError("Probability does not replay from raw logits")


def validate_baseline(record, row, binding, binding_hash):
    if record["status"] != "complete" or record["binding_sha256"] != binding_hash:
        raise ValueError("Baseline identity drift")
    for key in ("qid", "vid", "query"):
        if record["prediction"][key] != row[key]:
            raise ValueError("Baseline query identity mismatch")
    if record["media"]["source_sha256"] != binding["video_sha256"][row["vid"]]:
        raise ValueError("Baseline media mismatch")
    for item in record["clips"]:
        verify_logprob(item)
    scores = record["prediction"]["pred_saliency_scores"]
    validate_scores(scores, int(row["duration"] / 2))
    expected = highlight_scores([r["pyes"] for r in record["clips"]], record["media"]["segments"], row["duration"])
    if scores != expected:
        raise ValueError("Baseline score readout mismatch")


def read_baseline(run, row):
    run = Path(run)
    binding = json.loads((run / "binding.json").read_text())
    path = run / "records" / f"{row['qid']}.json"
    completion = json.loads((run / "completion.json").read_text())
    if completion["binding_sha256"] != sha256(run / "binding.json"):
        raise ValueError("Parent binding changed")
    if completion["record_receipt_sha256"] != sha256(run / "record_receipt.json"):
        raise ValueError("Parent record receipt changed")
    receipt = json.loads((run / "record_receipt.json").read_text())
    if sha256(path) != receipt[str(row["qid"])]:
        raise ValueError("Baseline receipt mismatch")
    record = json.loads(path.read_text())
    validate_baseline(record, row, binding, sha256(run / "binding.json"))
    return record


def validate_pair_record(record, row, binding, binding_hash, baseline):
    from ranking import candidates, make_scores

    if record["binding_sha256"] != binding_hash or record["qid"] != row["qid"]:
        raise ValueError("Pair checkpoint binding mismatch")
    if record["vid"] != row["vid"] or record["query"] != row["query"]:
        raise ValueError("Pair checkpoint query mismatch")
    original = baseline["prediction"]["pred_saliency_scores"]
    selected = candidates(original)
    if record["candidate_ids"] != selected:
        raise ValueError("Candidate selection changed")
    for arm in ("P_abs", "P_pair"):
        result = record["arms"][arm]
        if result["status"] == "complete":
            for item in result["requests"]:
                verify_logprob(item, strict=True)
                if item["prompt_sha256"] != digest(item["prompt"]):
                    raise ValueError("Prompt hash mismatch")
                traces = record["candidate_frames"]
                if item["evidence_sha256"] != digest([traces[str(item["a"])], traces[str(item["b"])]]):
                    raise ValueError("Request evidence mismatch")
            scores, diagnostics = make_scores(original, selected, result["requests"], arm)
            if scores != result["scores"] or diagnostics != result["ranking"]:
                raise ValueError("Pair ranking does not replay")
        elif result["status"] == "fallback":
            if not result.get("error") or result["scores"] != original:
                raise ValueError("Invalid fallback record")
        else:
            raise ValueError("Unfinished pair arm")
        validate_scores(result["scores"], len(original))
        if sorted(result["scores"]) != sorted(original):
            raise ValueError("Score multiset changed")
        if any(result["scores"][i] != value for i, value in enumerate(original) if i not in selected):
            raise ValueError("Noncandidate score changed")


def validate_stress_record(record, row, binding, binding_hash):
    from ranking import candidates

    if record["binding_sha256"] != binding_hash or record["qid"] != row["qid"] or record["status"] != "complete":
        raise ValueError("Stress identity or completion mismatch")
    baseline = read_baseline(binding["parent_run"], row)
    selected = candidates(baseline["prediction"]["pred_saliency_scores"])
    if selected != record["candidate_ids"]:
        raise ValueError("Stress candidates changed")
    expected = [(arm, selected[i], selected[(i + 1) % 5]) for i in range(5) for arm in ("P_abs", "P_pair")]
    actual = [(r["arm"], r["a"], r["b"]) for r in record["requests"]]
    if actual != expected:
        raise ValueError("Incomplete engineering stress cycle")
    for request in record["requests"]:
        if request.get("error"):
            raise ValueError(request["error"])
        verify_logprob(request, strict=True)
        traces = record["candidate_frames"]
        if request["evidence_sha256"] != digest([traces[str(request["a"])], traces[str(request["b"])]]):
            raise ValueError("Stress evidence changed")
