"""Versioned candidate selection and replay for arbitrary five-clip sets."""

import itertools
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/reze_evidence"))

import protocol as legacy  # noqa: E402
from ranking import candidates, ordered_pairs, preference  # noqa: E402


def dispersed_candidates(scores, k=5, radius=2):
    order = candidates(scores, len(scores))
    if k < 1 or radius < 0:
        raise ValueError("Invalid candidate count or suppression radius")
    selected = []
    for candidate in order:
        if all(abs(candidate - other) > radius for other in selected):
            selected.append(candidate)
            if len(selected) == k:
                break
    filled = []
    for candidate in order:
        if len(selected) >= min(k, len(scores)):
            break
        if candidate not in selected:
            selected.append(candidate)
            filled.append(candidate)
    return selected, {"radius_clip_ids": radius, "fill_ids": filled, "initial_topk": order[:k], "selected": selected}


def permutation_scores(original, selected, records):
    if len(selected) != min(5, len(original)) or len(set(selected)) != len(selected):
        raise ValueError("Invalid candidate set")
    if not all(isinstance(i, int) and 0 <= i < len(original) for i in selected):
        raise ValueError("Candidate outside score vector")
    expected = ordered_pairs(selected)
    pairs = [(r["a"], r["b"]) for r in records]
    if len(pairs) != len(set(pairs)) or set(pairs) != set(expected):
        raise ValueError("Missing or duplicated directed comparisons")
    values = {(r["a"], r["b"]): r["pyes"] for r in records}
    if not all(math.isfinite(p) and 0 <= p <= 1 for p in values.values()):
        raise ValueError("Invalid raw probability")
    edges = {(i, j): preference(values[i, j], values[j, i]) for i, j in expected}
    ranks = {i: sum(edges[i, j] for j in selected if j != i) for i in selected}
    order = sorted(selected, key=lambda i: (-ranks[i], -original[i], i))
    scores = list(original)
    for index, value in zip(order, sorted((original[i] for i in selected), reverse=True), strict=True):
        scores[index] = value
    cycles = 0
    for i, j, k in itertools.combinations(selected, 3):
        if (edges[i, j] > 0 and edges[j, k] > 0 and edges[k, i] > 0) or (
            edges[i, j] < 0 and edges[j, k] < 0 and edges[k, i] < 0
        ):
            cycles += 1
    return scores, {
        "order": order,
        "rank_values": {str(i): ranks[i] for i in selected},
        "neutral_unordered_pairs": sum(edges[i, j] == 0 for i, j in itertools.combinations(selected, 2)),
        "both_yes_pairs": sum(
            values[i, j] >= 0.6 and values[j, i] >= 0.6 for i, j in itertools.combinations(selected, 2)
        ),
        "cycles": cycles,
        "score_values_preserved": True,
    }


def parent_records(binding, row):
    baseline = legacy.read_baseline(binding["baseline_run"], row)
    pair_run = Path(binding["pair_run"])
    old_binding = json.loads((pair_run / "binding.json").read_text())
    receipt = json.loads((pair_run / "record_receipt.json").read_text())
    path = pair_run / "records" / f"{row['qid']}.json"
    if legacy.sha256(path) != receipt[str(row["qid"])]:
        raise ValueError("C0 record receipt changed")
    pair = json.loads(path.read_text())
    legacy.validate_pair_record(pair, row, old_binding, legacy.sha256(pair_run / "binding.json"), baseline)
    return baseline, pair


def load_run(run):
    run = Path(run)
    binding = json.loads((run / "binding.json").read_text())
    for name, expected in binding["source_sha256"].items():
        if legacy.sha256(ROOT / name) != expected:
            raise ValueError(f"Frozen source changed: {name}")
    for name, expected in binding["input_sha256"].items():
        if legacy.sha256(run / name) != expected:
            raise ValueError(f"Frozen input changed: {name}")
    for name, expected in binding["parent_receipts"].items():
        if legacy.sha256(Path(name)) != expected:
            raise ValueError(f"Parent receipt changed: {name}")
    rows = legacy.read_jsonl(run / "inference.jsonl")
    legacy.validate_rows(rows)
    if len(rows) != 100:
        raise ValueError("V1 requires all 100 fixed development queries")
    return binding, rows, legacy.sha256(run / "binding.json")


def validate_record(record, row, binding, binding_hash):
    baseline, _ = parent_records(binding, row)
    original = baseline["prediction"]["pred_saliency_scores"]
    selected, selection = dispersed_candidates(original)
    if any(record[k] != row[k] for k in ("qid", "vid", "query")):
        raise ValueError("Record query identity mismatch")
    if record["binding_sha256"] != binding_hash or record["selection"] != selection:
        raise ValueError("Record binding or selection mismatch")
    if record["candidate_ids"] != selected:
        raise ValueError("Candidate set changed")
    if record["status"] == "complete":
        for request in record["requests"]:
            legacy.verify_logprob(request, strict=True)
            if request["prompt_sha256"] != legacy.digest(request["prompt"]):
                raise ValueError("Prompt changed")
            traces = record["candidate_frames"]
            if request["evidence_sha256"] != legacy.digest([traces[str(request["a"])], traces[str(request["b"])]]):
                raise ValueError("Evidence trace changed")
        scores, diagnostics = permutation_scores(original, selected, record["requests"])
        if record["scores"] != scores or record["ranking"] != diagnostics:
            raise ValueError("Rank scores do not replay")
    elif record["status"] != "fallback" or not record.get("error") or record["scores"] != original:
        raise ValueError("Invalid fallback")
    legacy.validate_scores(record["scores"], len(original))
    if sorted(record["scores"]) != sorted(original):
        raise ValueError("Score multiset changed")
    if any(record["scores"][i] != s for i, s in enumerate(original) if i not in selected):
        raise ValueError("Noncandidate score changed")
