"""Behavioral checks for transfer isolation and complete-budget accounting."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import (  # noqa: E402
    ARMS,
    LIMITS,
    MAX_SHARD_CALLS,
    MAX_SHARD_TOKENS,
    BudgetStop,
    SemanticFailure,
    TransferLedger,
    one_video,
    verify_scores,
)


class Tokenizer:
    def encode(self, text, **_kwargs):
        return list(range(len(text) // 5 + 1))


class FakeBackend:
    tokenizer = Tokenizer()

    def __init__(self, fail_memory=False, fail_task=False):
        self.calls = []
        self.fail_memory, self.fail_task = fail_memory, fail_task

    def pair(self, key, query, a, b, traces):
        self.calls.append((key, query))
        return {"pyes": 0.5, "attempt": len(self.calls) - 1}

    def call(self, key, text, videos, evidence, schema, cap, arms, validate, references=None, **_kwargs):
        self.calls.append((key, text))
        if schema.__name__ == "TaskSpec":
            if self.fail_task:
                raise SemanticFailure("Malformed task")
            value = {"request_type": "generic", "purpose": "highlight", "constraints": []}
        elif schema.__name__ == "Facts":
            value = {"facts": [{"kind": "context", "text": "A visible person", "frames": references["FactItem.frames"][:1]}], "unknown": []}
        elif schema.__name__ == "MemoryDraft":
            if self.fail_memory:
                raise SemanticFailure("Invalid memory")
            value = {"items": [{"text": "A visible person", "fact_ids": [identity]} for identity in references["MemoryItem.fact_ids"]]}
        else:
            value = {"a_eligibility": "satisfied", "b_eligibility": "satisfied", "preference": "tie",
                     "values": [], "reason": "Similar visible content", "a_evidence": references["a_evidence"][:1],
                     "b_evidence": references["b_evidence"][:1], "missing_question": None}
        return validate(value), {"attempt": len(self.calls) - 1}


def inputs():
    row = {"qid": "test", "vid": "test", "query": "Highlight of this video.",
           "candidate_ids": [0, 1, 2, 3, 4], "scores": [1, .9, .8, .7, .6, .5]}
    traces = {str(c): {"frames": [{"frame_index": c, "frame_timestamps": [2*c, 2*c+.5]}]} for c in range(5)}
    targets = {c: {"video": np.zeros((4, 2, 2, 3), dtype=np.uint8), "metadata": {}} for c in range(5)}
    return row, targets, traces


def test_full_design_has_67_calls_and_no_observation():
    backend = FakeBackend()
    row, targets, traces = inputs()
    result = one_video(row, targets, traces, backend)
    assert len(backend.calls) == 67
    assert all(result["arms"][arm]["status"] == "complete" for arm in ARMS)
    assert all(result["arms"][arm]["scores"] == row["scores"] for arm in ARMS)
    assert not any("observation" in key for key, _ in backend.calls)
    assert result["memory"]["tokens"] <= 1024
    assert all(row["query"] not in text for key, text in backend.calls if "/facts/" in key or "/memory/" in key)


def test_memory_failure_does_not_block_direct_comparison():
    backend = FakeBackend(fail_memory=True)
    result = one_video(*inputs(), backend)
    assert result["arms"]["C_direct4"]["status"] == "complete"
    assert result["arms"]["E_memory"]["status"] == "fallback"
    assert result["arms"]["E_memory"]["scores"] == result["arms"]["P_pair"]["scores"]
    assert len(backend.calls) == 47


def test_task_failure_preserves_full_denominator():
    result = one_video(*inputs(), FakeBackend(fail_task=True))
    assert result["arms"]["P_pair"]["status"] == "complete"
    assert all(result["arms"][arm]["status"] == "fallback" for arm in ("C_direct4", "E_memory"))


def test_no_retry_even_after_interruption(tmp_path):
    ledger = TransferLedger(tmp_path / "ledger.json", LIMITS)
    ledger.reserve("lost", 10, 1, ["P_pair"], "hash", 8)
    restored = TransferLedger(tmp_path / "ledger.json", LIMITS)
    with pytest.raises(BudgetStop):
        restored.reserve("lost", 10, 1, ["P_pair"], "hash", 8)


def test_absolute_budget_covers_reserved_failed_calls(tmp_path):
    ledger = TransferLedger(tmp_path / "ledger.json", LIMITS)
    for index in range(MAX_SHARD_CALLS):
        ledger.reserve(str(index), 6000, 144, ["C_direct4"], str(index), 8)
    assert ledger.cost() == MAX_SHARD_TOKENS
    with pytest.raises(BudgetStop):
        ledger.reserve("extra", 1, 1, ["C_direct4"], "extra", 8)


def test_forbid_score_changes_outside_candidates():
    with pytest.raises(ValueError):
        verify_scores([1, .9, .8, .7, .6, .5], list(range(5)), [.5, .9, .8, .7, .6, 1])
