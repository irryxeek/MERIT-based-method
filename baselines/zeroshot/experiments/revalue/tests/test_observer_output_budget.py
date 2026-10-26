"""Reject over-budget requests before inference; preserve failed reservation cost."""

import time

import pytest
from budget import BudgetStop
from observer_output_probe import MAX_TOTAL, ProbeLedger


def ledger(tmp_path):
    return ProbeLedger(tmp_path / "ledger.json", deadline=time.time() + 900)


def test_reject_full_multimodal_request_above_cap(tmp_path):
    value = ledger(tmp_path)
    with pytest.raises(BudgetStop, match="2048"):
        value.reserve("one", 1889, 160, ["E_uniform"], "hash", 2)
    assert value.state["attempts"] == []


def test_global_cost_includes_unsettled_attempts_across_arms(tmp_path):
    value = ledger(tmp_path)
    for i in range(37):
        value.reserve(str(i), 1888, 160, ["E_uniform" if i % 2 else "E_targeted"], f"h{i}", 2)
    assert value.cost() == MAX_TOTAL
    with pytest.raises(BudgetStop, match="75776"):
        value.reserve("extra", 1, 1, ["E_targeted"], "extra", 2)


def test_zero_retry_budget_preserves_failed_attempt(tmp_path):
    value = ledger(tmp_path)
    number = value.reserve("one", 300, 160, ["E_uniform"], "hash", 2)
    value.settle(number, {"status": "runtime_error", "error": "transient"})
    with pytest.raises(BudgetStop, match="request budget"):
        value.reserve("one", 300, 160, ["E_uniform"], "hash", 2)
    assert value.cost() == 460


def test_shutdown_reserve_precedes_request(tmp_path):
    value = ProbeLedger(tmp_path / "ledger.json", deadline=time.time() + 59)
    with pytest.raises(BudgetStop, match="shutdown"):
        value.reserve("one", 300, 160, ["E_uniform"], "hash", 2)
    assert value.state["attempts"] == []
