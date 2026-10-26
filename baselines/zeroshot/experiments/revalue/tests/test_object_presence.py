"""Verifier invariants: labels stay hidden, abstentions explain gaps, budget is hard."""

import time

import pytest
from budget import BudgetStop
from object_presence_contract import ObjectPresence, case_jobs, prompt, validate_case_sources, validate_presence
from object_presence_probe import MAX_TOTAL, PresenceLedger


def sample():
    return {"case_id": "p01", "image_key": "1129_E_uniform_0_08", "qid": 1129,
            "object": "bottle", "region": "left edge", "claim": "A bottle is visible.",
            "reference_status": "SECRET_LABEL", "reference_reason": "SECRET_REASON",
            "old_caption": "SECRET_CAPTION", "stratum": "SECRET_STRATUM"}


def test_labels_and_old_captions_cannot_enter_model_text():
    case = sample()
    for job in case_jobs([case]):
        assert "SECRET" not in str(job)
        assert job["text"] == prompt(case, job["arm"])
    case["reference_status"] = "changed"
    assert prompt(case, "direct") == prompt(sample(), "direct")


def test_paired_jobs_counterbalance_order_and_keep_evidence():
    a, b = sample(), {**sample(), "case_id": "p02"}
    jobs = case_jobs([a, b])
    assert [j["arm"] for j in jobs] == ["direct", "evidence_first", "evidence_first", "direct"]
    assert len({j["key"] for j in jobs}) == 4
    assert len({j["image_key"] for j in jobs}) == 1
    for field in ("object", "region", "claim"):
        assert len({j[field] for j in jobs}) == 1


@pytest.mark.parametrize("status", ["supported", "contradicted", "cannot_determine"])
def test_valid_status_has_no_character_truncation(status):
    result = {"evidence": "Visible details " * 30, "status": status, "uncertainty": "Blur hides the region"}
    assert validate_presence(result) == result
    assert "maxLength" not in str(ObjectPresence.model_json_schema())


@pytest.mark.parametrize("value", [
    {"evidence": ",", "status": "supported", "uncertainty": ""},
    {"evidence": "Image is blurred", "status": "cannot_determine", "uncertainty": ","},
    {"evidence": "Clearly visible", "status": "yes", "uncertainty": ""},
    {"evidence": "Clearly visible", "status": "supported", "uncertainty": "", "confidence": 1},
])
def test_invalid_contract_is_not_repaired(value):
    with pytest.raises(ValueError):
        validate_presence(value)


def test_duplicate_and_unsafe_case_ids_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        case_jobs([sample(), sample()])
    with pytest.raises(ValueError, match="safe"):
        case_jobs([{**sample(), "case_id": "../../bad"}])


def test_second_claim_cannot_change_source_query():
    first = sample()
    second = {**first, "case_id": "p02", "qid": 881}
    sources = {first["image_key"]: {"qid": first["qid"]}}
    validate_case_sources([first, {**second, "qid": first["qid"]}], sources)
    with pytest.raises(ValueError, match="identity"):
        validate_case_sources([first, second], sources)


def ledger(tmp_path, seconds=900):
    return PresenceLedger(tmp_path / "ledger.json", deadline=time.time() + seconds)


def test_multimodal_token_cap_before_reservation(tmp_path):
    value = ledger(tmp_path)
    with pytest.raises(BudgetStop, match="2048"):
        value.reserve("a", 1889, 160, ["direct"], "hash", 2)
    assert value.state["attempts"] == []


def test_global_cost_counts_both_arms_and_lost_attempts(tmp_path):
    value = ledger(tmp_path)
    for i in range(60):
        value.reserve(str(i), 1888, 160, ["direct" if i % 2 else "evidence_first"], str(i), 2)
    assert value.cost() == MAX_TOTAL
    with pytest.raises(BudgetStop, match="122880"):
        value.reserve("extra", 1, 1, ["direct"], "hash", 2)


def test_zero_retry_and_shutdown_limits(tmp_path):
    value = ledger(tmp_path)
    number = value.reserve("a", 100, 160, ["direct"], "hash", 2)
    value.settle(number, {"status": "runtime_error", "error": "lost"})
    assert value.cost() == 260
    with pytest.raises(BudgetStop, match="request budget"):
        value.reserve("a", 100, 160, ["direct"], "hash", 2)
    value.deadline = time.time() + 59
    with pytest.raises(BudgetStop, match="shutdown"):
        value.reserve("b", 100, 160, ["direct"], "other", 2)


def test_unknown_arm_cannot_avoid_charging(tmp_path):
    value = ledger(tmp_path)
    with pytest.raises(ValueError, match="arm"):
        value.reserve("a", 100, 160, ["free"], "hash", 2)
    assert value.state["attempts"] == []
