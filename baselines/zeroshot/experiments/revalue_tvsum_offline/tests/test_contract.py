"""CPU contracts for a narrow planned input change, not model predictions."""

import importlib.util
import os
import sys
from copy import deepcopy
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "audit.py"
SPEC = importlib.util.spec_from_file_location("tvsum_offline_audit", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def task():
    return {"query": audit.QUERY, "request_type": "generic", "purpose": "Highlight of this video",
            "constraints": [deepcopy(audit.BAD_CONSTRAINT)]}


@pytest.fixture(scope="module")
def contracts():
    parent = os.environ.get("TVSUM_PARENT")
    if not parent:
        pytest.fail("TVSUM_PARENT must point to the completed frozen transfer; no contract check is skipped")
    return audit.load_contracts(Path(parent) / "source_snapshot")


def test_change_only_constraint_and_do_not_mutate_source():
    old = task()
    changed = audit.remove_exact_generic_object(old)
    assert old == task()
    assert changed["constraints"] == []
    assert {k for k in old if old[k] != changed[k]} == {"constraints"}


@pytest.mark.parametrize("update", [
    {"request_type": "specific"},
    {"request_type": "broad"},
    {"query": "Show highlights of a red car."},
    {"constraints": []},
    {"constraints": [{"kind": "object", "quote": "video", "number": None}]},
    {"constraints": [audit.BAD_CONSTRAINT, {"kind": "count", "quote": "top 3", "number": 3.0}]},
    {"unexpected": True},
])
def test_refuse_to_erase_other_requests_or_constraints(update):
    old = task()
    old.update(update)
    with pytest.raises(ValueError):
        audit.remove_exact_generic_object(old)


def test_spec_patch_preserves_all_non_task_inputs():
    old = {"text": "TaskSpec: " + audit.compact(task()),
           "rendered_prompt": "system\nTaskSpec: " + audit.compact(task()) + "\nassistant",
           "schema": {"anyOf": []}, "evidence": [{"model_rgb_sha256": "same-old-pixels"}],
           "output_cap": 384, "structured_output": {"grammar_sha256": "same-grammar"}}
    saved = deepcopy(old)
    changed = audit.patch_specification(old, task())
    assert old == saved
    for key in ("schema", "evidence", "output_cap", "structured_output"):
        assert changed[key] == old[key]
    assert audit.compact(audit.remove_exact_generic_object(task())) in changed["text"]


@pytest.mark.parametrize("text", ["no task here", audit.compact(task()) * 2])
def test_ambiguous_prompt_binding_is_rejected(text):
    with pytest.raises(ValueError):
        audit.patch_specification({"text": text, "rendered_prompt": text}, task())


def test_frozen_task_validator_accepts_semantically_bad_and_empty_constraints(contracts):
    for value in (task(), audit.remove_exact_generic_object(task())):
        assert contracts["validate_task"]({k: v for k, v in value.items() if k != "query"}, audit.QUERY) == value


def comparison(preference):
    return {"a_eligibility": "unknown", "b_eligibility": "unknown", "preference": preference,
            "values": [], "reason": "Synthetic contract example only.",
            "a_evidence": ["s1"], "b_evidence": ["s2"],
            "missing_question": "no actionable question" if preference == "insufficient" else None}


@pytest.mark.parametrize("preference", ["a_better", "b_better", "tie", "insufficient"])
def test_unknown_does_not_mechanically_force_tie(contracts, preference):
    value = comparison(preference)
    assert contracts["validate_comparison"](value, 0, 1, {0: ["s1"], 1: ["s2"]}, {}) == value


def test_no_evidence_or_violated_winner_is_rejected(contracts):
    value = comparison("a_better")
    value["a_eligibility"] = "violated"
    with pytest.raises(ValueError):
        contracts["validate_comparison"](value, 0, 1, {0: ["s1"], 1: ["s2"]}, {})
    value = comparison("tie")
    value["b_evidence"] = []
    with pytest.raises(ValueError):
        contracts["validate_comparison"](value, 0, 1, {0: ["s1"], 1: ["s2"]}, {})


def test_output_schema_allows_unknown_winner_without_task_dependency(contracts):
    schema = contracts["citation_schema"](contracts["Comparison"], {"a_evidence": ["s1"], "b_evidence": ["s2"]})
    for preference, side in (("a_better", "a"), ("b_better", "b")):
        branch = next(b for b in schema["anyOf"] if b["properties"]["preference"]["const"] == preference)
        assert "unknown" in branch["properties"][side + "_eligibility"]["enum"]


def test_pure_contract_loader_does_not_import_model_runtime(contracts):
    assert "Backend" not in contracts
    assert not {"torch", "transformers", "vllm", "openai", "anthropic", "pynvml"} & set(sys.modules)
