"""Actual R2 failures: source-span drift, numerical fields, and empty conflict plans."""

import json

import jsonschema
import pytest
import xgrammar
from backend import citation_schema, compact_grammar
from r2_evaluate import stage_gate
from r2_planner import resolve_plan
from r2_protocol import ARMS
from schema import Comparison
from task_grounding import numeric_options, quote_spans, task_schema, validate_grounded_task


@pytest.fixture(scope="module")
def compiler():
    info = xgrammar.TokenizerInfo([bytes([i]) for i in range(256)] + [b"<eos>"], stop_token_ids=[256])
    return xgrammar.GrammarCompiler(info, max_threads=2)


@pytest.mark.parametrize(
    "query,kind,quote,number",
    [
        ("Two men ride a bus together.", "object", "Two men", 2),
        ("Two girls show they're room on a train", "object", "Two girls", 2),
        (
            "Tourist walking around early in the morning at  Singapore streets",
            "event",
            "Tourist walking around early in the morning at Singapore streets",
            None,
        ),
        ("A man is browsing through a souvenir shop with clothing and trinkets in it.", "object", "a man", None),
        ("A woman is shaking a juice blend she made in the blender.", "object", "a woman", None),
        ("Woman walks around her hotel room after enterring it.", "event", "after entering it", None),
        ("Ski field view during winter as people freestyle on the snow.", "duration_seconds", "during winter", None),
    ],
)
def test_original_seven_invalid_outputs_blocked_before_decoding(compiler, query, kind, quote, number):
    value = {
        "request_type": "specific",
        "purpose": "watch the requested event",
        "constraints": [{"kind": kind, "quote": quote, "number": number}],
    }
    schema = task_schema(query)
    grammar = compiler.compile_grammar(compact_grammar(schema))
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(value, schema)
    assert not xgrammar.GrammarMatcher(grammar).accept_string(json.dumps(value, separators=(",", ":")))
    value["constraints"] = [{"kind": "event", "quote": query, "number": None}]
    assert xgrammar.GrammarMatcher(grammar).accept_string(json.dumps(value, separators=(",", ":")))
    assert validate_grounded_task(value, query)["query"] == query


def test_explicit_output_limits_grounded_and_object_counts_not_reinterpreted():
    assert numeric_options("Two men ride a bus together.") == []
    query = "Show 2 goals under 10 seconds"
    value = {"request_type": "specific", "purpose": "watch goals", "constraints": numeric_options(query)}
    jsonschema.validate(value, task_schema(query))
    task = validate_grounded_task(value, query)
    assert {c["kind"]: c["number"] for c in task["constraints"]} == {"count": 2, "duration_seconds": 10}
    assert numeric_options("Select top-three moments within two minutes")[-1]["number"] == 120
    assert "at  Singapore" in quote_spans("early at  Singapore streets")


def test_empty_planner_gets_auditable_bilateral_conflict_probes_only():
    candidates = [
        {"candidate": c, "target": [2.0 * c, 2.0 * c + 2], "allowed": [2.0 * c - 4, 2.0 * c + 6]} for c in [38, 39, 42]
    ]
    gaps = [{"gap_id": f"g{c}_42_conflict", "origin": "direction_conflict", "candidates": [c, 42]} for c in [38, 39]]
    result = resolve_plan({"actions": []}, gaps, candidates, {"query": "People stand by the pool in swimsuits."})
    assert result["source"] == "controller_conflict_probe"
    assert [a["candidate"] for a in result["actions"]] == [42, 38]
    assert all(a["gap"] == "attribution" for a in result["actions"])
    assert [(a["start"], a["end"]) for a in result["actions"]] == [(83, 87), (75, 79)]
    for non_conflicts in [[], [{"gap_id": "g", "origin": "insufficient", "candidates": [38, 42]}]]:
        assert not resolve_plan({"actions": []}, non_conflicts, candidates, {"query": "watch"})["actions"]


def test_valid_model_action_is_not_overridden():
    actions = [{"candidate": 0, "question": "Was the ball visibly caught?"}]
    assert resolve_plan({"actions": actions}, [], [], {}) == {"actions": actions, "source": "model", "rule": None}


def test_zero_observation_cannot_pass_confirmation_even_with_precision_gain():
    results = {a: {"unrounded_VeryGood": {"mAP": 40.0, "Hit1": 70.0}, "fallback_queries": 0} for a in ["E0", *ARMS]}
    results["E_targeted"]["unrounded_VeryGood"]["mAP"] = 42
    assert stage_gate(results, True, True, True, 0)["status"] == "not_passed"
    assert stage_gate(results, True, True, True, 1)["status"] == "passed"


def test_mixed_old_and_observation_ids_cannot_recombine_prefix_and_suffix(compiler):
    references = ["c42_f1", "obs0_c42_f0", "obs0_c42_f1", "obs0_c42_f2"]
    schema = citation_schema(Comparison, {"a_evidence": ["c40_f5"], "b_evidence": references})
    compiled = compiler.compile_grammar(compact_grammar(schema))
    value = {
        "a_eligibility": "satisfied",
        "b_eligibility": "satisfied",
        "preference": "tie",
        "values": ["action"],
        "reason": "Both show an action.",
        "a_evidence": ["c40_f5"],
        "b_evidence": [],
        "missing_question": None,
    }
    for identity in references + ["c42_f0", "c42_f2", "c40_f1", "obs0_c40_f5"]:
        value["b_evidence"] = [identity]
        accepted = xgrammar.GrammarMatcher(compiled).accept_string(json.dumps(value, separators=(",", ":")))
        assert accepted == (identity in references)
