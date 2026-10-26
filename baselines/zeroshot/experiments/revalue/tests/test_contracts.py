"""Behavioral checks for the failure modes explicitly required by R0."""

import copy
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from budget import BudgetStop, Ledger
from comparison import rerank, validate_comparison
from controller import Controller
from exporters import benchmark_prediction, user_delivery
from io_utils import digest
from memory import FactStore, build_memory, judgment, retrieve, valid_judgment
from observer import paired_indices, validate_action
from schema import numeric_limit, validate_task
from selector import select_events


class CharacterTokenizer:
    def encode(self, text, **kwargs):
        return list(text)


def task(query="Show 2 goals under 10 seconds"):
    return validate_task(
        {
            "request_type": "specific",
            "purpose": "watch goals",
            "constraints": [
                {"kind": "count", "quote": "2", "number": 2.0},
                {"kind": "duration_seconds", "quote": "10 seconds", "number": 10.0},
            ],
        },
        query,
    )


def event(identity, start=0.0, end=4.0):
    return {
        "event_id": identity,
        "start": start,
        "end": end,
        "eligibility": "satisfied",
        "evidence": ["s1"],
        "boundary_confirmed": True,
        "boundary_range": None,
        "necessary_contents": [{"content": "visible result", "supported": True}],
    }


@pytest.mark.parametrize("kind", ["generic", "broad", "specific"])
def test_request_types_do_not_invent_count_or_duration(kind):
    value = validate_task({"request_type": kind, "purpose": "watch", "constraints": []}, "Show this")
    assert numeric_limit(value, "count") is None
    assert numeric_limit(value, "duration_seconds") is None


def test_constraints_require_query_provenance():
    with pytest.raises(ValueError, match="provenance"):
        task("Show interesting clips")


def test_purpose_change_invalidates_judgment_not_facts():
    store = FactStore()
    store.extract(
        1, {"facts": [{"kind": "outcome", "text": "Ball enters net", "frames": ["s7"]}], "unknown": []}, ["s7"]
    )
    before = store.version
    first = task()
    record = judgment(first, store.dump(), {"preference": "tie"})
    second = copy.deepcopy(first)
    second["purpose"] = "watch crowd reactions"
    assert valid_judgment(record, first, store.dump())
    assert not valid_judgment(record, second, store.dump())
    assert store.version == before
    modified = store.dump()
    modified["c1_f0"]["text"] = "Goal missed"
    assert store.version == before
    assert not valid_judgment(record, first, modified)
    with pytest.raises(ValueError, match="overwritten"):
        store.extract(
            1, {"facts": [{"kind": "outcome", "text": "Goal missed", "frames": ["s7"]}], "unknown": []}, ["s7"]
        )


def test_no_foreign_fact_citation():
    with pytest.raises(ValueError, match="foreign"):
        FactStore().extract(1, {"facts": [{"kind": "event", "text": "X", "frames": ["s999"]}], "unknown": []}, ["s7"])


def test_capacity_evicts_whole_entries_and_blocks_cache_bypass():
    facts = {f"c{i}_f0": {"candidate": i, "frames": [f"s{i}"], "kind": "event", "text": "x"} for i in range(5)}
    draft = {"items": [{"text": "details " * 12, "fact_ids": [k]} for k in facts]}
    memory = build_memory(draft, facts, CharacterTokenizer(), capacity=500)
    assert 0 < len(memory["retained_fact_ids"]) < 5
    assert memory["deleted_fact_ids"]
    assert memory["tokens"] <= 500
    assert len(retrieve(memory, CharacterTokenizer())["items"]) == len(memory["retained_fact_ids"])
    memory["view"]["items"][0]["text"] += " smuggled audit history"
    with pytest.raises(ValueError, match="mutated"):
        retrieve(memory, CharacterTokenizer())


def comparison(state="tie"):
    return {
        "a_eligibility": "satisfied",
        "b_eligibility": "satisfied",
        "preference": state,
        "values": ["action"],
        "reason": "Both show the requested action",
        "a_evidence": ["s1"],
        "b_evidence": ["s2"],
        "missing_question": "Is the outcome visible?" if state == "insufficient" else None,
    }


def test_comparator_keeps_tie_and_insufficient_separate():
    ids = {1: ["s1"], 2: ["s2"]}
    for state in ("tie", "insufficient"):
        assert validate_comparison(comparison(state), 1, 2, ids, {})["preference"] == state
    bad = comparison("insufficient")
    bad["missing_question"] = None
    with pytest.raises(ValueError, match="question"):
        validate_comparison(bad, 1, 2, ids, {})


def test_comparison_rejects_wrong_event_attribution():
    bad = comparison()
    bad["a_evidence"] = ["s2"]
    with pytest.raises(ValueError, match="wrong candidate"):
        validate_comparison(bad, 1, 2, {1: ["s1"], 2: ["s2"]}, {})
    bad = comparison("a_better")
    bad["a_eligibility"] = "violated"
    with pytest.raises(ValueError, match="ineligible"):
        validate_comparison(bad, 1, 2, {1: ["s1"], 2: ["s2"]}, {})


def test_double_direction_agreement_and_score_multiset():
    original, selected = [0.9, 0.8, 0.7, 0.6, 0.5, 0.1], list(range(5))
    values = {(a, b): "tie" for a, b in itertools.permutations(selected, 2)}
    values[0, 1], values[1, 0] = "insufficient", "tie"
    values[0, 2], values[2, 0] = "b_better", "a_better"
    values[0, 3], values[3, 0] = "a_better", "a_better"
    records = [{"a": a, "b": b, "result": {"preference": v}} for (a, b), v in values.items()]
    scores, info = rerank(original, selected, records)
    edges = {(e["a"], e["b"]): e for e in info["edges"]}
    assert edges[0, 1]["state"] == edges[0, 3]["state"] == "unresolved"
    assert edges[1, 2]["state"] == "tie"
    assert edges[0, 2]["winner"] == 2
    assert scores[2] == 0.9 and sorted(scores) == sorted(original) and scores[-1] == 0.1
    with pytest.raises(ValueError):
        rerank(original, selected, records[:-1])


def test_paired_new_frames_exclude_entire_baseline_and_match_counts():
    pts = np.column_stack([np.arange(300) / 30, (np.arange(300) + 1) / 30])
    seen = set(range(0, 300, 15))  # Includes B0 frames far outside a candidate's four-frame subset.
    result = paired_indices(pts, [[0, 10]], [[1, 2]], seen, count=24)
    assert len(result["uniform"]) == len(result["targeted"]) == 24
    assert not (set(result["uniform"]) | set(result["targeted"])) & seen
    result = paired_indices(pts, [[0, 10]], [[1, 1.01]], seen)
    assert result["cancelled"] and not result["uniform"] and not result["targeted"]


def test_stitch_preserves_disjoint_original_ranges_and_rejects_outside():
    action = {
        "gap": "attribution",
        "tool": "Stitch",
        "question": "Same event?",
        "intervals": [[10.0, 12.0], [30.0, 32.0]],
        "frame_limit": 24,
    }
    assert validate_action(action, 40)["intervals"] == action["intervals"]
    with pytest.raises(ValueError):
        validate_action(action, 31)
    with pytest.raises(ValueError):
        validate_action({**action, "tool": "Segment"}, 40)


def test_interval_expansion_reselects_instead_of_truncating(tmp_path):
    events = [event("a"), event("b", 10, 14), event("c", 20, 23)]
    control = Controller(tmp_path / "state.json", events, task())
    first = control.decide(["a", "b", "c"])
    assert [e["event_id"] for e in first["events"]] == ["a", "b"]
    second = control.decide(["a", "b", "c"], [{"event_id": "a", "end": 7.0}])
    assert [e["event_id"] for e in second["events"]] == ["a", "c"]
    assert second["events"][0]["end"] == 7 and second["duration_seconds"] == 10


def test_attribution_and_boundary_unknown_are_not_success():
    a, b = event("a"), event("b", 10, 14)
    b["attribution_conflict"] = True
    result = select_events([a, b], ["a", "b"], task())
    assert result["status"] == "partial" and not result["count_satisfied"]
    a.update(boundary_confirmed=False, boundary_range=[0, 2])
    result = select_events([a], ["a"], task())
    assert result["status"] == "abstain" and result["unresolved"][0]["boundary_range"] == [0, 2]


def test_only_confirmed_duplicates_are_deduplicated():
    a, b = event("a"), event("b", 1, 5)
    assert len(select_events([a, b], ["a", "b"], task())["events"]) == 2
    a["confirmed_duplicate_group"] = b["confirmed_duplicate_group"] = "same_goal"
    result = select_events([a, b], ["a", "b"], task())
    assert len(result["events"]) == 1 and result["status"] == "partial"


def test_evidence_backed_candidate_addition_and_round_limit(tmp_path):
    initial = [event("a")]
    control = Controller(tmp_path / "state.json", initial, task())
    control.add_event(event("b", 10, 14))
    assert control.decide(["a", "b"])["status"] == "complete"
    control.add_event(event("c", 20, 23))
    with pytest.raises(ValueError):
        control.add_event(event("d"))
    resumed = Controller(tmp_path / "state.json", initial, task())
    resumed.decide(["a", "b", "c"])
    resumed.decide(["a", "b", "c"])
    with pytest.raises(ValueError, match="round budget"):
        resumed.decide(["a", "b", "c"])


def test_cached_observations_still_consume_persistent_display_budget(tmp_path):
    initial = [event("a")]
    for i in range(4):
        controller = Controller(tmp_path / "state.json", initial, task())
        controller.reserve_observation(str(i), [f"s{x}" for x in range(24)], cached=True)
    resumed = Controller(tmp_path / "state.json", initial, task())
    assert resumed.state["displayed_frames"] == 96
    with pytest.raises(ValueError, match="budget"):
        resumed.reserve_observation("4", ["s1", "s2"], cached=True)


def test_failed_attempt_and_shared_cost_survive_resume(tmp_path):
    limits = {"requests": 2, "extra_attempts": 1, "arm_incremental_tokens": {"a": 100, "b": 100}}
    path = tmp_path / "ledger.json"
    ledger = Ledger(path, limits)
    first = ledger.reserve("q/task", 10, 20, ["a", "b"], "hash", 0)
    ledger.settle(first, {"status": "runtime_error"})
    resumed = Ledger(path, limits)
    assert resumed.cost("a") == resumed.cost("b") == 30
    second = resumed.reserve("q/task", 10, 20, ["a", "b"], "hash", 0)
    resumed.settle(second, {"status": "invalid", "prompt_tokens": 10, "output_tokens": 5})
    assert resumed.cost("a") == 45
    with pytest.raises(ValueError, match="semantic"):
        resumed.reserve("q/task", 10, 20, ["a", "b"], "hash", 0)
    with pytest.raises(BudgetStop, match="token"):
        resumed.reserve("q/other", 50, 10, ["a"], "hash2", 0)
    with pytest.raises(ValueError, match="cannot change"):
        Ledger(path, {**limits, "requests": 100})


def test_exporter_does_not_fill_event_score_over_grid():
    row = {"qid": 1, "vid": "v", "query": "q"}
    original = [0.9, 0.8, 0.7, 0.6, 0.5, 0.1]
    with pytest.raises(ValueError):
        benchmark_prediction(row, [0.9] * 6, original, list(range(5)))
    selected = select_events([event("a")], ["a"], task())
    assert user_delivery(selected, "trace.json")["benchmark_score_vector"] is None
    assert digest(original) == digest([0.9, 0.8, 0.7, 0.6, 0.5, 0.1])
