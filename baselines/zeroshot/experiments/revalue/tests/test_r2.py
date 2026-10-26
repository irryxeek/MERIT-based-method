"""R2 controls: paired observations, bounded memory, exact billing and concurrency."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import r2_observer
import r2_worker
from budget import BudgetStop
from io_utils import compact, write_json
from memory import build_memory, retrieve
from prompts import OBSERVE
from r2_evaluate import audit_observation_request, stage_gate
from r2_planner import LocalPlan, actionable_gaps, candidate_ranges, update_memory, validate_plan
from r2_protocol import ARMS, OBSERVERS, SharedLedger, claim_query
from schema import Comparison
from test_workflow import FakeBackend, fixture


def limits(requests=100):
    return {"requests": requests, "extra_attempts": 4, "arm_incremental_tokens": {a: 10000 for a in ARMS}}


def test_global_ledger_atomic_across_workers(tmp_path):
    SharedLedger(tmp_path, limits())

    def invoke(i):
        ledger = SharedLedger(tmp_path, limits())
        number = ledger.reserve(str(i), 2, 5, ["M_decision"], str(i), 0)
        ledger.settle(number, {"status": "complete", "prompt_tokens": 2, "output_tokens": 3})
        return number

    with ThreadPoolExecutor(max_workers=8) as pool:
        numbers = list(pool.map(invoke, range(24)))
    ledger = SharedLedger(tmp_path, limits())
    assert sorted(numbers) == list(range(24))
    assert ledger.cost() == 120
    assert all(ledger.cost(arm) == 120 for arm in ["E_memory", *OBSERVERS])
    assert ledger.cost("C_direct4") == 0


def test_global_budget_cannot_be_oversubscribed(tmp_path):
    SharedLedger(tmp_path, limits(1))

    def invoke(i):
        try:
            return SharedLedger(tmp_path, limits(1)).reserve(str(i), 2, 5, ["C_direct4"], str(i), 0)
        except BudgetStop:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(invoke, [0, 1]))
    assert results.count(None) == 1
    assert len(SharedLedger(tmp_path, limits(1)).state["attempts"]) == 1


def test_reused_request_cost_counts_for_each_descendant_but_not_new_gpu_work(tmp_path):
    ledger = SharedLedger(tmp_path, limits())
    entry = {"charged_tokens": 100, "original_arms": ["M_plain", "M_decision"]}
    ledger.record_reuse("facts", entry)
    ledger.record_reuse("facts", entry)
    assert ledger.cost() == ledger.cost("C_direct4") == 0
    assert all(ledger.cost(arm) == 100 for arm in ["E_memory", *OBSERVERS])


def test_reuse_cannot_bypass_token_budget(tmp_path):
    ledger = SharedLedger(tmp_path, limits())
    with pytest.raises(BudgetStop):
        ledger.record_reuse("facts", {"charged_tokens": 10001, "original_arms": ["M_decision"]})
    assert not ledger.state.get("reused")


def test_query_claim_prevents_parallel_duplicate_and_completed_replay(tmp_path):
    with claim_query(tmp_path, 1) as first:
        assert first
        with claim_query(tmp_path, 1) as second:
            assert not second
    with claim_query(tmp_path, 1) as resumed:
        assert resumed
        write_json(tmp_path / "records/1.json", {"done": True})
    with claim_query(tmp_path, 1) as complete:
        assert not complete


def test_planning_cannot_invent_gaps_or_leave_candidate_neighborhood():
    gaps = [{"gap_id": "g", "candidates": [0, 1], "question": "Does the attempt visibly succeed?"}]
    candidates = candidate_ranges([0, 1, 2, 3, 4], 20.0)
    valid = {
        "actions": [
            {"gap_id": "g", "gap": "value", "question": gaps[0]["question"], "candidate": 0, "start": 1.0, "end": 2.0}
        ]
    }
    assert validate_plan(valid, gaps, candidates)["actions"][0]["question"] == gaps[0]["question"]
    for change in ({"candidate": 2}, {"end": 7.0}, {"gap_id": "invented"}):
        with pytest.raises(ValueError):
            validate_plan({"actions": [{**valid["actions"][0], **change}]}, gaps, candidates)


def test_paired_frames_exclude_whole_baseline_and_prior_new_frames():
    times = np.stack([np.arange(100) / 10, np.arange(1, 101) / 10], axis=1)
    actions = [{"allowed": [0.0, 6.0], "start": 1.0, "end": 2.0}, {"allowed": [0.0, 6.0], "start": 1.0, "end": 3.0}]
    seen = set(range(0, 100, 3))
    pairs = r2_observer.pair_actions(times, actions, seen)
    for arm in ("uniform", "targeted"):
        used = set()
        for pair in pairs:
            assert len(pair["uniform"]) == len(pair["targeted"]) <= 24
            assert not set(pair[arm]) & (seen | used)
            used.update(pair[arm])


def test_infeasible_target_cancels_both_observation_arms():
    times = np.stack([np.arange(10), np.arange(1, 11)], axis=1)
    pair = r2_observer.pair_actions(times, [{"allowed": [0.0, 10.0], "start": 0.0, "end": 1.0}], set())[0]
    assert pair["cancelled"] and not pair["uniform"] and not pair["targeted"]


def test_observation_audit_checks_actual_question_pts_and_frame_allocation():
    times = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])
    action = {"question": "Does the action complete?"}
    frames = [{"id": f"s{i}", "frame_index": i, "pts": times[i].tolist()} for i in (1, 2)]
    text = OBSERVE.format(
        question=action["question"], frames=compact([{"id": f["id"], "pts": f["pts"]} for f in frames])
    )
    raw = {"specification": {"text": text, "evidence": [{"action": action, "frames": frames}]}}
    assert audit_observation_request(raw, action, [1, 2], times, {0}) == 2
    with pytest.raises(ValueError):
        audit_observation_request(raw, action, [1, 2], times, {1})
    with pytest.raises(ValueError):
        audit_observation_request(raw, action, [0, 1], times, set())
    raw["specification"]["text"] = "A different question"
    with pytest.raises(ValueError):
        audit_observation_request(raw, action, [1, 2], times, {0})


def test_memory_update_has_no_access_to_evicted_old_facts():
    class Characters:
        def encode(self, value, **kwargs):
            return list(value)

    tok = Characters()
    facts = {str(i): {"candidate": i, "frames": [f"s{i}"], "kind": "event"} for i in range(3)}
    memory = build_memory({"items": [{"text": "x" * 450, "fact_ids": [str(i)]} for i in range(3)]}, facts, tok)
    assert memory["deleted_fact_ids"]
    obs = [
        {
            "action": {"candidate": 1},
            "result": {
                "facts": [{"kind": "event", "text": "A visible action completes.", "frames": ["s100"]}],
                "unknown": [],
            },
        }
    ]
    updated = update_memory(memory, obs, tok)
    assert updated["tokens"] <= 1024
    assert not set(memory["deleted_fact_ids"]) & set(updated["retained_fact_ids"])
    assert retrieve(updated, tok)["items"][0]["fact_ids"] == ["obs0_c1_f0"]


class R2Backend(FakeBackend):
    def __init__(self, observe=False, fail=None):
        super().__init__(fail)
        self.observe = observe

    def call(self, key, text, videos, evidence, schema, cap, arms, validate, references=None, output_schema=None):
        if schema is LocalPlan:
            self.calls.append(key)
            gap = evidence["gaps"][0]
            value = {
                "actions": [
                    {
                        "gap_id": gap["gap_id"],
                        "gap": "value",
                        "question": gap["question"] or "Is the action completed visibly within this target?",
                        "candidate": gap["candidates"][0],
                        "start": float(t),
                        "end": float(t + 1),
                    }
                    for t in (1, 3)
                ]
            }
            return validate(value), {"attempt": len(self.calls) - 1, "response_path": "fixture"}
        value, raw = super().call(key, text, videos, evidence, schema, cap, arms, validate, references)
        if schema is Comparison and "/M_decision/" in key and self.observe:
            if self.observe == "conflict":
                value.update(preference="a_better", missing_question=None)
            else:
                value.update(
                    preference="insufficient", missing_question="Does the action finish with a visible result?"
                )
            value = validate(value)
        return value, raw


def r2_fixture(monkeypatch):
    row, frozen, _ = fixture(monkeypatch)
    import worker

    baseline, pair = worker.parents({}, row)
    baseline["media"]["frame_indices"] = [[0, 1, 2, 3, 4, 50]]
    monkeypatch.setattr(r2_worker, "parents", lambda *args: (baseline, pair))
    times = np.stack([np.arange(200) / 10, np.arange(1, 201) / 10], axis=1)
    seen = {0, 1, 2, 3, 4, 50}
    frozen.update(source_timeline_sha256=r2_observer.timeline_hash(times), B0_seen_indices=sorted(seen))
    monkeypatch.setattr(r2_worker, "source_timeline", lambda *args: (None, times, seen))

    def read(reader, timestamps, ids, row, action, hw, arm):
        trace = {
            "frames": [{"id": f"s{i}", "frame_index": i, "pts": timestamps[i].tolist()} for i in ids],
            "action": action,
            "new_source_frames": len(ids),
            "displayed_frames": len(ids),
        }
        return {"video": np.zeros((len(ids), 32, 32, 3), dtype=np.uint8)}, trace

    monkeypatch.setattr(r2_worker, "read_frames", read)
    return row, frozen


def test_supported_ties_do_not_trigger_observation_or_planner(monkeypatch):
    row, frozen = r2_fixture(monkeypatch)
    backend = R2Backend()
    result = r2_worker.one_query(row, {}, "binding", backend, frozen)
    assert len(backend.calls) == 47
    assert not result["planning"]["called"]
    assert not actionable_gaps(result["arms"]["E_memory"])
    assert all(result["arms"][a]["status"] == "complete" for a in ARMS)
    assert result["arms"]["E_uniform"]["scores"] == result["arms"]["E_memory"]["scores"]


def test_full_r2_wiring_uses_92_calls_and_two_paired_observations(monkeypatch):
    row, frozen = r2_fixture(monkeypatch)
    backend = R2Backend(observe=True)
    result = r2_worker.one_query(row, {}, "binding", backend, frozen)
    assert len(backend.calls) == 92
    assert all(result["arms"][a]["status"] == "complete" for a in ARMS)
    assert all(len(result["observations"][a]) == 2 for a in OBSERVERS)
    for u, t in zip(result["observations"]["E_uniform"], result["observations"]["E_targeted"], strict=True):
        assert u["action"]["question"] == t["action"]["question"]
        assert u["trace"]["new_source_frames"] == t["trace"]["new_source_frames"]


def test_directional_conflicts_enter_question_planner(monkeypatch):
    row, frozen = r2_fixture(monkeypatch)
    backend = R2Backend(observe="conflict")
    result = r2_worker.one_query(row, {}, "binding", backend, frozen)
    assert len(backend.calls) == 92
    assert all(g["origin"] == "direction_conflict" for g in result["planning"]["gaps"])
    assert all(a["question"] for a in result["planning"]["actions"])
    assert all(result["arms"][a]["status"] == "complete" for a in ARMS)


def test_observation_semantic_failure_cancels_both_and_preserves_parent(monkeypatch):
    row, frozen = r2_fixture(monkeypatch)
    backend = R2Backend(observe=True, fail="/E_uniform/observe/")
    result = r2_worker.one_query(row, {}, "binding", backend, frozen)
    assert len(backend.calls) == 50
    for arm in OBSERVERS:
        assert result["arms"][arm]["status"] == "fallback"
        assert result["arms"][arm]["parent_arm"] == "E_memory"
        assert result["arms"][arm]["scores"] == result["arms"]["E_memory"]["scores"]
    assert result["arms"]["E_memory"]["status"] == "complete"


def test_confirmation_requires_precision_pair_balance_and_complete_budget():
    results = {a: {"unrounded_VeryGood": {"mAP": 40.0, "Hit1": 70.0}, "fallback_queries": 0} for a in ["E0", *ARMS]}
    results["E_targeted"]["unrounded_VeryGood"]["mAP"] = 41.0
    assert stage_gate(results, True, True, True, 1)["status"] == "passed"
    for flags in [(False, True, True), (True, False, True), (True, True, False)]:
        assert stage_gate(results, *flags, observed_queries=1)["status"] == "not_passed"
    results["E_targeted"]["unrounded_VeryGood"]["mAP"] = 40.99999
    assert stage_gate(results, True, True, True, 1)["status"] == "not_passed"
