"""Regression cases from the failed real R0: truncation, references, and reuse."""

import copy
import json
from types import SimpleNamespace

import backend
import jsonschema
import pytest
import xgrammar
from backend import Backend, citation_schema, compact_grammar, reuse_response
from budget import Ledger
from comparison import validate_comparison
from io_utils import digest, sha256, write_json
from repair import monitor_startup_recovery, observation_content_recovery, worker_remaining
from schema import Comparison, ObservationFacts
from worker import finish


def compared(preference="b_better"):
    return {
        "a_eligibility": "unknown",
        "b_eligibility": "satisfied",
        "preference": preference,
        "values": ["action"],
        "reason": "B shows the requested action; A remains unclear.",
        "a_evidence": ["s1"],
        "b_evidence": ["s2"],
        "missing_question": None,
    }


@pytest.fixture(scope="module")
def compiler():
    info = xgrammar.TokenizerInfo([bytes([i]) for i in range(256)] + [b"<eos>"], stop_token_ids=[256])
    return xgrammar.GrammarCompiler(info, max_threads=2)


@pytest.fixture(scope="module")
def comparison_schema():
    return citation_schema(Comparison, {"a_evidence": ["s1"], "b_evidence": ["s2"]})


@pytest.mark.parametrize("preference", ["a_better", "b_better", "tie"])
def test_winner_or_tie_cannot_drop_one_side(comparison_schema, preference):
    value = compared(preference)
    jsonschema.validate(value, comparison_schema)
    for side in ("a", "b"):
        missing = copy.deepcopy(value)
        missing[side + "_evidence"] = []
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(missing, comparison_schema)


def test_grammar_preserves_unknown_and_insufficient(comparison_schema, compiler):
    grammar = compiler.compile_grammar(compact_grammar(comparison_schema))
    value = compared("insufficient")
    value.update(
        a_eligibility="unknown",
        b_eligibility="unknown",
        a_evidence=[],
        b_evidence=[],
        missing_question="Which target visibly completes the action?",
    )
    jsonschema.validate(value, comparison_schema)
    assert validate_comparison(value, 0, 1, {0: ["s1"], 1: ["s2"]}, {}) == value
    assert xgrammar.GrammarMatcher(grammar).accept_string(json.dumps(value, separators=(",", ":")))
    value["b_eligibility"] = "satisfied"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(value, comparison_schema)
    assert not xgrammar.GrammarMatcher(grammar).accept_string(json.dumps(value, separators=(",", ":")))


def test_grammar_enforces_references_and_compact_syntax(comparison_schema, compiler):
    grammar = compiler.compile_grammar(compact_grammar(comparison_schema))
    valid = compared()
    assert xgrammar.GrammarMatcher(grammar).accept_string(json.dumps(valid, separators=(",", ":")))
    assert not xgrammar.GrammarMatcher(grammar).accept_string(json.dumps(valid, indent=2))
    for mutation in ({"a_evidence": []}, {"a_evidence": ["s2"]}, {"b_eligibility": "violated"}):
        invalid = {**valid, **mutation}
        assert not xgrammar.GrammarMatcher(grammar).accept_string(json.dumps(invalid, separators=(",", ":")))


def test_observation_compact_output_keeps_source_ids_and_limits_redundancy(compiler):
    schema = citation_schema(ObservationFacts, {"ObservationFact.frames": ["s1200", "s1204"]})
    grammar = compiler.compile_grammar(compact_grammar(schema))
    value = {
        "facts": [{"kind": "event", "text": "A player kicks the ball towards the goal.", "frames": ["s1200", "s1204"]}],
        "unknown": ["The result is off screen."],
    }
    assert ObservationFacts.model_validate(value).model_dump() == value
    assert xgrammar.GrammarMatcher(grammar).accept_string(json.dumps(value, separators=(",", ":")))
    value["facts"][0]["text"] = "x" * 97
    with pytest.raises(ValueError):
        ObservationFacts.model_validate(value)
    assert not xgrammar.GrammarMatcher(grammar).accept_string(json.dumps(value, separators=(",", ":")))


def test_empty_observation_cannot_pass_but_cited_uncertainty_can(compiler):
    schema = citation_schema(ObservationFacts, {"ObservationFact.frames": ["s0"]})
    grammar = compiler.compile_grammar(compact_grammar(schema))
    empty = {"facts": [], "unknown": []}
    with pytest.raises(ValueError):
        ObservationFacts.model_validate(empty)
    assert not xgrammar.GrammarMatcher(grammar).accept_string(json.dumps(empty, separators=(",", ":")))
    unclear = {
        "facts": [
            {"kind": "unknown", "text": "The final outcome remains outside the visible frame.", "frames": ["s0"]}
        ],
        "unknown": ["Did the action complete?"],
    }
    assert ObservationFacts.model_validate(unclear).model_dump() == unclear
    assert xgrammar.GrammarMatcher(grammar).accept_string(json.dumps(unclear, separators=(",", ":")))


def test_r0_gate_rejects_empty_payload_despite_real_frame_trace(tmp_path):
    import json

    rows = [{"qid": i} for i in range(20)]
    write_json(tmp_path / "cpu_tests.json", {"returncode": 0})
    record = {
        "arms": {a: {"status": "complete"} for a in ["C_direct4", "M_plain", "M_decision"]},
        "observation": {
            "trace": {"frames": [{"id": f"s{i}"} for i in range(24)]},
            "result": {"facts": [], "unknown": []},
        },
    }
    for row in rows:
        write_json(tmp_path / "R0/records" / f"{row['qid']}.json", {**record, "qid": row["qid"]})
    ledger = Ledger(tmp_path / "R0/ledger.json", {"requests": 0, "extra_attempts": 0, "arm_incremental_tokens": {}})
    finish(tmp_path, "R0", {}, rows, "binding", ledger)
    gate = json.loads((tmp_path / "R0/gate.json").read_text())
    assert gate["status"] == "failed"
    assert gate["checks"]["twenty_real_24_frame_observations"]
    assert not gate["checks"]["nonempty_source_cited_observations"]
    assert not (tmp_path / "formal_protocol.json").exists()


def test_reuse_requires_exact_unchanged_request_and_immutable_success(tmp_path):
    path = tmp_path / "original.json"
    spec = {"text": "original prompt", "evidence": ["pixel-hash"], "output_cap": 512}
    record = {"status": "complete", "specification": spec, "request_sha256": digest(spec), "parsed": {"x": 1}}
    write_json(path, record)
    index = {"q/facts/0": {"path": str(path), "file_sha256": sha256(path), "record_sha256": digest(record)}}
    assert reuse_response(index, "q/facts/0", spec, lambda x: x) == ({"x": 1}, record)
    assert reuse_response(index, "q/observation", spec, lambda x: x) is None
    with pytest.raises(ValueError, match="exact unchanged"):
        reuse_response(index, "q/facts/0", {**spec, "text": "modified prompt"}, lambda x: x)
    write_json(path, {**record, "parsed": {"x": 2}})
    with pytest.raises(ValueError, match="file changed"):
        reuse_response(index, "q/facts/0", spec, lambda x: x)


def test_repair_carries_time_launch_and_request_cost_without_charging_reuse_twice(tmp_path):
    binding = {
        "maximum_active_worker_seconds_per_stage": 7200,
        "maximum_worker_attempts_per_stage": 3,
        "historical_usage": {"R0": {"active_seconds": 2500, "worker_launches": 2, "attempts": 300}},
    }
    assert worker_remaining(binding, "R0", []) == (4700, 1)
    assert worker_remaining(binding, "R0", [{"seconds": 600}]) == (4100, 0)
    assert worker_remaining(binding, "R1", []) == (7200, 3)
    limits = {"requests": 140, "extra_attempts": 60, "arm_incremental_tokens": {}}
    ledger = Ledger(tmp_path / "ledger.json", limits)
    ledger.record_reuse("q/task", {"path": "frozen.json", "hash": "fixed"})
    resumed = Ledger(tmp_path / "ledger.json", limits)
    resumed.record_reuse("q/task", {"path": "frozen.json", "hash": "fixed"})
    assert len(resumed.state["reused"]) == 1 and not resumed.state["attempts"]
    assert resumed.cost() == 0
    assert binding["historical_usage"]["R0"]["attempts"] + limits["requests"] + limits["extra_attempts"] == 500
    with pytest.raises(ValueError, match="changed"):
        resumed.record_reuse("q/task", {"path": "different.json", "hash": "changed"})


def test_monitor_failure_cannot_abort_a_valid_generated_response(tmp_path, monkeypatch):
    limits = {"requests": 1, "extra_attempts": 0, "arm_incremental_tokens": {}}
    runner = Backend.__new__(Backend)
    runner.ledger = Ledger(tmp_path / "ledger.json", limits)
    runner.directory = tmp_path
    runner.reuse_index = {}
    runner.request = lambda text, videos: {"prompt": text}
    generated = SimpleNamespace(
        prompt_token_ids=[1, 2],
        outputs=[
            SimpleNamespace(
                token_ids=list(range(20)), text=json.dumps(compared(), separators=(",", ":")), finish_reason="stop"
            )
        ],
    )
    # No collective_rpc method is available: diagnostics must remain outside it.
    runner.engine = SimpleNamespace(
        llm_engine=SimpleNamespace(
            input_processor=SimpleNamespace(
                input_preprocessor=SimpleNamespace(preprocess=lambda request: {"prompt_token_ids": [1, 2]})
            )
        ),
        generate=lambda *args, **kwargs: [generated],
    )
    monkeypatch.setattr(backend, "gpu_memory_probe", lambda: {"probe_error": "Fixture unavailable"})
    value, raw = runner.call(
        "q/compare",
        "prompt",
        [],
        [],
        Comparison,
        384,
        ["a"],
        lambda value: validate_comparison(value, 0, 1, {0: ["s1"], 1: ["s2"]}, {}),
        references={"a_evidence": ["s1"], "b_evidence": ["s2"]},
    )
    assert value == compared() and raw["status"] == "complete"
    assert runner.ledger.state["attempts"][0]["status"] == "complete"


def test_nvml_probe_failure_is_a_diagnostic_value(monkeypatch):
    monkeypatch.delenv("REZE_EXPECTED_GPU_UUID", raising=False)
    assert "probe_error" in backend.gpu_memory_probe()


def test_single_startup_exception_cannot_cover_generated_outputs(tmp_path):
    write_json(tmp_path / "binding.json", {"protocol": "revalue-v3-r01-v2"})
    write_json(tmp_path / "R0/ledger.json", {"attempts": [{"number": 0}]})
    raw = {"status": "runtime_error", "error": "Object of type <class 'function'> is not serializable"}
    write_json(tmp_path / "R0/attempts/00000.json", raw)
    assert monitor_startup_recovery(tmp_path)["revised_R0_launch_ceiling"] == 4
    binding = {
        "maximum_active_worker_seconds_per_stage": 7200,
        "maximum_worker_attempts_per_stage": 3,
        "maximum_worker_attempts_by_stage": {"R0": 4, "R1": 3},
        "historical_usage": {"R0": {"worker_launches": 3, "active_seconds": 2600}},
    }
    assert worker_remaining(binding, "R0", []) == (4600, 1)
    assert worker_remaining(binding, "R1", []) == (7200, 3)
    write_json(tmp_path / "R0/attempts/00000.json", {**raw, "output_tokens": 1})
    with pytest.raises(ValueError, match="semantic regeneration"):
        monitor_startup_recovery(tmp_path)


def test_content_repair_requires_the_full_recorded_regression(tmp_path):
    write_json(tmp_path / "binding.json", {"protocol": "revalue-v3-r01-v4"})
    attempts = [{"number": i, "key": f"{i}/observation", "status": "complete"} for i in range(20)]
    write_json(tmp_path / "R0/ledger.json", {"attempts": attempts})
    for attempt in attempts:
        write_json(
            tmp_path / "R0/attempts" / f"{attempt['number']:05d}.json", {"parsed": {"facts": [{"kind": "unknown"}]}}
        )
    assert observation_content_recovery(tmp_path)["revised_R0_launch_ceiling"] == 6
    write_json(tmp_path / "R0/attempts/00000.json", {"parsed": {"facts": [{"kind": "event"}]}})
    with pytest.raises(ValueError, match="all-unknown regression"):
        observation_content_recovery(tmp_path)


def test_engineering_only_dispatch_does_not_start_duplicate_r1(tmp_path, monkeypatch):
    import sys

    import dispatch
    from io_utils import read_json, utc_now

    write_json(tmp_path / "R0/gate.json", {"status": "passed"})
    binding = {"created_at": utc_now(), "queue_expiry_hours": 72, "stage_order": ["R0"]}
    monkeypatch.setattr(dispatch, "load_run", lambda run: (binding, [], "binding"))
    monkeypatch.setattr(sys, "argv", ["dispatch.py", "--run", str(tmp_path)])

    def forbidden(*args, **kwargs):
        raise AssertionError("Engineering-only validation must not launch or evaluate R1")

    monkeypatch.setattr(dispatch.subprocess, "run", forbidden)
    monkeypatch.setattr(dispatch.subprocess, "Popen", forbidden)
    dispatch.main()
    state = read_json(tmp_path / "complete.json")
    assert state["status"] == "engineering_complete" and state["R1_started"] is False
