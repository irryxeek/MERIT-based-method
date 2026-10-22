"""Regression coverage for real failure modes, recovery budgets and fallback."""

import copy
import json

import pytest
from highlight_evidence import granularity
from highlight_evidence.robust_client import MemoryClient
from highlight_evidence.robust_memory import MemoryContract
from test_highlight_granularity import case as case

from merit_qvh.core import write_json


@pytest.fixture
def contract_case(case):
    values, frames, chunk, payload, prior = case
    contract = MemoryContract(granularity.units_for(chunk, "F2"), frames, values, 30, prior)
    candidate = {
        "observations": {
            e["id"]: {
                "description": e["description"],
                "evidence_ids": e["evidence_ids"],
                "values": {
                    v["dimension"]: {k: x for k, x in v.items() if k != "dimension"} for v in e["values"]
                },
            }
            for e in payload("F2")["events"]
        },
        "links": [],
    }
    return contract, candidate


def response(candidate, finish="stop"):
    return {"choices": [{"finish_reason": finish, "message": {"content": json.dumps(candidate)}}]}


def client_at(tmp_path, monkeypatch, responses):
    client = MemoryClient(
        "http://local-unused/v1",
        {"model": "test", "temperature": 0, "format_retries": 1, "transport_retries": 1},
        tmp_path / "requests",
        "test-protocol",
    )
    sent = []
    queue = iter(responses)

    def post(payload):
        sent.append(copy.deepcopy(payload))
        return next(queue)

    monkeypatch.setattr(client, "_post", post)
    return client, sent


def call(client, contract):
    return client.complete_memory(
        "F2_memory", [{"type": "text", "text": "test"}], 8192, contract, system="test"
    )


def test_decoder_schema_compiles_and_metadata_is_controller_owned(contract_case):
    import xgrammar

    contract, candidate = contract_case
    xgrammar.Grammar.from_json_schema(json.dumps(contract.schema), any_whitespace=False)
    result = contract.finalize(candidate)
    assert [(e["start"], e["end"]) for e in result["events"]] == [
        (10, 12),
        (12, 14),
        (14, 16),
        (16, 18),
        (18, 20),
    ]
    assert "start" not in contract.schema["properties"]["observations"]["properties"]["e0"]["properties"]


def test_161_character_cue_repairs_only_bad_field_and_caches(tmp_path, monkeypatch, contract_case):
    contract, candidate = contract_case
    cue = candidate["observations"]["e0"]["values"]["event_cue"]
    cue.update(status="observed", level=2, cue="x" * 161, evidence_ids=["f5"])
    fixed = {**cue, "cue": "Person raises an arm."}
    client, sent = client_at(
        tmp_path, monkeypatch, [response(candidate), response({"/observations/e0/values/event_cue": fixed})]
    )
    result = call(client, contract)
    assert len(sent) == 2 and result["quality"] == "validated" and result["repair_attempted"]
    assert result["events"][0]["values"][0]["cue"] == fixed["cue"]
    assert result["events"][1]["description"] == candidate["observations"]["e1"]["description"]
    repair_schema = sent[1]["response_format"]["json_schema"]["schema"]
    assert repair_schema["required"] == ["/observations/e0/values/event_cue"]
    assert "received=161" in sent[1]["messages"][-1]["content"]
    assert call(client, contract) == result and len(sent) == 2


@pytest.mark.parametrize("violation", ["neighbor_frame", "missing_context", "self_reference", "no_context"])
def test_persistent_bad_cue_becomes_unknown_without_inventing_evidence(
    tmp_path, monkeypatch, contract_case, violation
):
    contract, candidate = contract_case
    cue = candidate["observations"]["e0"]["values"]["importance_cue"]
    cue.update(status="observed", level=2, evidence_ids=["f5"], reference_event_ids=["w0:e0"])
    if violation == "neighbor_frame":
        cue["evidence_ids"] = ["f6"]
    else:
        cue["reference_event_ids"] = {
            "missing_context": ["w9:e0"],
            "self_reference": ["e0"],
            "no_context": [],
        }[violation]
    client, sent = client_at(
        tmp_path,
        monkeypatch,
        [response(candidate), response({"/observations/e0/values/importance_cue": cue})],
    )
    result = call(client, contract)
    assert len(sent) == 2 and result["quality"] == "degraded" and len(result["downgraded_cues"]) == 1
    value = result["events"][0]["values"][3]
    assert value["status"] == "unknown" and value["level"] is None
    assert value["evidence_ids"] == value["reference_event_ids"] == []
    assert result["downgraded_cues"][0]["original"] == cue


def test_patch_cannot_modify_valid_fields(contract_case):
    contract, candidate = contract_case
    candidate["observations"]["e0"]["values"]["event_cue"]["cue"] = "x" * 161
    issues = contract.problems(candidate)
    patch = {"/observations/e1/description": "A different unrequested description"}
    with pytest.raises(ValueError, match="only the listed"):
        contract.apply_patch(candidate, patch, issues)


def test_invalid_description_is_not_converted_to_fabricated_observation(contract_case):
    contract, candidate = contract_case
    candidate["observations"]["e0"]["evidence_ids"] = ["f6"]
    with pytest.raises(ValueError, match="Unrecoverable observation"):
        contract.finalize(candidate, degrade=True)


def test_truncation_is_bounded_and_terminal_failure_never_reissued(tmp_path, monkeypatch, contract_case):
    contract, candidate = contract_case
    client, sent = client_at(tmp_path, monkeypatch, [response(candidate, "length")] * 2)
    with pytest.raises(RuntimeError, match="Incomplete generation"):
        call(client, contract)
    with pytest.raises(RuntimeError, match="Previously failed"):
        call(client, contract)
    assert len(sent) == 2


def test_restart_replays_first_response_without_repeating_it(tmp_path, monkeypatch, contract_case):
    contract, candidate = contract_case
    cue = candidate["observations"]["e0"]["values"]["event_cue"]
    cue["cue"] = "x" * 161
    client, sent = client_at(tmp_path, monkeypatch, [response(candidate)])
    with pytest.raises(StopIteration):  # simulate interruption at the patch request
        call(client, contract)
    fixed = {**cue, "cue": "brief cue"}
    monkeypatch.setattr(client, "_post", lambda p: response({"/observations/e0/values/event_cue": fixed}))
    result = call(client, contract)
    ledger = json.loads(next((tmp_path / "requests").glob("*/*.json")).read_text())
    assert result["quality"] == "validated"
    assert len(ledger["attempts"]) == 3
    assert ledger["attempts"][1]["status"] == "interrupted_unknown"
    assert [a["format_attempt"] for a in ledger["attempts"]] == [0, 1, 1]


def test_window_failure_keeps_both_neighbor_windows_and_d0_only_for_failed_bins(tmp_path, monkeypatch):
    row = {"qid": 1, "vid": "v_0_60", "query": "activity", "duration": 60}
    frames = [{"clip_id": i} for i in range(15)]
    baseline = {"status": "ok", "pred_saliency_scores": [1.0] * 30}
    windows = [
        {
            "window_id": w,
            "start": w * 10,
            "end": (w + 1) * 10,
            "bin_ids": list(range(w * 5, (w + 1) * 5)),
            "context": [],
        }
        for w in range(3)
    ]
    write_json(
        tmp_path / "inputs/1.json",
        {
            "row": row,
            "D0": baseline,
            "E1_original": baseline,
            "frames": frames,
            "selected_windows": [0, 1, 2],
            "selected_bin_ids": list(range(15)),
            "windows": windows,
            "original_observations": [],
        },
    )
    visited = []

    def extract(client, frames, values, duration, window, arm, context, *, robust):
        visited.append((arm, window["window_id"]))
        if window["window_id"] == 1:
            raise RuntimeError("Broken middle window")
        return {**window, "events": [], "links": [], "rejected_links": [], "downgraded_cues": []}

    def score(client, stage, row, ids, supplied_frames, memory):
        assert ids == list(range(15)) and supplied_frames == frames
        return [{"clip_id": i, "score": 3.0} for i in ids]

    monkeypatch.setattr(granularity, "load_frames", lambda f: f)
    monkeypatch.setattr(granularity, "extract_units", extract)
    monkeypatch.setattr(granularity, "score_evidence", score)
    result = granularity.run_query(
        tmp_path,
        {
            "memory_contract": "structured_fields_v2",
            "endpoint": "unused",
            "vlm": {},
            "values": [],
        },
        "protocol",
        row,
    )
    for arm in granularity.ARMS:
        out = result[arm]
        assert out["status"] == "partial" and not out["memory_coverage_complete"]
        assert out["pred_saliency_scores"] == [3.0] * 5 + [1.0] * 5 + [3.0] * 5 + [1.0] * 15
        assert out["fallback_bin_ids"] == list(range(5, 10))
        assert [w for a, w in visited if a == arm] == [0, 1, 2]
