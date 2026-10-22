"""The full-memory baseline must not inherit candidate or D0 dependencies."""

import copy
import json

import pytest
from highlight_evidence import memory_only as module

from merit_qvh.core import write_json


@pytest.fixture
def events():
    return [
        {
            "id": f"g0:e{i}",
            "start": 2 * i,
            "end": 2 * i + 2,
            "description": "A person stands near a door.",
            "evidence_ids": [f"f{i}"],
            "values": [
                {
                    "dimension": d,
                    "status": "unknown",
                    "level": None,
                    "cue": "Not visible.",
                    "evidence_ids": [],
                    "reference_event_ids": [],
                }
                for d in ("event_cue", "emotion_cue", "novelty_cue", "importance_cue")
            ],
        }
        for i in range(2)
    ]


def test_full_text_inputs_hold_timeline_and_prompt_fixed_and_cannot_receive_d0(events):
    original = copy.deepcopy(events)
    row = {"query": "A person opens a door.", "duration": 4, "D0": "secret baseline", "selected_bin_ids": [1]}
    controls = []
    for arm in module.ARMS:
        content, contract = module.score_request(row, events, arm)
        assert len(content) == 1 and content[0]["type"] == "text"
        prompt, raw = content[0]["text"].split("\n", 1)
        body = json.loads(raw)
        assert body["original_query"] == row["query"]
        assert [e["clip_id"] for e in body["memory"]] == [0, 1]
        assert "secret baseline" not in raw and "selected_bin_ids" not in raw
        for e in body["memory"]:
            assert ("values" in e) == (arm == "Memory_cue")
            e.pop("values", None)
        controls.append((prompt, body, contract.schema))
    assert controls[0] == controls[1] and events == original
    with pytest.raises(ValueError, match="complete ordered"):
        module.score_request(row, events[:1], "Memory_text")


@pytest.mark.parametrize("scores", [[1], [1, 5], [float("nan"), 1], [True, 1], [0, float("inf")]])
def test_score_contract_rejects_missing_or_invalid_values_even_on_degrade(scores):
    contract = module.ScoreContract(2)
    assert contract.problems({"scores": scores})
    with pytest.raises(ValueError, match="Unusable"):
        contract.finalize({"scores": scores}, degrade=True)
    assert contract.finalize({"scores": [0, 3.5]})["scores"] == [0.0, 3.5]


def test_memory_prompt_has_no_query_or_previous_cue_fields_and_forbids_graph(events):
    frames = [
        {"image_id": "f0", "pts": 1.0, "url": "test0", "jpeg_sha256": "pixels0"},
        {"image_id": "f1", "pts": 3.0, "url": "test1", "jpeg_sha256": "pixels1"},
    ]
    values = [
        {"id": v["dimension"], "definition": "visible evidence", "visible_cues": "visible action"}
        for v in events[0]["values"]
    ]
    prior = copy.deepcopy(events[0])
    prior.update(id="previous", start=0, end=2)
    prior["values"][0]["cue"] = "never copy this old cue"
    content, contract, units = module.memory_request(
        frames, values, 4, {"window_id": 0, "start": 0, "end": 4, "bin_ids": [0, 1]}, [prior]
    )
    body = json.loads(content[0]["text"].split("\n", 1)[1])
    assert "query" not in body and "original_query" not in body
    assert "values" not in body["prior_observations"][0]
    assert "never copy this old cue" not in content[0]["text"]
    assert [u["evidence_ids"] for u in units] == [["f0"], ["f1"]]
    assert contract.schema["properties"]["links"]["maxItems"] == 0


def setup_run(tmp_path):
    row = {"qid": 1, "vid": "video", "query": "person", "duration": 12}
    write_json(
        tmp_path / "inputs/1.json",
        {
            "row": row,
            "frames": [{"clip_id": i} for i in range(6)],
            "windows": [
                {"window_id": 0, "start": 0, "end": 10, "bin_ids": [0, 1, 2, 3, 4]},
                {"window_id": 1, "start": 10, "end": 12, "bin_ids": [5]},
            ],
        },
    )
    # A failed D0 reference must not prevent memory-only inference.
    write_json(tmp_path / "references/1.json", {"D0": {"status": "failed", "pred_saliency_scores": [4] * 6}})
    c = {"endpoint": "unused", "vlm": {}, "values": [], "server": {"max_model_len": 32768}}
    return row, c


class FakeClient:
    def __init__(self, *args):
        self.events = []


def fake_extract(client, frames, values, duration, window, context):
    return {**window, "quality": "validated", "events": [{"id": str(i)} for i in window["bin_ids"]]}


def test_failed_memory_window_continues_build_but_never_falls_back_to_d0(tmp_path, monkeypatch):
    row, c = setup_run(tmp_path)
    calls = []
    monkeypatch.setattr(module, "MemoryClient", FakeClient)
    monkeypatch.setattr(module, "load_frames", lambda x: x)

    def extract(*args):
        w = args[4]
        calls.append(w["window_id"])
        if w["window_id"] == 0:
            raise RuntimeError("missing observation")
        assert args[5] == []
        return fake_extract(*args)

    monkeypatch.setattr(module, "extract_memory", extract)
    monkeypatch.setattr(
        module, "score_memory", lambda *args: pytest.fail("Incomplete memory must not be scored")
    )
    result = module.run_query(tmp_path, c, "test", row, None)
    assert calls == [0, 1] and result["memory"]["failed_windows"] == [0]
    assert all(result[a]["pred_saliency_scores"] == [2.0] * 6 for a in module.ARMS)
    assert all(result[a]["status"] == "failed" for a in module.ARMS)
    assert module.run_query(tmp_path, c, "test", row, None) == result
    assert calls == [0, 1]


def test_failed_score_arm_does_not_block_other_arm_or_reuse_failed_d0(tmp_path, monkeypatch):
    row, c = setup_run(tmp_path)
    calls = []
    monkeypatch.setattr(module, "MemoryClient", FakeClient)
    monkeypatch.setattr(module, "load_frames", lambda x: x)
    monkeypatch.setattr(module, "extract_memory", fake_extract)

    def score(client, row, events, arm, *args):
        calls.append(arm)
        assert len(events) == 6
        if arm == "Memory_text":
            raise RuntimeError("scorer failed")
        return {
            "scores": [3.0] * 6,
            "request_id": "test",
            "input_tokens_preflight": 100,
            "input_memory_sha256": "digest",
        }

    monkeypatch.setattr(module, "score_memory", score)
    result = module.run_query(tmp_path, c, "test", row, None)
    assert set(calls) == set(module.ARMS)
    assert result["Memory_text"]["pred_saliency_scores"] == [2.0] * 6
    assert result["Memory_cue"]["pred_saliency_scores"] == [3.0] * 6
    assert result["Memory_cue"]["status"] == "ok"
    assert "D0" not in result


def test_context_overflow_fails_without_silently_truncating_memory(events):
    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return [0] * 33000

    class Client:
        def complete_memory(self, *args, **kwargs):
            pytest.fail("Must not issue overflowing request")

    with pytest.raises(ValueError, match="no truncation"):
        module.score_memory(
            Client(), {"query": "person", "duration": 4}, events, "Memory_cue", Tokenizer(), 32768
        )


def test_unknown_interrupted_attempt_cost_is_not_counted_as_free(tmp_path):
    write_json(
        tmp_path / "requests/xx/test.json",
        {
            "stage": "Memory_text_score",
            "status": "failed",
            "attempts": [
                {"status": "interrupted_unknown", "seconds": 4, "payload": {"messages": []}},
                {
                    "status": "responded",
                    "seconds": 3,
                    "payload": {"messages": []},
                    "response": {"usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}},
                },
            ],
        },
    )
    cost = module.costs(tmp_path)["Memory_text_score"]
    assert cost["actual_attempts"] == 2 and cost["unknown_interrupted_attempts"] == 1
    assert cost["seconds_including_queue"] == 7 and cost["total_tokens"] == 12
