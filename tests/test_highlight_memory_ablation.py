"""Reader ablations must remove both cue pools and keep inference controls."""

import copy
import json

import pytest
from highlight_evidence import memory_ablation as ablation

from merit_qvh.core import write_json


@pytest.fixture
def memory():
    def event(eid):
        return {
            "id": eid,
            "start": 0,
            "end": 10,
            "description": "Person is visible.",
            "evidence_ids": ["f0"],
            "values": [
                {
                    "dimension": "importance_cue",
                    "status": "observed",
                    "level": 1,
                    "cue": "visible outcome",
                    "evidence_ids": ["f0"],
                    "reference_event_ids": ["w0:e0"],
                }
            ],
        }

    return {
        "selected_events": [event("g1:e0")],
        "reference_events": [event("w0:e0")],
        "links": [{"source": "w0:e0", "target": "g1:e0", "relation": "before", "evidence_ids": ["f0"]}],
        "caution": "Model observations, not truth.",
    }


def test_description_condition_removes_structured_cues_from_both_pools_without_mutation(memory):
    original = copy.deepcopy(memory)
    view = ablation.memory_view(memory, "B_description")
    assert memory == original
    assert all("values" not in e for e in view["selected_events"] + view["reference_events"])
    assert view["links"] == []
    assert "importance_cue" not in json.dumps(view)
    assert [e["id"] for e in view["reference_events"]] == [e["id"] for e in memory["reference_events"]]


def test_cue_condition_keeps_the_entire_common_context_pool_and_removes_only_links(memory):
    view = ablation.memory_view(memory, "C_cues")
    assert view == {**memory, "links": []}
    assert ablation.memory_view(memory, "D_full") == memory
    empty = ablation.memory_view(memory, "A_frames")
    assert empty == {"selected_events": [], "reference_events": [], "links": [], "caution": memory["caution"]}
    with pytest.raises(ValueError):
        ablation.memory_view(memory, "typo")


def test_actual_scorer_changes_only_memory_between_four_conditions(monkeypatch, memory):
    data = {
        "row": {"query": "person", "vid": "video", "duration": 20},
        "selected_bin_ids": [0, 1],
        "frames": [
            {"image_id": "f0", "clip_id": 0, "pts": 1, "url": "pixels0"},
            {"image_id": "f1", "clip_id": 1, "pts": 3, "url": "pixels1"},
        ],
    }
    monkeypatch.setattr(ablation, "load_frames", lambda f: f)
    controls = []
    for arm in ablation.ARMS:
        payload = ablation.expected_payload(
            {"vlm": {"model": "test", "temperature": 0}},
            data,
            ablation.memory_view(memory, arm),
            [{"clip_id": i, "score": 0} for i in (0, 1)],
        )
        assert payload["max_tokens"] == 1024 and payload["seed"] == 42
        content = payload["messages"][1]["content"]
        instruction, raw = content[0]["text"].split("\n", 1)
        body = json.loads(raw)
        assert body.pop("memory") == ablation.memory_view(memory, arm)
        content[0]["text"] = instruction + "\n" + json.dumps(body)
        controls.append(payload)
    assert controls[0] == controls[1] == controls[2] == controls[3]


def setup_query(tmp_path, memory, *, active=True):
    row = {"qid": 1, "vid": "video", "query": "person", "duration": 12}
    baseline = {"status": "ok", "pred_saliency_scores": [1.0] * 6}
    write_json(
        tmp_path / "inputs/1.json",
        {
            "row": row,
            "D0": baseline,
            "selected_bin_ids": [1, 3] if active else [],
            "selected_windows": [0] if active else [],
            "frames": [{"clip_id": 1}, {"clip_id": 3}] if active else [],
        },
    )
    write_json(
        tmp_path / "references/1.json",
        {
            "C10": {
                "status": "partial" if active else "inherited_no_candidates",
                "pred_saliency_scores": [1, 2, 1, 2, 1, 1] if active else baseline["pred_saliency_scores"],
                "scoring_memory": memory,
            }
        },
    )
    return row, {"source_protocol_id": "old", "vlm": {}, "endpoint": "unused"}


def test_failed_condition_retains_d0_and_other_conditions_continue_without_reissuing_cached_d(
    tmp_path, monkeypatch, memory
):
    row, config = setup_query(tmp_path, memory)
    calls = []
    monkeypatch.setattr(ablation, "load_frames", lambda f: f)

    def score(client, stage, row, ids, frames, view):
        calls.append(stage)
        if stage.startswith("B_description"):
            raise RuntimeError("Failed scorer")
        return [{"clip_id": i, "score": 3.0} for i in ids]

    monkeypatch.setattr(ablation, "score_evidence", score)
    result = ablation.run_query(tmp_path, config, "new", row)
    assert sorted(calls) == sorted(a + "_E1" for a in ablation.NEW_ARMS)
    assert result["D_full"]["pred_saliency_scores"] == [1, 2, 1, 2, 1, 1]
    assert result["D_full"]["reused"] and result["D_full"]["new_requests"] == []
    assert result["B_description"]["status"] == "fallback"
    assert result["B_description"]["pred_saliency_scores"] == [1] * 6
    for arm in ("A_frames", "C_cues"):
        assert result[arm]["status"] == "ok"
        assert result[arm]["pred_saliency_scores"] == [1, 3, 1, 3, 1, 1]
    assert ablation.run_query(tmp_path, config, "new", row) == result
    assert len(calls) == 3


def test_no_candidate_queries_never_load_pixels_or_issue_inference(tmp_path, monkeypatch, memory):
    row, config = setup_query(tmp_path, memory, active=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("No-candidate query must not infer")

    monkeypatch.setattr(ablation, "load_frames", forbidden)
    monkeypatch.setattr(ablation, "Client", forbidden)
    result = ablation.run_query(tmp_path, config, "new", row)
    for arm in ablation.ARMS:
        assert result[arm]["pred_saliency_scores"] == [1] * 6
        assert result[arm]["status"] == "inherited_no_candidates"
