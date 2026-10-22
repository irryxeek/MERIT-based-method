"""Pilot behavior: contextual provenance, stable sampling and bounded evidence."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
from highlight_evidence.controller import resolve_evidence
from highlight_evidence.inference import compare_candidates
from highlight_evidence.pilot import select_rows
from highlight_evidence.pilot_components import (
    choose_windows,
    context_for,
    extract_context_memory,
    score_evidence,
    select_needs,
)
from highlight_evidence.planner import Need


class ScriptedClient:
    def __init__(self, value):
        self.value = value

    def complete(self, stage, content, max_tokens, validator, *, system):
        return validator(copy.deepcopy(self.value))


@pytest.fixture
def context_case():
    values = json.loads(
        (Path(__file__).resolve().parents[1] / "configs/highlight_value_schema.example.json").read_text()
    )["values"]
    frames = [{"image_id": "f0", "pts": 11.0, "jpeg_sha256": "fixture", "url": "fixture"}]
    event = {
        "id": "e0",
        "start": 10,
        "end": 20,
        "description": "Player raises both arms",
        "evidence_ids": ["f0"],
        "values": [
            {
                "dimension": v["id"],
                "status": "unknown",
                "level": None,
                "cue": "not observed",
                "evidence_ids": [],
                "reference_event_ids": [],
            }
            for v in values
        ],
    }
    prior = copy.deepcopy(event)
    prior.update(id="w0:e0", start=0, end=10)
    return values, frames, {"events": [event], "links": []}, [prior]


@pytest.mark.parametrize("ref", [[], ["invented"], ["e0"]])
def test_contextual_claim_requires_an_observed_other_event(context_case, ref):
    values, frames, payload, prior = context_case
    payload["events"][0]["values"][2].update(
        status="observed", level=2, evidence_ids=["f0"], reference_event_ids=ref
    )
    with pytest.raises(ValueError):
        extract_context_memory(ScriptedClient(payload), frames, values, 30, 10, 20, "w1", prior)


def test_valid_reference_preserves_prior_namespace_and_unknown_null(context_case):
    values, frames, payload, prior = context_case
    payload["events"][0]["values"][2].update(
        status="observed", level=1, evidence_ids=["f0"], reference_event_ids=["w0:e0"]
    )
    result = extract_context_memory(ScriptedClient(payload), frames, values, 30, 10, 20, "w1", prior)
    assert result["events"][0]["id"] == "w1:e0"
    assert result["events"][0]["values"][2]["reference_event_ids"] == ["w0:e0"]
    assert result["events"][0]["values"][3]["level"] is None
    chunks = [{"window_id": 0, "events": prior, "links": []}, {"window_id": 1, **result}]
    assert context_for(chunks, [1])["reference_events"] == prior


def test_cross_window_link_preserves_citations_and_cannot_reverse_time(context_case):
    values, frames, payload, prior = context_case
    prior[0]["evidence_ids"] = ["prior_frame"]
    payload["links"] = [
        {"source": "w0:e0", "target": "e0", "relation": "before", "evidence_ids": ["prior_frame", "f0"]}
    ]
    result = extract_context_memory(ScriptedClient(payload), frames, values, 30, 10, 20, "w1", prior)
    assert result["links"][0]["source"] == "w0:e0"
    assert result["links"][0]["target"] == "w1:e0"
    assert (
        context_for([{"window_id": 0, "events": prior, "links": []}, {"window_id": 1, **result}], [1])[
            "reference_events"
        ]
        == prior
    )
    payload["links"][0].update(source="e0", target="w0:e0")
    rejected = extract_context_memory(ScriptedClient(payload), frames, values, 30, 10, 20, "w1", prior)
    assert rejected["links"] == [] and "backwards" in rejected["rejected_links"][0]["reason"]
    assert rejected["events"] == result["events"]
    payload["links"][0].update(source="w0:e0", target="e0", evidence_ids=["f0"])
    rejected = extract_context_memory(ScriptedClient(payload), frames, values, 30, 10, 20, "w1", prior)
    assert rejected["links"] == [] and "EACH endpoint" in rejected["rejected_links"][0]["reason"]


def test_gap_selection_uses_fixed_views_and_does_not_trust_model_timestamps(context_case):
    values, frames, payload, prior = context_case
    memory = extract_context_memory(ScriptedClient(payload), frames, values, 30, 10, 20, "w1", prior)
    chunks = [{"window_id": 1, **memory}]
    choice = {
        "view_id": "v1",
        "criterion": "reaction",
        "question": "Does the person raise both arms?",
        "reason": "Cannot see the full motion",
    }
    result = select_needs(ScriptedClient({"needs": [choice]}), "reactions", {}, chunks, [1], 30)
    assert result[0]["start"] == 14 and result[0]["end"] == 20
    assert result[0]["event_id"] == "w1:e0" and result[0]["question"] == choice["question"]
    choice["start"] = 0
    with pytest.raises(ValueError, match="no timestamps"):
        select_needs(ScriptedClient({"needs": [choice]}), "reactions", {}, chunks, [1], 30)


def test_window_retrieval_keeps_semantic_cue_and_baseline_candidates():
    chunks = []
    for i in range(5):
        e = {
            "description": f"event{i}",
            "values": [{"status": "observed", "dimension": "emotion_cue", "cue": "movement"}],
        }
        chunks.append({"window_id": i, "events": [e], "bin_ids": [i]})

    class Embeddings:
        def encode(self, texts):
            if len(texts) == 1:
                return np.array([[1.0]])
            # Semantic top0, cue top1, baseline top4; overlapping ranks cannot crowd out all three.
            return np.array([[10], [0], [1], [11], [0], [0], [0], [0], [0], [0]])

    selected, _ = choose_windows(
        chunks, [0, 0, 0, 0, 4], "reaction", {"facets": ["emotion_cue"]}, Embeddings()
    )
    assert selected == [0, 1, 4]


@pytest.mark.parametrize(
    "scores",
    [[{"clip_id": 1, "score": 3}], [{"clip_id": 0, "score": float("nan")}], [{"clip_id": 0, "score": 5}]],
)
def test_local_scores_cannot_change_bin_alignment_or_scale(scores):
    client = ScriptedClient({"clip_scores": scores})
    row = {"query": "action", "vid": "video", "duration": 2}
    with pytest.raises(ValueError):
        score_evidence(client, "test", row, [0], [], {})


def test_controller_uses_new_pts_and_preserves_canonical_frame_ids(monkeypatch):
    initial = [{"image_id": "f0", "pts": 0.5, "jpeg_sha256": "fixture", "url": "fixture"}]

    def decode(video, info, view):
        assert 0.5 not in view["pts"]
        return [
            {"image_id": f"canonical-{i}", "pts": p, "jpeg_sha256": "fixture", "url": "fixture"}
            for i, p in enumerate(view["pts"])
        ]

    class Verifier:
        def complete(self, stage, content, max_tokens, validator, *, system):
            identifiers = [json.loads(p["text"])["image_id"] for p in content[1:] if p["type"] == "text"]
            assert identifiers == [f"r1f{i}" for i in range(24)]
            return validator(
                {"event_id": "e", "criterion": "motion", "answer": "no", "evidence_ids": identifiers}
            )

    monkeypatch.setattr("highlight_evidence.controller.decode_view", decode)
    result = resolve_evidence(
        Verifier(),
        "action",
        [Need("e", "motion", "Is movement visible?", 0, 2)],
        [],
        initial,
        "fixture",
        {"pts": [i / 30 for i in range(60)]},
        2,
    )
    assert result["used_views"] == 1 and result["new_frames_decoded"] == 24 and result["gaps"] == []
    assert result["frames"][1]["source_image_id"] == "canonical-0"


def test_pilot_sampling_is_input_order_independent():
    rows = [{"qid": i, "vid": "v" + str(i)} for i in range(100)]
    assert len(select_rows(rows)) == 20
    assert select_rows(rows) == select_rows(list(reversed(rows)))


def test_comparison_retry_explains_the_actual_length_constraint():
    frames = [
        {"image_id": "a", "pts": 1, "jpeg_sha256": "fixture", "url": "fixture"},
        {"image_id": "b", "pts": 3, "jpeg_sha256": "fixture", "url": "fixture"},
    ]
    client = ScriptedClient({"winner": "tie", "basis": "x" * 241, "evidence_ids": ["a", "b"]})
    with pytest.raises(ValueError, match="1-240 characters"):
        compare_candidates(
            client,
            "action",
            "event",
            {"id": "left", "evidence_ids": ["a"]},
            {"id": "right", "evidence_ids": ["b"]},
            frames,
            4,
        )
