"""Behavior checks for evidence provenance, bounded source views and selection."""

import copy
import json

import pytest
from highlight_evidence.controller import resolve_evidence
from highlight_evidence.inference import compare_candidates, verify_view
from highlight_evidence.memory import extract_memory, retrieval_keys, validate_memory
from highlight_evidence.planner import (
    Budget,
    Need,
    comparison_pairs,
    merge_dense_scores,
    plan_views,
    select_events,
    unique_pts,
    unresolved,
)


@pytest.fixture
def sample():
    frames = [
        {"image_id": f"f{i}", "pts": i / 4, "jpeg_sha256": "test_fixture", "url": "fixture"}
        for i in range(16)
    ]
    values = [
        {
            "id": "reaction",
            "definition": "Visible reaction intensity",
            "visible_cues": "arm motion, duration, participants",
        }
    ]
    memory = {
        "events": [
            {
                "id": "a",
                "start": 0,
                "end": 2,
                "description": "A player raises both arms",
                "evidence_ids": ["f0", "f2", "f4", "f6"],
                "values": [
                    {
                        "dimension": "reaction",
                        "status": "observed",
                        "level": 2,
                        "cue": "both arms move upward",
                        "evidence_ids": ["f0", "f4"],
                    }
                ],
            },
            {
                "id": "b",
                "start": 2,
                "end": 4,
                "description": "Several people raise their arms",
                "evidence_ids": ["f8", "f10", "f12", "f14"],
                "values": [
                    {
                        "dimension": "reaction",
                        "status": "unknown",
                        "level": None,
                        "cue": "duration unseen",
                        "evidence_ids": [],
                    }
                ],
            },
        ],
        "links": [{"source": "a", "target": "b", "relation": "before", "evidence_ids": ["f0", "f8"]}],
    }
    return frames, values, memory


class FakeClient:
    """Scripted contract test only; never an accuracy result."""

    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    def complete(self, stage, content, max_tokens, validator, *, system):
        self.calls.append({"stage": stage, "content": content, "max_tokens": max_tokens, "system": system})
        return validator(copy.deepcopy(self.answer))


def test_memory_keys_and_namespaces(sample):
    frames, values, memory = sample
    client = FakeClient(memory)
    result = extract_memory(client, frames, values, 4, namespace="chunk0")
    assert result["events"][0]["id"] == "chunk0:a"
    assert result["links"][0]["target"] == "chunk0:b"
    assert len(retrieval_keys(result)) == 3  # Do not embed an unknown observation as a fact.
    context = json.loads(client.calls[0]["content"][0]["text"].split("\n")[-1])
    assert set(context) == {"duration", "value_definitions", "schema_example"}
    assert memory["events"][0]["id"] == "a"


@pytest.mark.parametrize(
    "change",
    ["uncited", "unknown_zero", "duplicate_facet", "causal", "backwards", "one_sided", "out_of_bounds"],
)
def test_reject_invalid_memory_evidence(sample, change):
    frames, values, m = sample
    if change == "uncited":
        m["events"][0]["values"][0]["evidence_ids"] = ["invented_frame"]
    elif change == "unknown_zero":
        m["events"][1]["values"][0]["level"] = 0
    elif change == "duplicate_facet":
        m["events"][0]["values"] *= 2
    elif change == "causal":
        m["links"][0]["relation"] = "causes"
    elif change == "backwards":
        m["links"][0].update(source="b", target="a")
    elif change == "one_sided":
        m["links"][0]["evidence_ids"] = ["f0"]
    else:
        m["events"][0]["evidence_ids"].append("f8")
    with pytest.raises(ValueError):
        validate_memory(m, frames, values, 4)


def test_unknown_is_not_absence_and_motion_needs_coverage(sample):
    frames, _, _ = sample
    need = Need("a", "reaction", "Does arm movement persist over the interval?", 0, 2)
    answer = {"event_id": "a", "criterion": "reaction", "answer": "no", "evidence_ids": ["f0", "f1"]}
    assert unresolved([need], [answer], frames, 4)[0]["reason"] == "temporal_coverage_gap"
    answer["evidence_ids"] = ["f0", "f2", "f4", "f6"]
    assert unresolved([need], [answer], frames, 4) == []
    answer["answer"] = "unknown"
    assert unresolved([need], [answer], frames, 4)[0]["reason"] == "explicitly_unknown"


def test_budget_and_real_frame_deduplication():
    pts = [i / 30 for i in range(300)]
    needs = [Need(f"e{i}", "outcome", "Is ball-to-net contact visible?", i, i + 2) for i in range(3)]
    gaps = unresolved(needs, [], [], 10)
    seen = [0.5, 1.5]
    plan = plan_views("Show successful shots", gaps, pts, 10, Budget(max_new_frames=30), seen=seen)
    chosen = [p for view in plan["views"] for p in view["pts"]]
    assert len(plan["views"]) == 2 and len(chosen) == 30
    assert len(chosen) == len(set(chosen)) and not set(chosen) & set(seen)
    assert set(chosen) <= set(pts)
    assert plan["deferred"][0]["deferred_reason"] == "budget_exhausted"
    assert all(view["original_query"] == "Show successful shots" for view in plan["views"])
    followup = plan_views("Show successful shots", gaps, pts, 10, used_views=2, used_frames=30)
    assert not followup["views"]  # Counters are cumulative across rounds.


def test_exhausted_or_sparse_video_does_not_pad():
    assert unique_pts([0.0, 0.04], 0, 2, 24) == [0.0, 0.04]
    assert unique_pts([0.0, 0.04], 0, 2, 24, seen=[0.0, 0.04]) == []
    with pytest.raises(ValueError):
        unique_pts([0.0, 0.0], 0, 2, 24)


def test_verification_and_comparison_cite_correct_evidence(sample):
    frames, _, _ = sample
    view = {
        "original_query": "Find strong reactions",
        "event_id": "a",
        "criterion": "reaction",
        "question": "Are both arms raised?",
        "focus": "subject",
        "start": 0,
        "end": 2,
        "pts": [f["pts"] for f in frames[:8]],
    }
    client = FakeClient(
        {"event_id": "a", "criterion": "reaction", "answer": "yes", "evidence_ids": ["f2", "f4"]}
    )
    assert verify_view(client, view, frames[:8], 4)["answer"] == "yes"
    assert "Find strong reactions" in client.calls[0]["content"][0]["text"]
    left, right = {"id": "a", "evidence_ids": ["f0", "f4"]}, {"id": "b", "evidence_ids": ["f8", "f12"]}
    client = FakeClient({"winner": "right", "evidence_ids": ["f4"], "basis": "more participants"})
    with pytest.raises(ValueError):
        compare_candidates(client, "Find strong reactions", "visible group reaction", left, right, frames, 4)
    client.answer["evidence_ids"].append("f12")
    assert (
        compare_candidates(client, "Find strong reactions", "visible group reaction", left, right, frames, 4)[
            "winner"
        ]
        == "right"
    )


def test_all_and_topk_do_not_modify_dense_output():
    events = [
        {"id": "a", "start": 0, "score": 3, "relevant": True},
        {"id": "b", "start": 10, "score": 3, "relevant": True},
        {"id": "c", "start": 20, "score": 4, "relevant": False},
    ]
    assert select_events(events, mode="all", threshold=2)["selected_ids"] == ["a", "b"]
    top = select_events(events, mode="topk", threshold=2, k=1)
    assert top["selected_ids"] == ["a"] and top["boundary_tie"]
    assert comparison_pairs(events) == [("a", "b")]
    initial = [1, 3, 1, 3, 0]
    assert merge_dense_scores(initial, [{"clip_id": 1, "score": 3.5}]) == [1, 3.5, 1, 3, 0]
    assert initial == [1, 3, 1, 3, 0]
    with pytest.raises(ValueError):
        merge_dense_scores(initial, [{"clip_id": 1, "score": 2}, {"clip_id": 1, "score": 3}])


def test_controller_stops_at_budget_on_decode_failures(monkeypatch):
    def fail(*args):
        raise ValueError("unreadable source frame")

    monkeypatch.setattr("highlight_evidence.controller.decode_view", fail)
    needs = [Need("a", "outcome", "Is the shot outcome visible?", 0, 2)]
    result = resolve_evidence(
        None, "Successful shot", needs, [], [], "unused.mp4", {"pts": [i / 30 for i in range(60)]}, 2
    )
    assert result["status"] == "unresolved"
    assert result["used_views"] == 2 and result["reserved_new_frames"] == 48
    assert len(result["steps"]) == 2 and all(r["status"] == "failed" for r in result["steps"])


def test_controller_resolved_negative_needs_no_model_call(sample):
    frames, _, _ = sample
    needs = [Need("a", "reaction", "Does the subject raise both arms?", 0, 2)]
    observations = [
        {"event_id": "a", "criterion": "reaction", "answer": "no", "evidence_ids": ["f0", "f2", "f4", "f6"]}
    ]
    result = resolve_evidence(
        None, "Strong reactions", needs, observations, frames, "unused.mp4", {"pts": []}, 4
    )
    assert result["status"] == "criteria_structurally_resolved"
    assert result["used_views"] == 0 and not result["steps"]
