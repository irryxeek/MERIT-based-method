"""Behavioral regressions for context leakage, source identity and static inputs."""

import hashlib
from copy import deepcopy

import numpy as np
import pytest
from comparison import allowed_references
from evidence_grounding import ground_observation, memory_source, temporal_scope, validate_source
from frame_observer import FRAME_PROMPT, grounded_description, still_payload, validate_description
from memory import build_memory, retrieve
from r2_planner import update_memory
from test_contracts import CharacterTokenizer


def source_trace():
    return {"frames": [
        {"id": "s1", "frame_index": 1, "pts": [47.9, 48.0]},
        {"id": "s2", "frame_index": 2, "pts": [48.0, 48.1]},
        {"id": "s3", "frame_index": 3, "pts": [49.99, 50.01]},
    ]}


def facts(refs):
    return {"facts": [{"kind": "context", "text": "A person is visible.", "frames": refs}], "unknown": []}


@pytest.mark.parametrize("refs,scope,candidate", [
    (["s1"], "outside", None), (["s2"], "inside", 24),
    (["s3"], "boundary", None), (["s1", "s2"], "mixed", None),
])
def test_source_intervals_determine_candidate_membership(refs, scope, candidate):
    bound = ground_observation(facts(refs), source_trace(), 24, [48.0, 50.0], "obs")
    fact = bound["obs_f0"]
    assert fact["grounding"]["scope"] == scope
    assert fact["candidate"] == candidate
    assert set(fact["grounding"]["frame_pts"]) == set(refs)
    assert fact["grounding"]["event_assignment"] == "unverified"
    view = {"items": [{"sources": {"obs_f0": memory_source(fact)}}]}
    allowed = allowed_references(24, {24: []}, view)
    assert ("obs_f0" in allowed) == (scope == "inside")


def test_unknown_has_no_temporal_or_event_claim():
    bound = ground_observation({"facts": [], "unknown": ["Identity unclear"]}, source_trace(), 24, [48, 50], "u")
    source = memory_source(bound["u_f0"])
    assert source["candidate"] is None and source["frames"] == []
    assert source["grounding"]["scope"] == "unknown"


@pytest.mark.parametrize("refs", [["s1", "s1"], ["s999"]])
def test_duplicate_or_foreign_frames_rejected(refs):
    with pytest.raises(ValueError, match="unobserved"):
        ground_observation(facts(refs), source_trace(), 24, [48, 50], "obs")


def test_tampered_scope_or_candidate_cannot_restore_context_citation():
    fact = ground_observation(facts(["s1"]), source_trace(), 24, [48, 50], "obs")["obs_f0"]
    bad = memory_source(fact)
    bad["candidate"] = 24
    with pytest.raises(ValueError, match="Context"):
        validate_source(bad)
    bad = memory_source(fact)
    bad["grounding"]["scope"] = "inside"
    with pytest.raises(ValueError, match="contradicts"):
        validate_source(bad)
    with pytest.raises(ValueError, match="Invalid"):
        temporal_scope({"s1": [float("nan"), 2]}, [0, 2])


def test_bounded_memory_preserves_exact_pts_without_restoring_raw_facts():
    bound = ground_observation(facts(["s1"]), source_trace(), 24, [48, 50], "obs")
    memory = build_memory({"items": [{"text": "A person.", "fact_ids": ["obs_f0"]}]}, bound, CharacterTokenizer())
    view = retrieve(memory, CharacterTokenizer())
    assert view["items"][0]["sources"]["obs_f0"]["grounding"]["frame_pts"] == {"s1": [47.9, 48.0]}
    assert allowed_references(24, {24: ["old"]}, view) == ["old"]
    assert "frame_receipts" not in view["items"][0]["sources"]["obs_f0"]


def test_update_requires_explicit_scope_activation_and_preserves_legacy_behavior():
    tokenizer = CharacterTokenizer()
    empty = build_memory({"items": []}, {}, tokenizer)
    observation = {"action": {"candidate": 24}, "trace": source_trace(), "result": facts(["s1"])}
    legacy = update_memory(empty, [observation], tokenizer)
    scoped = update_memory(empty, [observation], tokenizer, target_intervals={24: [48, 50]})
    assert "obs0_c24_f0" in allowed_references(24, {24: []}, retrieve(legacy, tokenizer))
    assert allowed_references(24, {24: []}, retrieve(scoped, tokenizer)) == []
    again = update_memory(scoped, [], tokenizer, target_intervals={24: [48, 50]})
    assert retrieve(again, tokenizer)["items"] == retrieve(scoped, tokenizer)["items"]


def test_one_frame_input_has_no_neighbor_query_or_invented_timestamp():
    pixels = np.arange(3 * 2 * 2 * 3, dtype=np.uint8).reshape(3, 2, 2, 3)
    trace = source_trace()
    trace["action"] = {"question": "LEAK_SENTINEL shower", "candidate": 24}
    for index, frame in enumerate(trace["frames"]):
        frame["rgb_sha256"] = hashlib.sha256(pixels[index].tobytes()).hexdigest()
    video = {"video": pixels, "metadata": {"frames_indices": [47900000, 48000000, 49990000], "fps": 1e6}}
    still, proof = still_payload(video, trace, 1)
    assert np.array_equal(still["video"][0], pixels[1])
    assert np.array_equal(still["video"][0], still["video"][1])
    assert still["metadata"]["frames_indices"] == [48000000, 48000000]
    assert proof["model_frame_ids"] == ["s2", "s2"]
    assert "action" not in proof and "shower" not in FRAME_PROMPT and "LEAK_SENTINEL" not in FRAME_PROMPT
    bound = grounded_description({"visible": "A person in a room.", "uncertainty": ""}, proof, 24, [48, 50], "one")
    assert bound["one_f0"]["frames"] == ["s2"]
    assert bound["one_f0"]["grounding"]["frame_pts"] == {"s2": [48.0, 48.1]}
    bad = deepcopy(trace)
    bad["frames"][1]["rgb_sha256"] = "bad"
    with pytest.raises(ValueError, match="pixels"):
        still_payload(video, bad, 1)
    with pytest.raises(ValueError, match="blank"):
        validate_description({"visible": " ", "uncertainty": ""})


@pytest.mark.parametrize("placeholder", [":", "..", " "])
def test_punctuation_only_visible_field_never_becomes_citable_fact(placeholder):
    proof = {"frames": [source_trace()["frames"][1]]}
    bound = grounded_description({"visible": placeholder, "uncertainty": "The scene is unclear."}, proof, 24, [48, 50], "empty")
    assert bound and all(f["kind"] == "unknown" for f in bound.values())
    view = {"items": [{"sources": {key: memory_source(fact) for key, fact in bound.items()}}]}
    assert allowed_references(24, {24: []}, view) == []
