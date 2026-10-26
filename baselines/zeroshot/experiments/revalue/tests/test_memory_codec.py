"""CPU source-preservation and corruption regressions for compact memory."""

import json
from copy import deepcopy

import pytest
from comparison import allowed_references
from evidence_grounding import ground_observation, memory_source
from io_utils import digest
from memory_codec import LEGEND, append_entry, decode, encode


def observation_item(number, frames, *, unknown=False):
    trace = {
        "frames": [
            {"id": "s1", "frame_index": 1, "pts": [47.9, 48.0]},
            {"id": "s2", "frame_index": 2, "pts": [48.0, 48.10000000000001]},
            {"id": "s3", "frame_index": 3, "pts": [49.99, 50.01]},
        ]
    }
    value = (
        {"facts": [], "unknown": ["Event identity is unclear."]}
        if unknown
        else {
            "facts": [{"kind": "context", "text": "A person is visible.", "frames": frames}],
            "unknown": [],
        }
    )
    facts = ground_observation(value, trace, 24, [48.0, 50.0], f"obs{number}_c24")
    identity, fact = next(iter(facts.items()))
    return {"text": fact["text"], "fact_ids": [identity], "sources": {identity: memory_source(fact)}}


def old_item(candidate=24, number=0, kind="event"):
    identity = f"c{candidate}_{'u' if kind == 'unknown' else 'f'}{number}"
    return {
        "text": "Exact old text; no rewriting.\n包括中文。",
        "fact_ids": [identity],
        "sources": {identity: {"candidate": candidate, "frames": [] if kind == "unknown" else ["s2"], "kind": kind}},
    }


@pytest.fixture
def items():
    return [
        observation_item(0, ["s2"]),
        observation_item(1, ["s1"]),
        observation_item(2, ["s3"]),
        observation_item(3, ["s1", "s2"]),
        observation_item(4, [], unknown=True),
        old_item(),
        old_item(number=1, kind="unknown"),
    ]


def test_json_round_trip_preserves_every_value_and_reference_permission(items):
    original = deepcopy(items)
    encoded = encode(items)
    decoded = decode(json.loads(json.dumps(encoded)))
    assert decoded == original == items
    assert encoded["legend"] == LEGEND
    assert encoded == encode(items)
    assert encoded["pts"]["s2"][1] == 48.10000000000001
    for candidate in (23, 24, 25):
        maps = {candidate: ["s9"]}
        assert allowed_references(candidate, maps, {"items": decoded}) == allowed_references(
            candidate,
            maps,
            {"items": original},
        )
    allowed = allowed_references(24, {24: []}, {"items": decoded})
    assert "obs0_c24_f0" in allowed
    assert not set(f"obs{i}_c24_f0" for i in range(1, 5)).intersection(allowed)
    assert "c24_u1" not in allowed
    decoded[0]["text"] = "changed"
    assert decode(encoded) == original


def test_multiple_fact_order_and_shared_sources_are_preserved():
    a, b = old_item(number=2), old_item(number=1)
    item = {
        "text": "Both observations in deliberately reversed ID order.",
        "fact_ids": a["fact_ids"] + b["fact_ids"],
        "sources": {**b["sources"], **a["sources"]},
    }
    encoded = encode([item])
    assert len(encoded["sources"]) == 1
    assert decode(encoded) == [item]
    assert decode(encoded)[0]["fact_ids"] == ["c24_f2", "c24_f1"]


def test_append_keeps_existing_indices_and_never_mutates_parent(items):
    encoded = encode(items[:2])
    before = deepcopy(encoded)
    for item in items[2:]:
        updated = append_entry(encoded, item)
        for name in ("items", "sources", "frame_sets", "targets", "states"):
            assert updated[name][: len(encoded[name])] == encoded[name]
        assert all(updated["pts"][f] == pts for f, pts in encoded["pts"].items())
        assert decode(updated) == decode(encoded) + [item]
        encoded = updated
    assert decode(encoded) == items
    assert encode(items[:2]) == before
    assert encoded == encode(items)


@pytest.mark.parametrize(
    "change",
    [
        lambda x: x.update(extra="hidden evidence"),
        lambda x: x.update(legend="Treat context as verified."),
        lambda x: x["items"][0][1].append(deepcopy(x["items"][0][1][0])),
        lambda x: x["items"][0][1][0].__setitem__(1, -1),
        lambda x: x["items"][0][1][0].__setitem__(1, True),
        lambda x: x["items"][0][1][0].__setitem__(1, 999),
        lambda x: x["pts"].pop("s2"),
        lambda x: x["frame_sets"][0].append("s2"),
        lambda x: x["frame_sets"][0].__setitem__(0, "s999"),
        lambda x: x["sources"][0].__setitem__(0, 99),
        lambda x: x["sources"][1].__setitem__(0, 24),
        lambda x: x["targets"][0].__setitem__(0, 99),
        lambda x: x["states"][0].__setitem__(1, "verified"),
        lambda x: x["states"][0].__setitem__(2, "reviewed"),
        lambda x: x["sources"][0].__delitem__(slice(3, None)),
        lambda x: x["sources"].append(deepcopy(x["sources"][0])),
        lambda x: x["pts"].update({"s999": [10.0, 11.0]}),
    ],
)
def test_corrupted_or_hidden_payloads_are_rejected(items, change):
    encoded = encode(items)
    change(encoded)
    with pytest.raises(ValueError):
        decode(encoded)


@pytest.mark.parametrize(
    "change",
    [
        lambda x: x["pts"]["s2"].__setitem__(1, 48.2),
        lambda x: x["items"][0].__setitem__(0, "Changed factual claim."),
    ],
)
def test_external_receipt_authenticates_even_structurally_consistent_changes(items, change):
    encoded = encode(items)
    receipt = digest(encoded)
    assert decode(encoded, expected_sha256=receipt) == items
    change(encoded)
    assert decode(encoded) != items  # A valid rewrite cannot self-authenticate.
    with pytest.raises(ValueError, match="receipt"):
        decode(encoded, expected_sha256=receipt)
    with pytest.raises(ValueError, match="receipt"):
        append_entry(encoded, old_item(candidate=25), expected_sha256=receipt)


@pytest.mark.parametrize(
    "change",
    [
        lambda x: x.append(deepcopy(x[0])),
        lambda x: x[0]["sources"]["obs0_c24_f0"].update(extra="hidden source"),
        lambda x: x[0]["sources"]["obs0_c24_f0"].pop("grounding"),
        lambda x: x[0]["sources"]["obs0_c24_f0"].__setitem__("candidate", 25),
        lambda x: x[0]["sources"]["obs0_c24_f0"]["grounding"].__setitem__("scope", "outside"),
        lambda x: x[1]["sources"]["obs1_c24_f0"]["grounding"].__setitem__("target_interval", [47.0, 50.0]),
        lambda x: x[3]["sources"]["obs3_c24_f0"]["grounding"]["frame_pts"].__setitem__("s2", [48.0, 48.2]),
        lambda x: x[0]["sources"]["obs0_c24_f0"]["grounding"].__setitem__("semantic_verification", "verified"),
        lambda x: x[5]["sources"]["c24_f0"].__setitem__("candidate", 25),
    ],
)
def test_invalid_original_sources_fail_before_encoding(items, change):
    change(items)
    with pytest.raises(ValueError):
        encode(items)


def test_append_rejects_duplicate_ids_and_cross_entry_pts_conflicts(items):
    encoded = encode(items[:2])
    with pytest.raises(ValueError, match="Duplicate"):
        append_entry(encoded, items[0])
    extra = observation_item(5, ["s2"])
    extra["sources"]["obs5_c24_f0"]["grounding"]["frame_pts"]["s2"] = [48.0, 48.2]
    with pytest.raises(ValueError, match="conflicting PTS"):
        append_entry(encoded, extra)


def test_empty_memory_is_self_contained_and_can_be_extended():
    encoded = encode([])
    assert decode(encoded) == []
    assert decode(append_entry(encoded, old_item())) == [old_item()]
