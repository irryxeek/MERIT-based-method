"""Granularity controls: same pixels, distinct supports, and frozen provenance."""

import copy
import json
from pathlib import Path

import pytest
from highlight_evidence.granularity import extract_units, scoring_memory, units_for


class CaptureClient:
    def __init__(self, payload):
        self.payload = payload

    def complete(self, stage, content, max_tokens, validator, *, system):
        self.stage = stage
        self.content = content
        self.max_tokens = max_tokens
        return validator(copy.deepcopy(self.payload))


@pytest.fixture
def case():
    values = json.loads(
        (Path(__file__).resolve().parents[1] / "configs/highlight_value_schema.example.json").read_text()
    )["values"]
    frames = [
        {
            "image_id": f"f{i}",
            "clip_id": i,
            "pts": 2 * i + 1.0,
            "jpeg_sha256": f"hash{i}",
            "url": f"pixels{i}",
        }
        for i in range(5, 10)
    ]
    chunk = {"window_id": 1, "start": 10, "end": 20, "bin_ids": list(range(5, 10))}

    def payload(arm):
        return {
            "events": [
                {
                    **u,
                    "description": "Person raises an arm in the supplied picture",
                    "values": [
                        {
                            "dimension": v["id"],
                            "status": "unknown",
                            "level": None,
                            "cue": "insufficient observation",
                            "evidence_ids": [],
                            "reference_event_ids": [],
                        }
                        for v in values
                    ],
                }
                for u in units_for(chunk, arm)
            ],
            "links": [],
        }

    prior = copy.deepcopy(payload("C10")["events"][0])
    prior.update(id="w0:e0", start=0, end=10, evidence_ids=["f0"])
    return values, frames, chunk, payload, [prior]


def test_only_unit_spec_changes_between_memory_requests(case):
    values, frames, chunk, payload, prior = case
    requests = []
    for arm in ("C10", "F2"):
        client = CaptureClient(payload(arm))
        result = extract_units(client, frames, values, 30, chunk, arm, prior)
        assert client.max_tokens == 8192
        assert len(result["events"]) == (1 if arm == "C10" else 5)
        assert all(e["id"].startswith("g1:") for e in result["events"])
        text, data = client.content[0]["text"].split("\n", 1)
        data = json.loads(data)
        data.pop("observation_units")
        data.pop("schema_example_first_unit_only")
        requests.append((text, data, client.content[1:]))
    assert requests[0] == requests[1]


@pytest.mark.parametrize("violation", ["wrong_frame", "wrong_bounds", "missing_unit", "self_reference"])
def test_fine_units_cannot_claim_another_bins_pixels_or_omit_coverage(case, violation):
    values, frames, chunk, payload, prior = case
    data = payload("F2")
    event = data["events"][0]
    if violation == "wrong_frame":
        event["evidence_ids"] = ["f6"]
    elif violation == "wrong_bounds":
        event["end"] = 20
    elif violation == "missing_unit":
        data["events"].pop()
    else:
        event["values"][2]["reference_event_ids"] = ["e0"]
    with pytest.raises(ValueError):
        extract_units(CaptureClient(data), frames, values, 30, chunk, "F2", prior)


def test_frozen_context_is_not_rebound_to_new_fine_observations(case):
    values, frames, chunk, payload, prior = case
    data = payload("F2")
    data["events"][0]["values"][2].update(
        status="observed", level=1, evidence_ids=["f5"], reference_event_ids=["w0:e0"]
    )
    data["links"] = [{"source": "w0:e0", "target": "e0", "relation": "before", "evidence_ids": ["f0", "f5"]}]
    result = extract_units(CaptureClient(data), frames, values, 30, chunk, "F2", prior)
    memory = scoring_memory([result], prior)
    assert memory["reference_events"] == prior
    assert memory["links"][0]["target"] == "g1:e0"
    assert memory["selected_events"][0]["values"][2]["reference_event_ids"] == ["w0:e0"]


def test_invalid_optional_edge_remains_explicit_without_destroying_nodes(case):
    values, frames, chunk, payload, prior = case
    data = payload("F2")
    data["links"] = [{"source": "e0", "target": "w0:e0", "relation": "before", "evidence_ids": ["f0", "f5"]}]
    result = extract_units(CaptureClient(data), frames, values, 30, chunk, "F2", prior)
    assert len(result["events"]) == 5 and result["links"] == []
    assert "backwards" in result["rejected_links"][0]["reason"]


def test_final_partial_window_preserves_real_grid_support():
    window = {"bin_ids": [70, 71], "start": 140, "end": 144}
    assert [(u["start"], u["end"]) for u in units_for(window, "F2")] == [(140, 142), (142, 144)]
    assert [(u["start"], u["end"]) for u in units_for(window, "C10")] == [(140, 144)]
