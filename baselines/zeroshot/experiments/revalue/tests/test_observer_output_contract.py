"""CPU regressions for the single-factor schema and fixed frame selection."""

import json
from copy import deepcopy

import pytest
from frame_observer import FRAME_OUTPUT_CAP, FRAME_PROMPT, FrameDescription
from io_utils import sha256
from observer_output_contract import RelaxedFrameDescription, select_jobs, unbounded_schema, validate_unbounded


def test_generation_schema_changes_only_two_maximum_lengths():
    original = FrameDescription.model_json_schema()
    expected = deepcopy(original)
    assert expected["properties"]["visible"].pop("maxLength") == 160
    assert expected["properties"]["uncertainty"].pop("maxLength") == 80
    assert unbounded_schema() == expected
    assert RelaxedFrameDescription.model_json_schema() == expected
    assert FrameDescription.model_json_schema() == original
    modified = unbounded_schema()
    modified["properties"]["visible"]["title"] = "Changed"
    assert unbounded_schema() == expected
    assert FRAME_OUTPUT_CAP == 160 and "one short, concrete sentence" in FRAME_PROMPT


def test_long_strings_are_returned_complete_without_added_semantic_filtering():
    value = {
        "visible": "  " + "A descriptive sentence with its original words preserved. " * 5,
        "uncertainty": "An explicitly unresolved detail remains uncertain. " * 4 + "  ",
    }
    assert len(value["visible"]) > 160 and len(value["uncertainty"]) > 80
    assert validate_unbounded(value) == value
    assert validate_unbounded({"visible": ",,,", "uncertainty": ""}) == {"visible": ",,,", "uncertainty": ""}
    assert validate_unbounded({"visible": "", "uncertainty": "Unclear."})["visible"] == ""


@pytest.mark.parametrize(
    "value",
    [
        {"visible": "  ", "uncertainty": "\n"},
        {"visible": 123, "uncertainty": "Unknown."},
        {"visible": None, "uncertainty": "Unknown."},
        {"visible": [], "uncertainty": "Unknown."},
        {"visible": "A person.", "uncertainty": "", "extra": "not allowed"},
    ],
)
def test_old_strict_type_and_nonblank_contract_remains(value):
    with pytest.raises(ValueError):
        validate_unbounded(value)


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def grounding(tmp_path):
    known = []
    for qid in range(1, 8):
        for arm_index, arm in enumerate(("E_uniform", "E_targeted")):
            for number in range(2):
                for position in range(24):
                    index = qid * 10000 + arm_index * 1000 + number * 100 + position
                    frame = {"id": f"s{index}", "frame_index": index, "pts": [position, position + 0.1]}
                    key = f"{qid}/{arm}/frame_observe/{number}/{position}"
                    response = tmp_path / "probe/attempts" / f"{index}.json"
                    _write(response, {"key": key, "request_sha256": "a" * 64, "text": "Original output."})
                    record = {
                        "qid": qid,
                        "arm": arm,
                        "number": number,
                        "position": position,
                        "key": key,
                        "proof": {"frames": [frame]},
                        "status": "complete",
                        "result": {"visible": "A person.", "uncertainty": ""},
                        "response_path": str(response),
                    }
                    _write(tmp_path / "frames" / f"{qid}_{arm}_{number}_{position:02d}.json", record)
                    if qid == 1 and arm_index == number == 0 and position in {*range(9), 12}:
                        known.append({"qid": qid, "frame": frame["id"], "issue": "Historical example."})
    _write(
        tmp_path / "review/assistant_findings.json",
        {
            "corrected_known_scene_misattributions": known[:7],
            "remaining_examples": known[7:],
        },
    )
    return tmp_path


def test_fixed_selection_is_sorted_unique_and_retains_parent_receipts(grounding):
    jobs = select_jobs(grounding)
    assert len(jobs) == len({job["key"] for job in jobs}) == 37
    identities = [(j["qid"], j["arm"], j["number"], j["position"]) for j in jobs]
    assert identities == sorted(identities)
    assert sum("systematic_position_12" in j["selection_reasons"] for j in jobs) == 28
    assert sum(len(j["selection_reasons"]) == 2 for j in jobs) == 1
    for job in jobs:
        assert job["record_sha256"] == sha256(job["record_path"])
        assert job["response_sha256"] == sha256(job["response_path"])
        assert job["record"]["proof"]["frames"][0] == job["frame"]
        assert job["request_sha256"] == "a" * 64


def test_selection_never_uses_gt_current_text_status_or_character_length(grounding):
    before = [(job["key"], job["selection_reasons"]) for job in select_jobs(grounding)]
    for path in (grounding / "frames").glob("*.json"):
        record = json.loads(path.read_text())
        record.update(status="invalid", result={"visible": "," * 999, "uncertainty": ""}, gt_score=999)
        _write(path, record)
    findings_path = grounding / "review/assistant_findings.json"
    findings = json.loads(findings_path.read_text())
    for group in findings.values():
        for item in group:
            item.update(issue="Different narrative, same frozen identity.", new="Altered description.")
    _write(findings_path, findings)
    _write(grounding / "labels.json", {"select_different_frames": True})
    assert [(job["key"], job["selection_reasons"]) for job in select_jobs(grounding)] == before


def test_identical_duplicate_keys_are_deduplicated_but_conflicts_fail(grounding):
    original = grounding / "frames/1_E_uniform_0_12.json"
    record = json.loads(original.read_text())
    duplicate = grounding / "frames/z_duplicate.json"
    _write(duplicate, record)
    assert len(select_jobs(grounding)) == 37
    record["result"]["visible"] = "Conflicting saved record."
    _write(duplicate, record)
    with pytest.raises(ValueError, match="Conflicting"):
        select_jobs(grounding)


def test_missing_source_occurrence_stops_instead_of_shrinking_sample(grounding):
    (grounding / "frames/1_E_uniform_0_00.json").unlink()
    with pytest.raises(ValueError, match="672"):
        select_jobs(grounding)


def test_missing_historical_example_stops_instead_of_substitution(grounding):
    path = grounding / "review/assistant_findings.json"
    findings = json.loads(path.read_text())
    findings["remaining_examples"][0]["frame"] = "s99999999"
    _write(path, findings)
    with pytest.raises(ValueError, match="regression frame is absent"):
        select_jobs(grounding)
