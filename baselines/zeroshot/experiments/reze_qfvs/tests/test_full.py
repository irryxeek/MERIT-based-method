"""Full validation keeps all queries, fixed inputs and a separate diagnostic subset."""

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aggregate_full import CONDITIONS, grouped_results, validate_pair  # noqa: E402
from pairing import check_paired_trace  # noqa: E402
from shared import sha256  # noqa: E402


def pair():
    rows = [
        {"qid": f"P{vid:02}__q{q}", "vid": f"P{vid:02}", "query": "action"} for q in range(45) for vid in range(1, 5)
    ]
    binding = {
        "scope": "full180",
        "prompt_mode": "reze_action_v1",
        "selected_qids": [r["qid"] for r in rows],
        "labels_sha256": {},
        "model_path": "model",
        "model_revision": "revision",
        "media_verification_sha256": "media",
        "window_seconds": 3,
        "sampling_fps": 2,
        "frames_per_full_window": 6,
        "batch_size": 16,
        "readout": "native",
        "summary_fraction": 0.02,
        "evaluator_revision": "official",
        "previously_diagnosed_qids": [r["qid"] for r in rows[:4]],
        "pixel_reference_sha256": "reference",
    }
    concept = {**binding, "prompt_mode": "visual_concept_or_v1", "selector_ablation": {"suppression_radius_shots": 2}}
    return [binding, concept], [rows, [{**r, "query": "concept"} for r in rows]]


def test_full_pair_accepts_only_query_wording_difference():
    bindings, rows = pair()
    validate_pair(bindings, rows)
    rows[1][10]["duration"] = 1
    with pytest.raises(ValueError, match="video inputs"):
        validate_pair(bindings, rows)


def test_full_pair_rejects_missing_queries_or_changed_parameters():
    bindings, rows = pair()
    with pytest.raises(ValueError, match="180"):
        validate_pair(bindings, [rows[0], rows[1][:-1]])
    bindings[1]["summary_fraction"] = 0.03
    with pytest.raises(ValueError, match="summary_fraction"):
        validate_pair(bindings, rows)


def test_report_separates_four_development_queries_from_remaining_176():
    bindings, manifests = pair()
    dev = set(bindings[0]["previously_diagnosed_qids"])
    rows = []
    for row in manifests[0]:
        metric = dict.fromkeys(("P", "R", "F1"), 100 if row["qid"] in dev else 0)
        conditions = {name: {"metrics_percent": {"oracle": metric, "user_mean": metric}} for name in CONDITIONS}
        rows.append({**row, "conditions": conditions})
    groups = grouped_results(rows, dev)
    assert groups["full180"]["n_queries"] == 180
    assert groups["remaining176"]["macro_metrics_percent"]["Concept_temporal"]["oracle"]["F1"] == 0
    assert groups["previously_diagnosed4"]["macro_metrics_percent"]["Concept_temporal"]["oracle"]["F1"] == 100
    with pytest.raises(ValueError, match="denominators"):
        grouped_results(rows[:-1], dev)


def test_new_queries_use_only_same_video_pixels_from_pilot(tmp_path):
    trace = {
        "frame_indices": [[0, 1]],
        "requested_times": [[0, 1]],
        "frame_timestamps": [[[0, 1], [1, 2]]],
        "model_rgb_sha256": ["pixels"],
        "original_height_width": [10, 20],
        "resized_height_width": [32, 64],
        "sampled_original_rgb_sha256": "original",
        "model_input_frames": 2,
    }
    path = tmp_path / "chunks/P01__old/000000.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"segments": [[0, 3]], "trace": trace}))
    receipt = {
        "run": str(tmp_path),
        "trace_qid_by_video": {"P01": "P01__old"},
        "files": {"chunks/P01__old/000000.json": sha256(path)},
    }
    row = {"qid": "P01__new", "vid": "P01"}
    check_paired_trace(receipt, row, 0, [[0, 3]], trace)
    changed = copy.deepcopy(trace)
    changed["model_rgb_sha256"] = ["other"]
    with pytest.raises(ValueError, match="model_rgb"):
        check_paired_trace(receipt, row, 0, [[0, 3]], changed)
    with pytest.raises(KeyError):
        check_paired_trace(receipt, {"qid": "P02__new", "vid": "P02"}, 0, [[0, 3]], trace)
