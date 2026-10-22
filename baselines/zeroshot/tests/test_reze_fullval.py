"""Full-val adapter checks: exact denominator, original B0 cost, and paired evidence."""

import json
import os
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/reze_fullval"))
from runner import (  # noqa: E402
    audit_evidence,
    baseline_cost,
    check_complete,
    evaluate,
    exact_coverage,
    metric_arrays,
    protocol,
)


def row(qid=1):
    return {"qid": qid, "vid": "v_0_10", "query": "An event", "duration": 10, "video_path": "/unused"}


def test_full_denominator_rejects_missing_extra_and_duplicate_records():
    rows = [row(1), row(2)]
    exact_coverage(rows, {"1": "a", "2": "b"}, 2)
    for receipt in ({"1": "a"}, {"1": "a", "2": "b", "3": "c"}):
        with pytest.raises(ValueError, match="coverage"):
            exact_coverage(rows, receipt, 2)
    with pytest.raises(ValueError, match="Incomplete"):
        exact_coverage(rows, {"1": "a", "2": "b"}, 1550)
    with pytest.raises(ValueError, match="duplicate"):
        exact_coverage([row(1), row(1)], {"1": "a"}, 2)
    with pytest.raises(ValueError, match="unauthorized"):
        exact_coverage([row(1) | {"saliency_scores": [4]}], {"1": "a"}, 1)


@pytest.mark.parametrize("count_field", ["queries", "completed_queries"])
def test_both_completion_schemas_and_record_tampering(tmp_path, count_field):
    protocol.atomic_json(tmp_path / "binding.json", {"test": True})
    protocol.atomic_json(tmp_path / "records/1.json", {"test": "record"})
    protocol.atomic_json(tmp_path / "record_receipt.json", {"1": protocol.sha256(tmp_path / "records/1.json")})
    protocol.atomic_json(
        tmp_path / "completion.json",
        {
            "status": "complete",
            count_field: 1,
            "binding_sha256": protocol.sha256(tmp_path / "binding.json"),
            "record_receipt_sha256": protocol.sha256(tmp_path / "record_receipt.json"),
        },
    )
    check_complete(tmp_path, [row()])
    protocol.atomic_json(tmp_path / "records/1.json", {"test": "changed"})
    with pytest.raises(ValueError, match="Record receipt"):
        check_complete(tmp_path, [row()])


def test_original_full_baseline_cost_reconstructed_without_cost_field():
    record = {
        "clips": [{"prompt_tokens": 11}, {"prompt_tokens": 17}],
        "media": {"model_input_frames": 12},
        "generate_seconds": 1.5,
    }
    assert baseline_cost(record) == {
        "logical_requests": 2,
        "input_tokens": 28,
        "output_tokens": 2,
        "model_frames": 12,
        "inference_seconds": 1.5,
    }


def test_evidence_mismatch_cannot_pass_on_matching_call_counts():
    first = {"a": 1, "b": 2, "evidence_sha256": "a", "model_frames": 8}
    record = {"candidate_frames": {}, "arms": {a: {"requests": [deepcopy(first)]} for a in ("P_abs", "P_pair")}}
    assert audit_evidence(record, {}) == 1
    record["arms"]["P_pair"]["requests"][0]["evidence_sha256"] = "b"
    with pytest.raises(ValueError, match="Unmatched"):
        audit_evidence(record, {})


def test_partial_eval_refused_before_opening_labels(tmp_path, monkeypatch):
    protocol.atomic_json(tmp_path / "plan.json", {"queries": 1550})
    monkeypatch.setattr(protocol, "load_run", lambda run: ({}, [row()]))
    with pytest.raises(ValueError, match="Partial"):
        evaluate(tmp_path)


def test_official_metric_replay_including_tied_max_and_fallback():
    root = Path(os.environ.get("QVH_EVALUATOR_ROOT", "third_party/moment_detr"))
    labels = [
        {
            "qid": 1,
            "vid": "v",
            "query": "event",
            "duration": 4,
            "relevant_clip_ids": [0, 1],
            "saliency_scores": [[4, 4, 4], [1, 1, 1]],
        }
    ]
    fallback = [{"qid": 1, "vid": "v", "query": "event", "pred_saliency_scores": [0.5, 0.5]}]
    metrics, values = metric_arrays(fallback, labels, root)
    assert metrics["HL-min-VeryGood"]["HL-Hit1"] == 100
    assert values.shape == (1, 2) and values[0, 1] == 100
    # Same-score fallback remains identical in paired differences.
    _, repeated = metric_arrays(json.loads(json.dumps(fallback)), labels, root)
    np.testing.assert_array_equal(values - repeated, np.zeros((1, 2)))
