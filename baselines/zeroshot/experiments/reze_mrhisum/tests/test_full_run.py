import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aggregate_shards import METRIC_NAMES, PROTOCOL_FIELDS, aggregate_metrics, main, merge_details  # noqa: E402
from prepare import execution_limits, select_videos, shard_selection  # noqa: E402
from shared import atomic_json, sha256, write_jsonl  # noqa: E402


def test_all_available_shards_preserve_every_eligible_video_once():
    ids = [f"video_{i}" for i in range(11)]
    receipts = {vid: {"status": "downloaded_verified"} for vid in ids}
    receipts[ids[-1]]["status"] = "source_unavailable"
    receipts[ids[-2]]["status"] = "downloaded_timeline_review"
    selected, eligible = select_videos(ids, receipts, None)
    left, right = [shard_selection(selected, index, 2) for index in range(2)]
    assert set(left).isdisjoint(right)
    assert set(left + right) == set(eligible) == set(ids[:-2])
    assert len(left) - len(right) == 1
    assert select_videos(ids, receipts, 3)[0] == selected[:3]


@pytest.mark.parametrize("index,count", [(2, 2), (-1, 2), (0, 0), (0, 3)])
def test_invalid_shard_cannot_silently_skip_videos(index, count):
    with pytest.raises(ValueError):
        shard_selection(["a", "b"], index, count)


def test_full_run_limits_allow_full_inference_and_bound_combined_gpu_usage():
    pilot = execution_limits(False)
    full = execution_limits(True, 2)
    assert pilot["maximum_active_worker_seconds"] == 7200
    assert full["maximum_active_worker_seconds"] == 12 * 3600
    assert full["maximum_worker_attempts"] == 3
    assert full["request_overhead"] * 2 == 512


def metric_row(vid, score):
    return {"vid": vid, **{key: score for key in METRIC_NAMES}}


def test_macro_metrics_use_videos_not_unweighted_shard_averages():
    details = merge_details(["a", "b", "c"], [[metric_row("a", 10), metric_row("c", 30)], [metric_row("b", 80)]])
    assert [row["vid"] for row in details] == ["a", "b", "c"]
    assert aggregate_metrics(details)["AP50"]["mean"] == pytest.approx(40)
    with pytest.raises(ValueError, match="Duplicate"):
        merge_details(["a", "b"], [[metric_row("a", 1)], [metric_row("a", 2)]])
    with pytest.raises(ValueError, match="cover"):
        merge_details(["a", "b"], [[metric_row("a", 1)]])


def finished_shards(root):
    shard_ids = [["a", "c"], ["b"]]
    plan = {
        "scope": "test_available",
        "official_test_count": 4,
        "selected_ids": ["a", "b", "c"],
        "source_sha256": {},
        "shards": [{"index": i, "directory": f"shard_{i}", "video_ids": ids} for i, ids in enumerate(shard_ids)],
    }
    atomic_json(root / "plan.json", plan)
    for index, ids in enumerate(shard_ids):
        run = root / f"shard_{index}"
        binding = {key: "fixed" for key in PROTOCOL_FIELDS}
        binding.update(
            source_sha256={}, labels_sha256={}, selected_qids=ids, excluded_status_counts={"source_unavailable": 1}
        )
        atomic_json(run / "binding.json", binding)
        rows = [metric_row(vid, {"a": 10, "b": 80, "c": 30}[vid]) for vid in ids]
        write_jsonl(run / "per_video_metrics.jsonl", rows)
        write_jsonl(run / "summary_predictions.jsonl", [{"vid": vid, "predicted_summary": [0, 1]} for vid in ids])
        evaluation = {
            "binding_sha256": sha256(run / "binding.json"),
            "evaluated_videos": len(ids),
            "per_video_metrics_sha256": sha256(run / "per_video_metrics.jsonl"),
            "summary_predictions_sha256": sha256(run / "summary_predictions.jsonl"),
            "cost": {
                "worker_seconds": 10,
                "formal_requests": len(ids),
                "charged_requests_including_smoke_and_failed_attempts": len(ids) + 9,
            },
            "diagnostics": {
                "answers": {"Yes": len(ids)},
                "score_modes": {"two_way": len(ids)},
                "gt_boundary_tie_videos": {"50": 0, "15": 0},
                "summary_length_differences": {"-1": len(ids)},
                "constant_normalized_prediction_videos": 0,
            },
        }
        atomic_json(run / "evaluation.json", evaluation)
        atomic_json(run / "worker_attempts.json", [{"started_epoch": 100 + index, "seconds": 10}])
        atomic_json(run / "complete.json", {"evaluation_sha256": sha256(run / "evaluation.json")})
    return plan


def test_complete_report_checks_coverage_and_separates_wall_time_from_gpu_time(tmp_path, monkeypatch):
    finished_shards(tmp_path)
    monkeypatch.setattr(sys, "argv", ["aggregate_shards.py", "--run", str(tmp_path)])
    marker = tmp_path / "shard_1/complete.json"
    content = marker.read_text()
    marker.unlink()
    main()
    assert not (tmp_path / "evaluation.json").exists()
    assert json.loads((tmp_path / "status.json").read_text())["status"] == "in_progress"
    marker.write_text(content)
    main()
    result = json.loads((tmp_path / "evaluation.json").read_text())
    assert result["evaluated_videos"] == 3
    assert result["official_test_coverage"] == 0.75
    assert result["metrics_percent"]["AP50"]["mean"] == pytest.approx(40)
    assert result["cost"]["worker_seconds"] == 20
    assert result["cost"]["inference_wall_seconds"] == 11
    assert result["cost"]["formal_requests"] == 3
    assert (tmp_path / "complete.json").exists()


def test_changed_per_video_metrics_are_rejected_before_final_report(tmp_path, monkeypatch):
    finished_shards(tmp_path)
    path = tmp_path / "shard_0/per_video_metrics.jsonl"
    path.write_text(path.read_text() + json.dumps(metric_row("extra", 100)) + "\n")
    monkeypatch.setattr(sys, "argv", ["aggregate_shards.py", "--run", str(tmp_path)])
    with pytest.raises(ValueError, match="Per-video metrics changed"):
        main()
    assert not (tmp_path / "complete.json").exists()
