import itertools
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from dataset import author_splits  # noqa: E402
from dispatch import claim_free_gpu, free_gpu  # noqa: E402
from media import frame_intervals  # noqa: E402
from metrics import (  # noqa: E402
    ap_of_ranked_binary,
    author_top5_map,
    merge_details,
    random_expected_ap,
    summarize,
    two_second_scores,
    video_metrics,
)
from shared import (  # noqa: E402
    QUERY,
    atomic_json,
    extract_pyes,
    second_scores,
    segments_for,
    sha256,
    write_jsonl,
)


def test_readout_fractional_duration_and_shared_boundaries():
    segments = segments_for(7.9)
    curve = second_scores([0, 1, 0], segments, 7.9, 7)
    assert curve == [0, 0, 0, 0.5, 1, 1, 0.5]
    assert two_second_scores(curve, 3) == [0, 0.25, 1]
    with pytest.raises(ValueError, match="2-second grid"):
        two_second_scores(curve, 4)
    with pytest.raises(ValueError, match="curve length"):
        second_scores([0, 1, 0], segments, 7.9, 8)


@pytest.mark.parametrize("labels,expected", [([1, 1, 1], 1), ([0, 0, 0], 0), ([0, 1], 0.25), ([1, 0, 1], 19 / 24)])
def test_author_trapezoidal_ap(labels, expected):
    assert ap_of_ranked_binary(labels) == pytest.approx(expected)


@pytest.mark.parametrize("positives", range(7))
def test_random_expectation_matches_exhaustive_rankings(positives):
    labels = [1] * positives + [0] * (6 - positives)
    brute = np.mean([ap_of_ranked_binary([labels[i] for i in order]) for order in itertools.permutations(range(6), 5)])
    assert random_expected_ap(6, positives) == pytest.approx(brute)


@pytest.mark.parametrize("tied", [False, True])
def test_port_matches_unmodified_author_evaluator(tied):
    rng = np.random.default_rng(43)
    annotations, predictions, details = {}, {}, []
    for vid, n in (("a", 6), ("b", 17), ("c", 32)):
        labels = rng.integers(1, 6, size=(n, 20)).tolist()
        pred = [0.5] * n if tied else rng.random(n).tolist()
        annotations[vid] = {"anno": labels}
        predictions[vid] = pred
        details.append(video_metrics(pred, labels))
    assert author_top5_map(predictions, annotations) == pytest.approx(
        np.mean([d["Top5_AP"] for d in details]), abs=0.000501
    )


def test_torch_lower_median_and_no_positive_annotation():
    labels = np.array([1, 2, 2, 3, 4, 5])[:, None].repeat(20, axis=1)
    result = video_metrics([0, 0.1, 0.2, 0.3, 0.4, 1], labels)
    assert result["positive_clip_counts"] == [3] * 20
    empty = video_metrics([0.5] * 6, np.ones((6, 20)))
    assert empty["Top5_AP"] == empty["random_expected_Top5_AP"] == 0
    assert empty["top5_boundary_tie"] is True


def test_split_and_exact_coverage():
    splits = author_splits()
    val = [vid for s in splits.values() for vid in s["val"]]
    train = [vid for s in splits.values() for vid in s["train"]]
    assert len(val) == len(set(val)) == 10 and len(train) == 40
    assert not set(val) & set(train)
    with pytest.raises(ValueError, match="exact disjoint"):
        merge_details(["a", "b"], [[{"vid": "a"}], [{"vid": "a"}]])
    with pytest.raises(ValueError, match="exact disjoint"):
        merge_details(["a", "b"], [[{"vid": "a"}]])
    assert merge_details(["b", "a"], [[{"vid": "a"}], [{"vid": "b"}]]) == [{"vid": "b"}, {"vid": "a"}]


def test_macro_aggregation_uses_videos_not_shard_means():
    details = [{"Top5_AP": x, "constant_Top5_AP": 0, "random_expected_Top5_AP": 1} for x in (0, 0, 90)]
    summary = summarize(details)
    assert summary["Top5_AP"]["mean"] == 30
    assert summary["Top5_AP_minus_random_expectation"]["mean"] == 29


def test_gpu_with_compute_process_is_not_idle(monkeypatch):
    def output(cmd, **kwargs):
        return "4, GPU-test, 100, 0\n" if "--query-gpu=index,uuid,memory.used,utilization.gpu" in cmd else "GPU-test\n"

    monkeypatch.setattr(subprocess, "check_output", output)
    assert free_gpu(["GPU-test"])[0] is None


def test_two_shards_lease_different_cards_and_reuse_released_card(monkeypatch, tmp_path):
    import dispatch

    def available(candidates):
        return {"uuid": candidates[0], "index": 4}, {}

    monkeypatch.setattr(dispatch, "free_gpu", available)
    first, _, lease1 = claim_free_gpu(["GPU-a", "GPU-b"], tmp_path)
    second, _, lease2 = claim_free_gpu(["GPU-a", "GPU-b"], tmp_path)
    assert (first["uuid"], second["uuid"]) == ("GPU-a", "GPU-b")
    assert claim_free_gpu(["GPU-a", "GPU-b"], tmp_path)[0] is None
    lease1.close()
    again, _, lease3 = claim_free_gpu(["GPU-a", "GPU-b"], tmp_path)
    assert again["uuid"] == "GPU-a"
    lease2.close()
    lease3.close()


def test_duplicate_pts_kept_and_missing_ends_repaired():
    raw = np.array([[0, 0.5], [0.5, 0.5], [0.5, 0.5], [1, 1]])
    intervals, repairs = frame_intervals(raw, 2)
    assert repairs == 3
    assert intervals.tolist() == [[0, 0.5], [0.5, 1], [0.5, 1], [1, 1.5]]
    with pytest.raises(ValueError, match="nondecreasing"):
        frame_intervals([[0.5, 1], [0, 0.5]], 2)


def test_fresh_process_evaluation_reconstructs_raw_scores(tmp_path):
    """Exercise the real evaluator entry point, artifact gates and unmodified author method."""
    video = tmp_path / "video.mp4"
    video.write_bytes(b"synthetic metadata fixture; no decoder used by evaluation")
    row = {
        "qid": "fixture",
        "vid": "fixture",
        "query": QUERY,
        "duration": 12.0,
        "n_seconds": 12,
        "video_path": str(video),
        "media_size": video.stat().st_size,
        "media_mtime_ns": video.stat().st_mtime_ns,
        "media_sha256": sha256(video),
    }
    write_jsonl(tmp_path / "inference.jsonl", [row])
    segments = segments_for(12)
    atomic_json(tmp_path / "media_verification.json", {"fixture": {"decoded_timeline": {"segments": segments}}})
    labels = {"fixture": {"anno": np.array([1, 2, 3, 4, 5, 2])[:, None].repeat(20, axis=1).tolist()}}
    atomic_json(tmp_path / "evaluation_inputs/tvsum_anno.json", labels)
    atomic_json(
        tmp_path / "evaluation_inputs/dataset_audit.json",
        {"videos": {"fixture": {"n_2s_clips": 6, "category": "VT", "umt_split": "val"}}},
    )
    binding = {
        "n_queries": 1,
        "selected_qids": ["fixture"],
        "source_sha256": {},
        "scope": "synthetic",
        "inference_sha256": sha256(tmp_path / "inference.jsonl"),
        "batch_size": 16,
        "media_verification_sha256": sha256(tmp_path / "media_verification.json"),
        "expected_model_requests": 4,
        "labels_sha256": {
            n: sha256(tmp_path / "evaluation_inputs" / n) for n in ("tvsum_anno.json", "dataset_audit.json")
        },
    }
    atomic_json(tmp_path / "binding.json", binding)
    bh = sha256(tmp_path / "binding.json")
    records = []
    for prob in (0.1, 0.4, 0.7, 0.9):
        raw = {9454: math.log(prob), 2753: math.log(1 - prob)}
        p, mode = extract_pyes(raw, 9454, 2753)
        records.append({"pyes": p, "score_mode": mode, "top20": [{"id": k, "logprob": v} for k, v in raw.items()]})
    atomic_json(
        tmp_path / "chunks/fixture/000000.json",
        {
            "binding_sha256": bh,
            "qid": "fixture",
            "start": 0,
            "segments": segments,
            "trace": {"model_rgb_sha256": ["synthetic"] * 4},
            "records": records,
        },
    )
    atomic_json(
        tmp_path / "predictions/fixture.json",
        {"binding_sha256": bh, "second_scores": second_scores([r["pyes"] for r in records], segments, 12.0, 12)},
    )
    atomic_json(tmp_path / "inference_complete.json", {"binding_sha256": bh, "queries": 1})
    atomic_json(
        tmp_path / "query_summaries/fixture.json",
        {
            "binding_sha256": bh,
            "model_requests": 4,
            "input_tokens": 20,
            "frames": 24,
            "inference_seconds": 1,
            "decode_preprocess_seconds": 1,
            "generated_answers": {"No": 4},
            "score_modes": {"two_way": 4},
        },
    )
    atomic_json(tmp_path / "request_attempts/fixture.json", {"binding_sha256": bh, "requests": 4})
    atomic_json(tmp_path / "worker_attempts.json", [{"seconds": 2}])
    completed = subprocess.run(
        [sys.executable, str(HERE / "evaluate.py"), "--run", str(tmp_path)],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads((tmp_path / "evaluation.json").read_text())
    assert result["evaluated_videos"] == 1 and result["author_metric_parity_passed"]
    assert result["cost"]["formal_requests"] == 4


def build_aggregate_fixture(root):
    from aggregate_shards import PROTOCOL_FIELDS

    splits = author_splits()
    videos = [(v, cat, split) for cat, group in splits.items() for split, ids in group.items() for v in ids]
    plan = {
        "source_sha256": {},
        "selected_ids": [v for v, _, _ in videos],
        "umt_val_ids": [v for v, _, split in videos if split == "val"],
        "shards": [],
    }
    for index in (0, 1):
        part = videos[index::2]
        shard = root / f"shard_{index}"
        ids = [v for v, _, _ in part]
        plan["shards"].append({"index": index, "directory": shard.name, "video_ids": ids})
        binding = {name: "fixed" for name in PROTOCOL_FIELDS}
        binding.update(source_sha256={}, selected_qids=ids)
        atomic_json(shard / "binding.json", binding)
        details = [
            {
                "vid": vid,
                "category": cat,
                "umt_split": split,
                "Top5_AP": 80 if split == "val" else 40,
                "constant_Top5_AP": 10,
                "random_expected_Top5_AP": 20,
            }
            for vid, cat, split in part
        ]
        write_jsonl(shard / "per_video_metrics.jsonl", details)
        write_jsonl(shard / "hd_predictions.jsonl", [{"vid": v, "scores": [0.5]} for v in ids])
        result = {
            "binding_sha256": sha256(shard / "binding.json"),
            "author_metric_parity_passed": True,
            "per_video_metrics_sha256": sha256(shard / "per_video_metrics.jsonl"),
            "hd_predictions_sha256": sha256(shard / "hd_predictions.jsonl"),
            "evaluated_videos": 25,
            "cost": {
                "formal_requests": 25,
                "charged_requests_including_smoke_and_failed_attempts": 25,
                "worker_seconds": 10,
            },
            "diagnostics": {
                "answers": {"No": 25},
                "score_modes": {"two_way": 25},
                "top5_boundary_tie_videos": 0,
                "constant_prediction_videos": 0,
            },
        }
        atomic_json(shard / "evaluation.json", result)
        atomic_json(
            shard / "complete.json",
            {"binding_sha256": result["binding_sha256"], "evaluation_sha256": sha256(shard / "evaluation.json")},
        )
        atomic_json(shard / "worker_attempts.json", [{"started_epoch": 100 + index, "seconds": 10}])
    atomic_json(root / "plan.json", plan)


def call_aggregate(root):
    return subprocess.run(
        [sys.executable, str(HERE / "aggregate_shards.py"), "--run", str(root)],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_aggregate_fresh_process_separates_val10_from_all50(tmp_path):
    build_aggregate_fixture(tmp_path)
    completed = call_aggregate(tmp_path)
    assert completed.returncode == 0, completed.stderr
    result = json.loads((tmp_path / "evaluation.json").read_text())
    assert result["metrics_percent"]["umt_val10"]["Top5_AP"]["mean"] == 80
    assert result["metrics_percent"]["all50_zero_shot"]["Top5_AP"]["mean"] == 48
    assert result["cost"]["worker_seconds"] == 20
    assert result["cost"]["inference_wall_seconds_including_between_shard_queue"] == 11


def test_aggregate_rejects_changed_metrics_and_incomplete_scope(tmp_path):
    build_aggregate_fixture(tmp_path)
    completion = tmp_path / "shard_1/complete.json"
    saved = completion.read_bytes()
    completion.unlink()  # Synthetic fixture only: simulate a pending worker.
    pending = call_aggregate(tmp_path)
    assert pending.returncode == 0 and not (tmp_path / "complete.json").exists()
    assert json.loads((tmp_path / "status.json").read_text())["status"] == "in_progress"
    completion.write_bytes(saved)
    with (tmp_path / "shard_1/per_video_metrics.jsonl").open("a") as stream:
        stream.write("\n")
    rejected = call_aggregate(tmp_path)
    assert rejected.returncode != 0 and "artifact changed" in rejected.stderr
    assert not (tmp_path / "complete.json").exists()
