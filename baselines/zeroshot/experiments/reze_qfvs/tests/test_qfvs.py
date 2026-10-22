import os
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared import qfvs_scores, query_text, select_summary, validate_chunk

# isort: split
from dispatch import free_gpu, worker_environment
from evaluate import semantic_metrics
from reze_core import extract_pyes, segments_for, segments_to_curve


def test_five_second_readout_keeps_inclusive_reze_boundaries():
    segments = [(0, 3), (3, 6), (6, 7.4)]
    scores = qfvs_scores([0.2, 0.8, 0.4], segments, 7.4, 2)
    assert scores == pytest.approx([0.38, 0.7])


@pytest.mark.parametrize(
    "duration,count", [(13911.066667, 2783), (18457.266667, 3692), (10756.733333, 2152), (17940.2, 3588)]
)
def test_actual_qfvs_tail_counts_without_padding(duration, count):
    segments = segments_for(duration)
    scores = qfvs_scores([0.4] * len(segments), segments, duration, count)
    assert len(scores) == count
    np.testing.assert_allclose(scores, 0.4)


def test_no_fake_score_for_unobserved_full_shot():
    with pytest.raises(ValueError, match="incompatible"):
        qfvs_scores([0.5], [(0, 3)], 3, 10)


def test_fractional_only_shot_requires_explicit_future_policy():
    segments = segments_for(5.2)
    with pytest.raises(ValueError, match="no scored integer second"):
        qfvs_scores([0.5] * len(segments), segments, 5.2, 2)


def test_summary_budget_and_ties_are_deterministic():
    assert select_summary([0.5] * 101) == [0, 1]
    values = [0.0] * 100
    values[80], values[90], values[40] = 0.9, 0.8, 0.8
    assert select_summary(values) == [80, 40]


@pytest.mark.parametrize("values", [[0.2, float("nan")], [-0.1] * 100, [1.1] * 100])
def test_invalid_scores_fail(values):
    with pytest.raises(ValueError):
        select_summary(values)


def test_query_combines_either_concept_without_using_tags():
    assert query_text(["Book", "Street"]) == "The video shows either book or street."
    assert "cup or glass" in query_text(["Cupglass", "Hands"])


def test_metric_matches_semantics_even_at_different_times():
    tags = [np.array([[1, 0], [1, 0], [0, 1]], dtype=int)]
    assert semantic_metrics([0], [1], tags, 0) == {"P": 100.0, "R": 100.0, "F1": 100.0}
    assert semantic_metrics([0], [2], tags, 0) == {"P": 0.0, "R": 0.0, "F1": 0.0}
    with pytest.raises(ValueError, match="duplicate"):
        semantic_metrics([0, 0], [1], tags, 0)


def test_metric_weighted_precision_and_recall():
    tags = [np.array([[1, 1], [1, 0], [0, 1]], dtype=int)]
    values = semantic_metrics([0], [1, 2], tags, 0)
    assert values == pytest.approx({"P": 50.0, "R": 25.0, "F1": 100 / 3})


def test_gpu_numeric_index_is_distinct_from_verified_uuid():
    with patch(
        "dispatch.subprocess.check_output",
        side_effect=[
            "7, GPU-busy, 100, 0\n4, GPU-free, 15, 0\n",
            "GPU-busy\n",
        ],
    ):
        gpu, _ = free_gpu(["GPU-busy", "GPU-free"])
    assert gpu["index"] == 4
    env = worker_environment(gpu)
    assert env["CUDA_VISIBLE_DEVICES"] == "4"
    assert env["REZE_EXPECTED_GPU_UUID"] == "GPU-free"


def test_gpu_with_compute_process_is_never_selected():
    with patch("dispatch.subprocess.check_output", side_effect=["4, GPU-a, 15, 0\n", "GPU-a\n"]):
        gpu, _ = free_gpu(["GPU-a"])
    assert gpu is None


def test_same_video_different_query_cannot_reuse_checkpoint():
    p, mode = extract_pyes({9454: -1.0, 2753: -2.0}, 9454, 2753)
    data = {
        "binding_sha256": "run",
        "qid": "P01__A_B",
        "start": 0,
        "segments": [[0, 3]],
        "trace": {"model_rgb_sha256": ["pixels"]},
        "records": [
            {"pyes": p, "score_mode": mode, "top20": [{"id": 9454, "logprob": -1.0}, {"id": 2753, "logprob": -2.0}]}
        ],
    }
    validate_chunk(data, "run", {"qid": "P01__A_B"}, 0, [(0, 3)])
    with pytest.raises(ValueError, match="another query"):
        validate_chunk(data, "run", {"qid": "P01__A_C"}, 0, [(0, 3)])
    data["records"][0]["pyes"] = 0.1
    with pytest.raises(ValueError, match="raw log"):
        validate_chunk(data, "run", {"qid": "P01__A_B"}, 0, [(0, 3)])


def test_readout_is_exactly_mean_of_frozen_curve():
    segments = segments_for(12.6)
    probabilities = [0.1, 0.8, 0.2, 0.9, 0.3]
    curve = segments_to_curve(probabilities, segments, 12.6)
    expected = [curve[:5].mean(), curve[5:10].mean(), curve[10:].mean()]
    assert qfvs_scores(probabilities, segments, 12.6, 3) == expected


def test_streaming_first_windows_match_qv_baseline_pixels():
    import json

    import torch
    from media import StreamVideo
    from reze_media import prepare_video

    torch.set_num_threads(2)
    manifest = Path((os.environ.get("ZEROSHOT_ROOT", "baselines/zeroshot") + "/manifests/reze_smoke_inference.jsonl"))
    if not manifest.exists():
        pytest.skip("Requires prepared local QV media manifests")
    row = json.loads(manifest.read_text().splitlines()[0])
    baseline, trace = prepare_video(row, limit_segments=3)
    video = StreamVideo(row)
    streamed, actual = video.read(0, 3)
    assert actual["frame_indices"] == trace["frame_indices"]
    assert actual["frame_timestamps"] == trace["frame_timestamps"]
    for x, y in zip(baseline, streamed):
        assert x["metadata"] == y["metadata"]
        assert x["rgb_sha256"] == y["rgb_sha256"]
        np.testing.assert_array_equal(x["video"], y["video"])
