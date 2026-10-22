import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared import svh_scores, validate_chunk

# isort: split
from dispatch import free_gpu, worker_environment
from evaluate import strict_predictions
from reze_core import extract_pyes, highlight_scores, segments_for


def test_even_timeline_is_identical_to_qv_readout():
    segments = segments_for(12.6)
    probabilities = [0.1, 0.8, 0.2, 0.9, 0.5]
    assert svh_scores(probabilities, segments, 12.6, 6) == highlight_scores(probabilities, segments, 12.6)


def test_partial_final_bin_is_retained_without_padding():
    segments = segments_for(7.4)
    assert svh_scores([0.2, 0.8, 0.4], segments, 7.4, 4) == pytest.approx([0.2, 0.35, 0.8, 0.6])


def test_count_mismatch_is_rejected():
    with pytest.raises(ValueError, match="Media/grid"):
        svh_scores([0.3], [(0, 3)], 3, 1)


def test_evaluator_never_pads_or_truncates():
    with pytest.raises(ValueError, match="lengths"):
        strict_predictions([{"vid": "x", "pred_saliency_scores": [0.4]}], {"x": [0, 1]}, ["x"])


def test_duplicate_or_missing_predictions_rejected():
    with pytest.raises(ValueError, match="IDs"):
        strict_predictions([{"vid": "x", "pred_saliency_scores": [0.4]}] * 2, {"x": [0]}, ["x", "y"])


def test_gpu_with_small_compute_job_is_not_selected():
    with patch(
        "dispatch.subprocess.check_output",
        side_effect=[
            "3, GPU-a, 100, 0\n6, GPU-b, 15, 0\n",
            "GPU-a\n",
        ],
    ):
        selected, _ = free_gpu(["GPU-a", "GPU-b"])
    assert selected["uuid"] == "GPU-b"
    env = worker_environment(selected)
    assert env["CUDA_VISIBLE_DEVICES"] == "6"
    assert env["REZE_EXPECTED_GPU_UUID"] == "GPU-b"


def test_gpu_waits_when_all_devices_occupied():
    with patch("dispatch.subprocess.check_output", side_effect=["3, GPU-a, 2000, 0\n", "GPU-a\n"]):
        selected, _ = free_gpu(["GPU-a"])
    assert selected is None


def test_checkpoint_rejects_changed_binding_and_probability():
    p, mode = extract_pyes({9454: -1.0, 2753: -2.0}, 9454, 2753)
    data = {
        "binding_sha256": "abc",
        "vid": "x",
        "start": 0,
        "segments": [[0, 3]],
        "trace": {"model_rgb_sha256": ["pixel-hash"]},
        "records": [
            {"pyes": p, "score_mode": mode, "top20": [{"id": 9454, "logprob": -1.0}, {"id": 2753, "logprob": -2.0}]}
        ],
    }
    validate_chunk(data, "abc", {"vid": "x"}, 0, [(0, 3)])
    with pytest.raises(ValueError, match="another run"):
        validate_chunk(data, "changed", {"vid": "x"}, 0, [(0, 3)])
    data["records"][0]["pyes"] = 0.5
    with pytest.raises(ValueError, match="raw log"):
        validate_chunk(data, "abc", {"vid": "x"}, 0, [(0, 3)])


def test_streaming_matches_frozen_qv_frames_and_metadata():
    # Compare real local QV pixels, timestamps, resize and chunk boundaries.
    import torch
    from media import StreamVideo
    from reze_media import prepare_video

    torch.set_num_threads(2)
    manifest = Path((os.environ.get("ZEROSHOT_ROOT", "baselines/zeroshot") + "/manifests/reze_smoke_inference.jsonl"))
    if not manifest.exists():
        pytest.skip("Requires prepared local QV media manifests")
    row = json.loads(manifest.read_text().splitlines()[0])
    expected, trace = prepare_video(row, limit_segments=5)
    video = StreamVideo(row)
    first, first_trace = video.read(0, 2)
    second, second_trace = video.read(2, 3)
    actual = first + second
    assert first_trace["frame_indices"] + second_trace["frame_indices"] == trace["frame_indices"]
    assert first_trace["requested_times"] + second_trace["requested_times"] == trace["requested_times"]
    assert first_trace["frame_timestamps"] + second_trace["frame_timestamps"] == trace["frame_timestamps"]
    for left, right in zip(actual, expected):
        assert left["metadata"] == right["metadata"]
        assert left["rgb_sha256"] == right["rgb_sha256"]
        np.testing.assert_array_equal(left["video"], right["video"])
