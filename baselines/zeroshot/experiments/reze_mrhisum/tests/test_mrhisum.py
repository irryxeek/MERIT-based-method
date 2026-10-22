import itertools
import os
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dispatch import free_gpu, worker_environment
from evaluate import expected_random_ap, hd_metrics, summary_metric
from media import frame_intervals
from prepare import select_videos
from reze_core import extract_pyes
from shared import QUERY, second_scores, validate_chunk


def test_native_grid_keeps_reze_boundary_averaging_and_partial_five_second_bin():
    scores = second_scores([0.2, 0.8, 0.4], [(0, 3), (3, 6), (6, 7)], 7, 7)
    assert scores == pytest.approx([0.2, 0.2, 0.2, 0.5, 0.8, 0.8, 0.6])
    assert np.mean(scores[:5]) == pytest.approx(0.38)
    assert np.mean(scores[5:]) == pytest.approx(0.7)


def test_zero_duration_decoder_intervals_use_next_pts_without_changing_frame_starts():
    raw = np.array([[0.0, 0.0], [0.04, 0.04], [0.08, 0.08]])
    fixed, count = frame_intervals(raw, 25)
    assert count == 3
    np.testing.assert_array_equal(fixed[:, 0], raw[:, 0])
    np.testing.assert_allclose(fixed[:, 1], [0.04, 0.08, 0.12])
    valid, count = frame_intervals(fixed, 25)
    assert count == 0
    np.testing.assert_array_equal(fixed, valid)


def test_duplicate_frame_starts_keep_pts_and_existing_right_sided_sampling():
    raw = np.array([[0.0, 0.04], [0.04, 0.08], [0.04, 0.08], [0.08, 0.12]])
    fixed, count = frame_intervals(raw, 25)
    np.testing.assert_array_equal(fixed, raw)
    assert count == 0
    assert np.searchsorted(fixed[:, 0], 0.04, side="right") - 1 == 2
    with pytest.raises(ValueError, match="nondecreasing"):
        frame_intervals(raw[[0, 3, 1, 2]], 25)


def test_zero_duration_with_duplicate_pts_uses_next_strictly_later_start():
    raw = np.array([[0.0, 0.0], [0.04, 0.04], [0.04, 0.04], [0.08, 0.08]])
    fixed, count = frame_intervals(raw, 25)
    np.testing.assert_array_equal(fixed[:, 0], raw[:, 0])
    np.testing.assert_allclose(fixed[:, 1], [0.04, 0.08, 0.08, 0.12])
    assert count == 4


def test_real_video_with_equal_tail_pts_can_be_sampled_without_retiming():
    import json

    import torch
    from media import StreamVideo

    torch.set_num_threads(2)
    root = Path((os.environ.get("REPRO_DATA_ROOT", "data") + "/MR.HiSum/raw/retries/20261021_v1"))
    if not (root / "effective_receipts.json").exists():
        pytest.skip("Requires historical MR.HiSum video receipts")
    row = json.loads((root / "effective_receipts.json").read_text())["video_31260"]
    video = StreamVideo({"vid": "video_31260", "video_path": row["path"], "duration": row["annotated_seconds"]})
    assert video.duplicate_start_pairs == 1
    np.testing.assert_array_equal(video.timestamps, video.raw_timestamps)
    for index in [0, len(video.segments) // 2, len(video.segments) - 1]:
        clips, trace = video.read(index, 1)
        assert len(clips) == 1
        expected = np.searchsorted(video.raw_timestamps[:, 0], trace["requested_times"][0], side="right") - 1
        np.testing.assert_array_equal(trace["frame_indices"][0], expected)


def test_real_vp9_with_zero_decoder_durations_can_be_sampled():
    import json

    import torch
    from media import StreamVideo

    torch.set_num_threads(2)
    root = Path((os.environ.get("REPRO_DATA_ROOT", "data") + "/MR.HiSum/raw/retries/20261021_v1"))
    if not (root / "effective_receipts.json").exists():
        pytest.skip("Requires historical MR.HiSum video receipts")
    record = json.loads((root / "effective_receipts.json").read_text())["video_9757"]
    video = StreamVideo({"vid": "video_9757", "video_path": record["path"], "duration": record["annotated_seconds"]})
    assert video.repaired_intervals == len(video.reader)
    for start in (0, len(video.segments) // 2, len(video.segments) - 1):
        clips, trace = video.read(start, 1)
        assert len(clips) == 1
        assert all(end > left for _, end in trace["frame_timestamps"][0] for left in [video.segments[start][0]])


@pytest.mark.parametrize("duration,count", [(7, 8), (7.4, 7)])
def test_grid_mismatch_is_not_silently_truncated(duration, count):
    with pytest.raises(ValueError, match="annotation grid"):
        second_scores([0.2], [(0, 3)], duration, count)


def test_official_five_second_metrics_perfect_and_constant_rankings():
    gt = np.repeat(np.arange(11) / 10, 5).tolist() + [0.95]
    perfect = hd_metrics(gt, gt)
    constant = hd_metrics([0.5] * len(gt), gt)
    assert perfect["n_5s_shots"] == 12
    assert perfect["AP50"] == pytest.approx(100)
    assert perfect["AP15"] == pytest.approx(100)
    assert constant["AP50"] == pytest.approx(50)
    assert constant["AP15"] == pytest.approx(100 / 12)


def test_native_gt_top_fraction_tie_is_preserved():
    gt = np.repeat(np.arange(11) / 10, 5).tolist() + [1.0]
    values = hd_metrics(gt, gt)
    assert values["gt_boundary_tie_15"]
    assert values["AP15"] == pytest.approx(50)


def test_expected_random_ap_matches_exhaustive_rankings():
    values = []
    for positions in itertools.combinations(range(5), 2):
        target = [int(i in positions) for i in range(5)]
        values.append(average_precision_score(target, [5, 4, 3, 2, 1]))
    assert expected_random_ap(5, 2) == pytest.approx(np.mean(values), abs=1e-12)


def test_native_summary_keeps_published_shared_endpoint_convention():
    cps = [[i, min(i + 10, 99)] for i in range(0, 100, 10)]
    scores = [0.0] * 90 + [1.0] * 10
    gt = [0] * 90 + [1] * 9 + [0]
    result = summary_metric(scores, cps, gt)
    assert result["prediction_length"] == 99
    assert result["selected_seconds"] == 9
    assert result["F1"] == pytest.approx(100, abs=1e-4)


def test_media_selection_preserves_scope_and_excludes_unavailable_or_unaligned_video():
    ids = ["video_1", "video_2", "video_3", "video_4"]
    records = {vid: {"status": "downloaded_verified"} for vid in ids}
    records["video_2"]["status"] = "source_unavailable"
    records["video_3"]["status"] = "downloaded_timeline_review"
    selected, eligible = select_videos(ids, records, 2)
    assert set(selected) == set(eligible) == {"video_1", "video_4"}
    assert QUERY == "Highlight of this video."
    with pytest.raises(ValueError, match="disagree"):
        select_videos(ids[:-1], records, 1)


def test_changed_raw_probabilities_or_video_id_cannot_reuse_checkpoint():
    p, mode = extract_pyes({9454: -1.0, 2753: -2.0}, 9454, 2753)
    data = {
        "binding_sha256": "run",
        "qid": "video_1",
        "start": 0,
        "segments": [[0, 3]],
        "trace": {"model_rgb_sha256": ["pixels"]},
        "records": [
            {"pyes": p, "score_mode": mode, "top20": [{"id": 9454, "logprob": -1.0}, {"id": 2753, "logprob": -2.0}]}
        ],
    }
    validate_chunk(data, "run", {"qid": "video_1"}, 0, [(0, 3)])
    with pytest.raises(ValueError, match="another query"):
        validate_chunk(data, "run", {"qid": "video_2"}, 0, [(0, 3)])
    data["records"][0]["pyes"] = 0.5
    with pytest.raises(ValueError, match="raw log"):
        validate_chunk(data, "run", {"qid": "video_1"}, 0, [(0, 3)])


def test_dispatch_never_uses_an_occupied_gpu_and_checks_uuid_mapping():
    with patch(
        "dispatch.subprocess.check_output", side_effect=["6, GPU-busy, 100, 0\n7, GPU-free, 15, 0\n", "GPU-busy\n"]
    ):
        gpu, _ = free_gpu(["GPU-busy", "GPU-free"])
    assert gpu["index"] == 7
    env = worker_environment(gpu)
    assert env["CUDA_VISIBLE_DEVICES"] == "7"
    assert env["REZE_EXPECTED_GPU_UUID"] == "GPU-free"


def test_actual_streaming_pixels_and_frame_metadata_match_qv_baseline():
    import json

    import torch
    from media import StreamVideo
    from reze_media import prepare_video

    torch.set_num_threads(2)
    path = Path((os.environ.get("ZEROSHOT_ROOT", "baselines/zeroshot") + "/manifests/reze_smoke_inference.jsonl"))
    if not path.exists():
        pytest.skip("Requires prepared local QV media manifests")
    row = json.loads(path.read_text().splitlines()[0])
    baseline, trace = prepare_video(row, limit_segments=3)
    streamed, actual = StreamVideo(row).read(0, 3)
    assert actual["frame_indices"] == trace["frame_indices"]
    assert actual["frame_timestamps"] == trace["frame_timestamps"]
    for a, b in zip(baseline, streamed):
        assert a["metadata"] == b["metadata"]
        assert a["rgb_sha256"] == b["rgb_sha256"]
        np.testing.assert_array_equal(a["video"], b["video"])
