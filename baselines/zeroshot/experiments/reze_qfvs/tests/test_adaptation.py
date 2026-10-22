"""Regression tests for prompt isolation, fixed budgets and paired visual evidence."""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared import query_text, select_summary, sha256  # noqa: E402

# isort: split
from adaptation import ACTION_MODE, CONCEPT_MODE, make_qfvs_request, temporal_summary
from pairing import check_paired_trace
from reze_backend import make_request


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        self.messages = copy.deepcopy(messages)
        self.options = kwargs
        return "<|video_pad|>" + str(messages)


def test_original_prompt_request_is_unchanged():
    clip = {"video": np.zeros((6, 2, 2, 3), dtype=np.uint8), "metadata": {"fps": 2}}
    tokenizer = Tokenizer()
    query = query_text(["Building", "Boat"])
    old = make_request(tokenizer, query, clip)
    new = make_qfvs_request(tokenizer, query, clip, ACTION_MODE)
    assert old["prompt"] == new["prompt"]
    assert old["mm_processor_kwargs"] == new["mm_processor_kwargs"]
    assert new["multi_modal_data"]["video"][0][0] is clip["video"]


def test_concept_prompt_preserves_video_and_uses_noun_or_question():
    clip = {"video": np.zeros((6, 2, 2, 3), dtype=np.uint8), "metadata": {"fps": 2}}
    tokenizer = Tokenizer()
    query = query_text(["Building", "Boat"], CONCEPT_MODE)
    request = make_qfvs_request(tokenizer, query, clip, CONCEPT_MODE)
    assert "architectural structure" in query
    assert "at least one" in tokenizer.messages[0]["content"]
    assert "strict video action verifier" not in request["prompt"]
    assert tokenizer.messages[1]["content"][0]["type"] == "video"
    assert request["multi_modal_data"]["video"][0] == (clip["video"], clip["metadata"])
    assert tokenizer.options["enable_thinking"] is False
    with pytest.raises(ValueError, match="Unknown"):
        make_qfvs_request(tokenizer, query, clip, "unknown")


def test_temporal_selection_changes_neighbors_only_with_fixed_budget():
    scores = [0.0] * 200
    for index, value in ((40, 1.0), (41, 0.9), (42, 0.8), (43, 0.7), (46, 0.6), (90, 0.5)):
        scores[index] = value
    assert select_summary(scores) == [40, 41, 42, 43]
    assert temporal_summary(scores) == [40, 43, 46, 90]
    assert temporal_summary(scores, radius=0) == select_summary(scores)


@pytest.mark.parametrize("n", [2152, 2783, 3588, 3692])
def test_temporal_budget_and_ties_for_all_native_video_lengths(n):
    selected = temporal_summary([0.5] * n)
    assert len(selected) == int(n * 0.02)
    assert selected == list(range(0, 3 * len(selected), 3))
    with pytest.raises(ValueError, match="budget"):
        temporal_summary([0.5] * n, fraction=1.0)


def test_pairing_rejects_pixel_drift_and_accepts_single_window_probes(tmp_path):
    trace = {
        "frame_indices": [[0, 1], [2, 3]],
        "requested_times": [[0, 1], [3, 4]],
        "frame_timestamps": [[[0, 1], [1, 2]], [[3, 4], [4, 5]]],
        "model_rgb_sha256": ["pixel-a", "pixel-b"],
        "original_height_width": [10, 20],
        "resized_height_width": [32, 64],
        "sampled_original_rgb_sha256": "original",
        "model_input_frames": 4,
        "decode_preprocess_seconds": 9,
    }
    chunk = {"segments": [[0, 3], [3, 6]], "trace": trace}
    path = tmp_path / "chunks/q/000000.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(chunk))
    receipt = {"run": str(tmp_path), "files": {"chunks/q/000000.json": sha256(path)}}
    check_paired_trace(receipt, {"qid": "q"}, 0, chunk["segments"], trace)
    single = {
        **trace,
        **{key: trace[key][1:] for key in ("frame_indices", "requested_times", "frame_timestamps", "model_rgb_sha256")},
    }
    single["decode_preprocess_seconds"] = 1
    check_paired_trace(receipt, {"qid": "q"}, 1, [[3, 6]], single)
    single["model_rgb_sha256"] = ["changed"]
    with pytest.raises(ValueError, match="model_rgb"):
        check_paired_trace(receipt, {"qid": "q"}, 1, [[3, 6]], single)
    path.write_text(json.dumps({**chunk, "extra": "modified"}))
    with pytest.raises(ValueError, match="chunk changed"):
        check_paired_trace(receipt, {"qid": "q"}, 0, chunk["segments"], trace)
