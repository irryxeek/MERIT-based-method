import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from common import atomic_json, pad_vtg_tail, validate_manifest, validate_scores


def test_label_fields_never_enter_inference_manifest():
    row = {"qid": 1, "vid": "video", "query": "cooking", "duration": 6}
    validate_manifest([row])
    with pytest.raises(ValueError):
        validate_manifest([{**row, "saliency_scores": [4]}])
    with pytest.raises(ValueError):
        validate_manifest([{**row, "relevant_windows": [[0, 2]]}])


def test_audited_missing_tail_is_zero_padded_not_interpolated():
    assert pad_vtg_tail([0.0, 1.0], 3) == [0.0, 1.0, 0.0]
    with pytest.raises(ValueError):
        pad_vtg_tail([0.0, 1.0], 4)
    with pytest.raises(ValueError):
        pad_vtg_tail([0.0, 1.0], 1)


@pytest.mark.parametrize("scores", [[float("nan")], [float("inf")], ["0.4"]])
def test_invalid_scores_stop_run(scores):
    with pytest.raises(ValueError):
        validate_scores(scores, 1)


def test_duplicate_queries_are_rejected():
    row = {"qid": 1, "vid": "video", "query": "cooking", "duration": 6}
    with pytest.raises(ValueError):
        validate_manifest([row, row])


def test_upstream_numpy_timestamps_are_preserved(tmp_path):
    import json

    import numpy as np

    output = tmp_path / "prediction.json"
    atomic_json(output, {"pred_relevant_windows": [[np.int64(2), np.int64(8), np.float32(0.5)]]})
    assert json.loads(output.read_text()) == {"pred_relevant_windows": [[2, 8, 0.5]]}
