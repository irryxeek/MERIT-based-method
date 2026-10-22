import copy
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from reze_run_io import validate_record


def fixture():
    row = {"qid": 7, "vid": "video", "query": "An action", "duration": 6}
    record = {
        "status": "complete",
        "binding_sha256": "bound",
        "prediction": {**{k: row[k] for k in ("qid", "vid", "query")}, "pred_saliency_scores": [0.5] * 3},
        "media": {"source_sha256": "video_hash", "segments": [(0, 3), (3, 6)]},
        "clips": [
            {
                "pyes": 0.5,
                "score_mode": "two_way",
                "top20": [{"id": 9454, "logprob": math.log(0.4)}, {"id": 2753, "logprob": math.log(0.4)}],
            }
        ]
        * 2,
    }
    return row, record, {"video_sha256": {"video": "video_hash"}}


def test_valid_checkpoint_recomputes_raw_probabilities_and_full_hd_grid():
    row, record, binding = fixture()
    validate_record(record, row, binding, "bound")


@pytest.mark.parametrize("mutation", ["qid", "binding", "video", "probability", "logprob", "hd_score"])
def test_inconsistent_checkpoint_fails_instead_of_becoming_a_cache_hit(mutation):
    row, original, binding = fixture()
    record = copy.deepcopy(original)
    if mutation == "qid":
        record["prediction"]["qid"] += 1
    elif mutation == "binding":
        record["binding_sha256"] = "different"
    elif mutation == "video":
        record["media"]["source_sha256"] = "different"
    elif mutation == "probability":
        record["clips"][0]["pyes"] = 0.7
    elif mutation == "logprob":
        record["clips"][0]["top20"][0]["logprob"] = math.log(0.1)
    else:
        record["prediction"]["pred_saliency_scores"][0] = 0.7
    with pytest.raises(ValueError):
        validate_record(record, row, binding, "bound")
