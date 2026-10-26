import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import dispersed_candidates, legacy, ordered_pairs, permutation_scores  # noqa: E402
from ranking import candidates, make_scores  # noqa: E402
from worker_v1 import Ledger  # noqa: E402


def test_dispersion_fixed_radius_ties_and_fill():
    selected, trace = dispersed_candidates([0.5] * 20)
    assert selected == [0, 3, 6, 9, 12]
    assert trace["fill_ids"] == []
    selected, trace = dispersed_candidates([0.9, 0.8, 0.7, 0.6, 0.5])
    assert selected == [0, 3, 1, 2, 4]
    assert trace["fill_ids"] == [1, 2, 4]
    assert dispersed_candidates([0.5, 0.4])[0] == [0, 1]
    with pytest.raises(ValueError):
        dispersed_candidates([float("nan")])


def test_arbitrary_candidate_permutation_changes_only_selected_clips():
    original = [0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
    selected = [0, 3, 6, 9, 1]
    records = [{"a": a, "b": b, "pyes": 0.9 if a > b else 0.1} for a, b in ordered_pairs(selected)]
    scores, ranking = permutation_scores(original, selected, records)
    assert ranking["order"] == [9, 6, 3, 1, 0]
    assert sorted(scores) == sorted(original)
    assert all(scores[i] == value for i, value in enumerate(original) if i not in selected)
    with pytest.raises(ValueError, match="Missing or duplicated"):
        permutation_scores(original, selected, records[:-1])
    with pytest.raises(ValueError, match="candidate set"):
        permutation_scores(original, [0, 0, 3, 6, 9], records)


def test_generalized_ranking_exactly_replays_legacy_for_c0():
    rng = np.random.default_rng(42)
    for _ in range(50):
        scores = rng.random(30).tolist()
        selected = candidates(scores)
        records = [{"a": a, "b": b, "pyes": float(rng.random())} for a, b in ordered_pairs(selected)]
        assert permutation_scores(scores, selected, records) == make_scores(scores, selected, records, "P_pair")


def test_inference_manifest_rejects_label_leakage():
    row = {"qid": 1, "vid": "x", "query": "q", "duration": 10, "video_path": "video"}
    legacy.validate_rows([row])
    with pytest.raises(ValueError, match="unauthorized"):
        legacy.validate_rows([{**row, "saliency_scores": [5]}])


def test_interrupted_reservations_keep_budget_and_completed_results_do_not_double_charge(tmp_path):
    binding = {
        "maximum_engineering_requests": 100,
        "maximum_formal_attempts": 40,
        "maximum_incremental_tokens": 1000000,
    }
    ledger = Ledger(tmp_path, binding)
    path, attempt = ledger.reserve(1, "formal", 20)
    assert ledger.totals()["formal"] == 20
    assert Ledger(tmp_path, binding).totals()["tokens"] == 20 * 4097
    records = [{"prompt_tokens": 100, "output_tokens": 1}] * 20
    ledger.commit(path, attempt, records)
    assert ledger.totals() == {"engineering": 0, "formal": 20, "tokens": 2020}
    ledger.reserve(2, "formal", 20)
    with pytest.raises(RuntimeError, match="budget"):
        ledger.reserve(2, "formal", 20)
    assert len(list((tmp_path / "request_attempts").glob("*.json"))) == 2
    assert json.loads(path.read_text())["status"] == "returned"


def test_token_budget_stops_before_model_request(tmp_path):
    ledger = Ledger(
        tmp_path,
        {"maximum_engineering_requests": 100, "maximum_formal_attempts": 2100, "maximum_incremental_tokens": 10},
    )
    with pytest.raises(RuntimeError, match="budget"):
        ledger.reserve(1, "formal", 20)
    assert not (tmp_path / "request_attempts").exists()
