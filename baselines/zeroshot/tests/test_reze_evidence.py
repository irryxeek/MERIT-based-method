"""Protocol invariants that protect the paired experiment from confounds."""

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/reze_evidence"))
import protocol  # noqa: E402
from backend import score_requests  # noqa: E402
from candidate_frames import EvidenceUnavailable, eligible_observations, select_frames  # noqa: E402
from ranking import candidates, make_scores, ordered_pairs, preference  # noqa: E402
from stage_a_eval import choose_arm  # noqa: E402
from worker import Ledger  # noqa: E402


def test_candidate_selection_and_all_orders():
    selected = candidates([0.1, 0.8, 0.8, 0.6, 0.9, 0.2, 0.1])
    assert selected == [4, 1, 2, 3, 5]
    pairs = ordered_pairs(selected)
    assert len(pairs) == len(set(pairs)) == 20
    assert all((b, a) in pairs for a, b in pairs)


@pytest.mark.parametrize(
    "forward,backward,expected", [(0.6, 0.4, 0.2), (0.4, 0.6, -0.2), (0.9, 0.8, 0), (0.1, 0.2, 0), (0.59, 0.4, 0)]
)
def test_neutral_and_symmetric_preference(forward, backward, expected):
    assert preference(forward, backward) == pytest.approx(expected)
    assert preference(backward, forward) == pytest.approx(-expected)


@pytest.mark.parametrize("arm", ["P_abs", "P_pair"])
def test_only_permute_original_candidate_scores(arm):
    original = [0.01, 0.8, 0.7, 0.6, 0.5, 0.9, 0.02]
    selected = candidates(original)
    records = [{"a": a, "b": b, "pyes": 0.9 if a < b else 0.1} for a, b in ordered_pairs(selected)]
    scores, diagnostics = make_scores(original, selected, records, arm)
    assert diagnostics["order"] == [1, 2, 3, 4, 5]
    assert scores[1] == 0.9 and scores[5] == 0.5
    assert sorted(scores) == sorted(original)
    assert scores[0] == original[0] and scores[6] == original[6]


def test_both_yes_and_ties_do_not_create_spurious_preference():
    original = [0.9, 0.9, 0.8, 0.7, 0.6, 0.01]
    selected = candidates(original)
    records = [{"a": a, "b": b, "pyes": 0.8} for a, b in ordered_pairs(selected)]
    scores, diagnostics = make_scores(original, selected, records, "P_pair")
    assert scores == original
    assert diagnostics["both_yes_pairs"] == diagnostics["neutral_unordered_pairs"] == 10


def test_cycles_and_missing_comparisons_are_not_hidden():
    original = [0.9, 0.8, 0.7, 0.6, 0.5]
    wins = {(0, 1), (1, 2), (2, 0)}
    records = [
        {"a": a, "b": b, "pyes": 0.9 if (a, b) in wins else 0.1 if (b, a) in wins else 0.5}
        for a, b in ordered_pairs(candidates(original))
    ]
    assert make_scores(original, candidates(original), records, "P_pair")[1]["cycles"] == 1
    with pytest.raises(ValueError, match="Missing or duplicated"):
        make_scores(original, candidates(original), records[:-1], "P_pair")


def test_observed_frames_cross_baseline_window_boundaries_byte_exactly():
    clips = [{"video": np.arange(start, start + 6, dtype=np.uint8).reshape(6, 1, 1, 1)} for start in (0, 6)]
    media = {
        "requested_times": [[i / 2 for i in range(6)], [i / 2 for i in range(6, 12)]],
        "frame_indices": [list(range(6)), list(range(6, 12))],
        "frame_timestamps": [[[i / 2, i / 2 + 0.04] for i in range(6)], [[i / 2, i / 2 + 0.04] for i in range(6, 12)]],
    }
    targets, traces = select_frames(clips, media, [1])
    assert targets[1]["video"].ravel().tolist() == [4, 5, 6, 7]
    assert [f["baseline_window"] for f in traces["1"]["frames"]] == [0, 0, 1, 1]
    assert targets[1]["metadata"]["frames_indices"] == [0, 1, 2, 3]
    with pytest.raises(EvidenceUnavailable):
        select_frames(clips, media, [4])


def test_short_tail_never_fabricates_an_observation():
    clips = [{"video": np.zeros((1, 1, 1, 3), dtype=np.uint8)}]
    media = {"requested_times": [[2.0]], "frame_indices": [[60]], "frame_timestamps": [[[2.0, 2.04]]]}
    targets, trace = select_frames(clips, media, [1])
    assert len(targets[1]["video"]) == 1
    assert trace["1"]["frames_observed"] == trace["1"]["native_temporal_padding_frames"] == 1


def test_irregular_residual_frame_is_retained_at_its_true_local_time():
    media = {
        "requested_times": [[144.0, 144.5293, 145.0586, 145.5879, 146.1172]],
        "frame_indices": [[4320, 4335, 4351, 4367, 4383]],
        "frame_timestamps": [[[i, i + 0.03] for i in [144.0, 144.5, 145.03, 145.56, 146.1]]],
    }
    clips = [{"video": np.arange(5, dtype=np.uint8).reshape(5, 1, 1, 1)}]
    targets, trace = select_frames(clips, media, [72, 73])
    assert targets[72]["video"].ravel().tolist() == [0, 1, 2, 3]
    assert targets[73]["video"].ravel().tolist() == [4]
    assert trace["73"]["frames"][0]["requested_time"] == 146.1172
    meta = targets[73]["metadata"]
    assert meta["frames_indices"][0] / meta["fps"] == pytest.approx(0.1172)
    assert trace["73"]["native_temporal_padding_frames"] == 1


def test_overfull_boundary_bin_uses_four_ordered_nearest_observed_samples():
    media = {"requested_times": [[146.0, 146.5], [147.0, 147.498, 147.996, 148.494]]}
    assert [x[0] for x in eligible_observations(media, 73)] == [146.0, 146.5, 147.0, 147.498]


def test_inference_manifest_rejects_ground_truth():
    row = {"qid": 1, "vid": "a", "query": "jump", "duration": 2, "video_path": "a.mp4"}
    protocol.validate_rows([row])
    with pytest.raises(ValueError, match="unauthorized"):
        protocol.validate_rows([row | {"saliency_scores": [4]}])


def test_missing_yes_logprob_is_invalid_only_in_new_arm():
    record = {"top20": [{"id": 2753, "logprob": -0.1}], "pyes": 0.0, "score_mode": "yes_missing_top20"}
    protocol.verify_logprob(record)
    with pytest.raises(ValueError, match="missing"):
        protocol.verify_logprob(record, strict=True)
    record = {
        "top20": [{"id": 2753, "logprob": math.log(0.2)}, {"id": 9454, "logprob": math.log(0.8)}],
        "pyes": 0.7,
        "score_mode": "two_way",
    }
    with pytest.raises(ValueError, match="replay"):
        protocol.verify_logprob(record, strict=True)


def test_attempt_budget_survives_resume(tmp_path):
    ledger = Ledger(tmp_path / "attempts.jsonl", 40)
    ledger.reserve(1, 20)
    resumed = Ledger(tmp_path / "attempts.jsonl", 40)
    assert resumed.used == 20
    resumed.reserve(2, 20)
    with pytest.raises(RuntimeError, match="budget"):
        resumed.reserve(3, 1)


def test_predeclared_gates_select_simpler_passing_arm_or_stop():
    results = {arm: {"fallback_queries": 0, "cold_total_token_ratio": 2} for arm in ("P_abs", "P_pair")}
    comparison = {
        "P_abs_minus_B0": {"delta_percentage_points": {"mAP": 1.1, "Hit1": 0}},
        "P_pair_minus_B0": {"delta_percentage_points": {"mAP": 1.3, "Hit1": 0}},
        "P_pair_minus_P_abs": {"delta_percentage_points": {"mAP": 0.2, "Hit1": 0}},
    }
    assert choose_arm(results, comparison)["selected_arm"] == "P_abs"
    results["P_abs"]["cold_total_token_ratio"] = 2.251
    assert choose_arm(results, comparison)["selected_arm"] == "P_pair"
    results["P_pair"]["fallback_queries"] = 2
    assert choose_arm(results, comparison)["selected_arm"] is None


def test_semantic_failure_is_recorded_without_regeneration():
    calls = []
    answer = SimpleNamespace(
        token_ids=[2753],
        text="No",
        finish_reason="length",
        logprobs=[{2753: SimpleNamespace(decoded_token="No", logprob=-0.1, rank=1)}],
    )

    def generate(requests, sampling, use_tqdm):
        calls.append(len(requests))
        return [SimpleNamespace(outputs=[answer], prompt_token_ids=list(range(10)))]

    result, _ = score_requests(
        SimpleNamespace(generate=generate), None, [{"prompt": "query"}], {"Yes": 9454, "No": 2753}
    )
    assert calls == [1]
    assert result[0]["error"] == "Required Yes/No logprob missing"
    assert result[0]["prompt_tokens"] == 10 and result[0]["output_tokens"] == 1
    assert "pyes" not in result[0]


def test_parent_completion_receipt_detects_modified_cache(tmp_path):
    row = {"qid": 1, "vid": "a", "query": "jump", "duration": 6, "video_path": "a.mp4"}
    binding = {"video_sha256": {"a": "source_hash"}}
    protocol.atomic_json(tmp_path / "binding.json", binding)
    logs = {"top20": [{"id": i, "logprob": math.log(0.5)} for i in (9454, 2753)], "pyes": 0.5, "score_mode": "two_way"}
    record = {
        "status": "complete",
        "binding_sha256": protocol.sha256(tmp_path / "binding.json"),
        "prediction": {"qid": 1, "vid": "a", "query": "jump", "pred_saliency_scores": [0.5] * 3},
        "clips": [logs, logs],
        "media": {"source_sha256": "source_hash", "segments": [[0, 3], [3, 6]]},
    }
    protocol.atomic_json(tmp_path / "records/1.json", record)
    protocol.atomic_json(tmp_path / "record_receipt.json", {"1": protocol.sha256(tmp_path / "records/1.json")})
    protocol.atomic_json(
        tmp_path / "completion.json",
        {
            "binding_sha256": protocol.sha256(tmp_path / "binding.json"),
            "record_receipt_sha256": protocol.sha256(tmp_path / "record_receipt.json"),
        },
    )
    assert protocol.read_baseline(tmp_path, row) == record
    protocol.atomic_json(tmp_path / "record_receipt.json", {"1": "forged_hash"})
    with pytest.raises(ValueError, match="receipt changed"):
        protocol.read_baseline(tmp_path, row)
