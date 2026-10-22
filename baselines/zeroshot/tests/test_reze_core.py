import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from reze_core import extract_pyes, highlight_scores, segments_for, segments_to_curve


def test_shared_boundary_seconds_follow_published_readout():
    curve = segments_to_curve([0.2, 0.8], [(0, 3), (3, 6)], 6)
    np.testing.assert_allclose(curve, [0.2, 0.2, 0.2, 0.5, 0.8, 0.8])
    np.testing.assert_allclose(highlight_scores([0.2, 0.8], [(0, 3), (3, 6)], 6), [0.2, 0.35, 0.8])


def test_probability_is_two_way_not_full_vocab_or_hard_label():
    value, mode = extract_pyes({1: math.log(0.06), 2: math.log(0.14), 3: math.log(0.8)}, 1, 2)
    assert mode == "two_way"
    assert value == pytest.approx(0.3)


def test_top20_fallback_preserves_published_behavior():
    assert extract_pyes({1: math.log(0.9)}, 1, 2) == (0.9, "no_missing_top20")
    assert extract_pyes({2: math.log(0.9)}, 1, 2) == (0.0, "yes_missing_top20")
    assert extract_pyes({3: math.log(0.9)}, 1, 2) == (0.0, "yes_missing_top20")


def test_short_residual_threshold_and_no_invented_segment():
    assert segments_for(6.29) == [(0.0, 3.0), (3.0, 6.0)]
    assert segments_for(6.3) == [(0.0, 3.0), (3.0, 6.0), (6.0, 6.3)]
    assert segments_for(2.0) == [(0.0, 2.0)]
    with pytest.raises(ValueError):
        segments_for(0.2)


def test_unobserved_tail_uses_nearest_per_second_score():
    np.testing.assert_allclose(segments_to_curve([0.7], [(0, 3)], 6), np.full(6, 0.7))
    assert len(highlight_scores([0.7], [(0, 3)], 7)) == 3


@pytest.mark.parametrize("scores", [[float("nan")], [1.1], [-0.1], []])
def test_invalid_segment_score_cannot_become_an_hd_prediction(scores):
    with pytest.raises(ValueError):
        highlight_scores(scores, [(0, 3)], 3)


def test_mismatched_temporal_support_fails():
    with pytest.raises(ValueError):
        segments_to_curve([0.5], [(0, 7)], 6)
    with pytest.raises(ValueError):
        extract_pyes({1: float("nan"), 2: -1.0}, 1, 2)
