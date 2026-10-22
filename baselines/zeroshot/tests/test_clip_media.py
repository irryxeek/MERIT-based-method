import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from clip_media import normalized_features, select_centers


def test_exact_centers_and_incomplete_terminal_bin():
    times = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 4.8]])
    indices, requested, end = select_centers(times, 6)
    assert indices.tolist() == [1, 3, 4]
    assert requested.tolist() == [1, 3, 5]
    assert end == 4.8


def test_wholly_absent_clip_rejected():
    with pytest.raises(ValueError):
        select_centers([[0, 1], [1, 2], [2, 3], [3, 4]], 6)


def test_cosine_features_preserve_negative_similarity():
    features = normalized_features([[3, 4], [-3, -4]])
    np.testing.assert_allclose(features @ features[0], [1, -1], atol=1e-6)
    with pytest.raises(ValueError):
        normalized_features([[0, 0]])
