"""Map the benchmark's two-second centers to real decoded frame timestamps."""

import numpy as np


def select_centers(timestamps, duration):
    timestamps = np.asarray(timestamps)
    if timestamps.ndim != 2 or timestamps.shape[1] != 2 or not len(timestamps):
        raise ValueError("Expected nonempty [frame_start, frame_end] timestamps")
    if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps[:, 0]) < 0):
        raise ValueError("Invalid or unsorted frame timestamps")
    end = float(timestamps[-1, 1])
    if end <= 0 or duration - end >= 2 or end - duration > 1:
        raise ValueError(f"Unexpected decoded duration {end}, annotated {duration}")
    requested = 1.0 + 2.0 * np.arange(int(duration / 2))
    indices = np.clip(np.searchsorted(timestamps[:, 0], requested, side="right") - 1, 0, len(timestamps) - 1)
    # A selected frame must belong to its intended evaluation bin.
    if np.any(timestamps[indices, 1] <= 2.0 * np.arange(len(requested))):
        raise ValueError("A complete evaluation clip has no visual evidence")
    return indices, requested, end


def normalized_features(features):
    features = np.asarray(features, dtype=np.float32)
    norms = np.linalg.norm(features, axis=-1, keepdims=True)
    if not np.isfinite(features).all() or np.any(norms <= 0):
        raise ValueError("Invalid CLIP features")
    return features / norms
