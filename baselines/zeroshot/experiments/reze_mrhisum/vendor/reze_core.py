"""REZE scoring and HD readout, reimplemented from arXiv:2608.04480v1 Appendix A."""

import math

import numpy as np
from scipy.interpolate import interp1d

SYSTEM_PROMPT = (
    "You are a strict video action verifier. Your default answer is "
    "'No'. Answer 'Yes' ONLY if you are highly confident the described "
    "action is clearly and actively occurring in the frames. If there "
    "is any doubt, answer 'No'. Output: a single word 'Yes' or 'No'."
)


def user_prompt(query, n_frames):
    return f"Action: {query}\nIs this action CLEARLY occurring in these {n_frames} frames? Answer:"


def segments_for(duration):
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Nonpositive or nonfinite duration")
    segments = []
    for start in np.arange(0.0, duration, 3.0):
        end = min(float(start) + 3.0, duration)
        if end - start >= 0.3 - 1e-9:
            segments.append((float(start), end))
    if not segments:
        raise ValueError("No retained segment")
    return segments


def extract_pyes(logprobs, yes_id, no_id):
    """Use the published top-20 fallback; retain a flag for auditing."""
    yes, no = logprobs.get(yes_id), logprobs.get(no_id)
    if any(not math.isfinite(x) or x > 1e-5 for x in logprobs.values()):
        raise ValueError("Invalid log probability")
    if yes is not None and no is not None:
        maximum = max(yes, no)
        ey, en = math.exp(yes - maximum), math.exp(no - maximum)
        return ey / (ey + en), "two_way"
    if yes is not None:
        return math.exp(yes), "no_missing_top20"
    return 0.0, "yes_missing_top20"


def segments_to_curve(scores, segments, duration):
    """Match the appendix's inclusive boundary seconds and nearest extrapolation."""
    if len(scores) != len(segments) or not len(scores):
        raise ValueError("Missing or mismatched segment scores")
    values = np.asarray(scores, dtype=float)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Scores outside [0,1]")
    n = max(1, int(duration))
    curve, counts = np.zeros(n), np.zeros(n)
    for score, (start, end) in zip(values, segments):
        if start < 0 or end <= start or end > duration + 1e-6:
            raise ValueError("Segment outside timeline")
        for index in range(max(0, int(start)), min(n, int(end) + 1)):
            curve[index] += score
            counts[index] += 1
    valid = counts > 0
    if not valid.any():
        raise ValueError("No covered second")
    curve[valid] /= counts[valid]
    if not valid.all():
        indices = np.where(valid)[0]
        if len(indices) > 1:
            curve = interp1d(indices, curve[indices], kind="nearest", fill_value="extrapolate")(np.arange(n))
        else:
            curve = np.full(n, curve[indices[0]])
    return curve


def highlight_scores(scores, segments, annotation_duration):
    curve = segments_to_curve(scores, segments, annotation_duration)
    count = int(annotation_duration / 2)
    if count < 1:
        raise ValueError("No evaluation clip")
    return curve[: 2 * count].reshape(count, 2).mean(axis=1).tolist()
