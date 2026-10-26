"""Real source observations, original PTS and strictly paired new-frame budgets."""

import hashlib

import numpy as np
from schema import Action


def validate_action(action, duration):
    action = Action.model_validate(action).model_dump()
    if not action["question"].strip():
        raise ValueError("Observation needs an explicit evidence question")
    if action["tool"] != "Stitch" and len(action["intervals"]) != 1:
        raise ValueError("Only Stitch can contain multiple original intervals")
    for interval in action["intervals"]:
        if len(interval) != 2 or not 0 <= interval[0] < interval[1] <= duration:
            raise ValueError("Observation outside source bounds")
    return action


def available_indices(timestamps, intervals, excluded):
    times = np.asarray(timestamps)[:, 0]
    mask = np.zeros(len(times), dtype=bool)
    for start, end in intervals:
        mask |= (times >= start) & (times < end)
    return [int(i) for i in np.flatnonzero(mask) if int(i) not in excluded]


def uniform_indices(available, count):
    if count > len(available) or count < 2:
        raise ValueError("Insufficient unique source frames")
    return [available[i] for i in np.linspace(0, len(available) - 1, count).round().astype(int)]


def paired_indices(timestamps, allowed, targeted, baseline_seen, used_uniform=(), used_targeted=(), count=24):
    if count < 2 or count > 24:
        raise ValueError("Frame budget out of bounds")
    for lo, hi in targeted:
        if not any(a <= lo < hi <= b for a, b in allowed):
            raise ValueError("Targeted interval outside the paired permitted range")
    uniform = available_indices(timestamps, allowed, set(baseline_seen) | set(used_uniform))
    target = available_indices(timestamps, targeted, set(baseline_seen) | set(used_targeted))
    n = min(count, len(uniform), len(target))
    if n < 2:
        return {"cancelled": True, "reason": "fewer_than_two_matched_new_frames", "uniform": [], "targeted": []}
    return {"cancelled": False, "uniform": uniform_indices(uniform, n), "targeted": uniform_indices(target, n)}


def read_observation(row, action, excluded, resized_height_width):
    import decord
    import torch
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms.functional import resize

    action = validate_action(action, float(row["duration"]))
    reader = decord.VideoReader(row["video_path"], num_threads=1)
    timestamps = np.asarray(reader.get_frame_timestamp(np.arange(len(reader))))
    available = available_indices(timestamps, action["intervals"], set(excluded))
    count = min(action["frame_limit"], len(available))
    ids = uniform_indices(available, count)
    raw = reader.get_batch(ids).asnumpy()
    pixels = (
        resize(
            torch.from_numpy(raw.copy()).permute(0, 3, 1, 2),
            resized_height_width,
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        .permute(0, 2, 3, 1)
        .contiguous()
        .numpy()
    )
    frames = [
        {
            "id": f"s{i}",
            "frame_index": i,
            "pts": timestamps[i].tolist(),
            "source_rgb_sha256": hashlib.sha256(raw[k].tobytes()).hexdigest(),
            "rgb_sha256": hashlib.sha256(pixels[k].tobytes()).hexdigest(),
        }
        for k, i in enumerate(ids)
    ]
    trace = {
        "action": action,
        "frames": frames,
        "displayed_frames": count,
        "new_source_frames": count,
        "padding_frames": count % 2,
        "rgb_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
    }
    # Absolute source timestamps are preserved across disjoint Stitch segments.
    target = {
        "video": pixels,
        "metadata": {
            "fps": 1000000.0,
            "total_num_frames": int(row["duration"] * 1000000),
            "frames_indices": [round(float(timestamps[i, 0]) * 1000000) for i in ids],
            "do_sample_frames": False,
        },
    }
    return target, trace
