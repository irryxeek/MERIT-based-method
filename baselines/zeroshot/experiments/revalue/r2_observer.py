"""Paired real source-frame selection, verified against every frame seen by B0."""

import hashlib

import numpy as np
from io_utils import digest
from observer import paired_indices


def source_timeline(row, baseline):
    import decord

    reader = decord.VideoReader(row["video_path"], num_threads=1)
    timestamps = np.asarray(reader.get_frame_timestamp(np.arange(len(reader))))
    seen = set()
    for indices, pts in zip(baseline["media"]["frame_indices"], baseline["media"]["frame_timestamps"], strict=True):
        if len(indices) != len(pts):
            raise ValueError("Parent PTS count mismatch")
        for index, expected in zip(indices, pts, strict=True):
            if timestamps[index].tolist() != expected:
                raise ValueError("Parent B0 PTS no longer matches the source video")
            seen.add(index)
    return reader, timestamps, seen


def pair_actions(timestamps, actions, seen):
    used = {"uniform": set(), "targeted": set()}
    output = []
    for index, action in enumerate(actions):
        paired = paired_indices(
            timestamps,
            [action["allowed"]],
            [[action["start"], action["end"]]],
            seen,
            used["uniform"],
            used["targeted"],
            24,
        )
        output.append({"number": index, "action": action, **paired})
        if not paired["cancelled"]:
            for arm in used:
                used[arm].update(paired[arm])
    return output


def read_frames(reader, timestamps, ids, row, action, height_width, arm):
    import torch
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms.functional import resize

    if len(ids) < 2 or len(ids) > 24 or len(set(ids)) != len(ids):
        raise ValueError("Observation frame count/uniqueness violation")
    raw = reader.get_batch(ids).asnumpy()
    pixels = (
        resize(
            torch.from_numpy(raw.copy()).permute(0, 3, 1, 2),
            height_width,
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
    intervals = [action["allowed"]] if arm == "E_uniform" else [[action["start"], action["end"]]]
    trace = {
        "action": action,
        "sampling_intervals": intervals,
        "frames": frames,
        "displayed_frames": len(ids),
        "new_source_frames": len(ids),
        "padding_frames": len(ids) % 2,
        "repeated_source_frames": 0,
        "rgb_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
    }
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


def timeline_hash(timestamps):
    return digest(timestamps.tolist())
