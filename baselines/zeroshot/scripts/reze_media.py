"""Decode true frames once per video and construct independently scored short clips."""

import hashlib
import time

import numpy as np
from common import sha256
from reze_core import segments_for


def prepare_video(row, limit_segments=None):
    import decord
    import torch
    from qwen_vl_utils.vision_process import smart_resize
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms.functional import resize

    started = time.time()
    reader = decord.VideoReader(row["video_path"], num_threads=1)
    timestamps = np.asarray(reader.get_frame_timestamp(np.arange(len(reader))))
    observed = float(timestamps[-1, 1])
    if row["duration"] - observed >= 2 or observed - row["duration"] > 1:
        raise ValueError(f"Unexplained duration mismatch: {observed} vs {row['duration']}")
    duration = min(float(row["duration"]), observed)
    segments = segments_for(duration)
    if limit_segments is not None:
        segments = segments[:limit_segments]
    all_indices, clip_indices, requested_by_clip = [], [], []
    for start, end in segments:
        # Sampling offset and short-residual rounding are implementation choices:
        # the paper specifies 2 fps / six full-clip frames, but not these details.
        count = max(2, int(round((end - start) * 2)))
        requested = np.linspace(start, end, count, endpoint=False)
        indices = np.searchsorted(timestamps[:, 0], requested, side="right") - 1
        indices = np.clip(indices, 0, len(reader) - 1)
        if np.any(timestamps[indices, 1] <= start):
            raise ValueError("Sample falls outside the retained clip")
        clip_indices.append(indices.tolist())
        requested_by_clip.append(requested.tolist())
        all_indices.extend(indices.tolist())
    frames = reader.get_batch(all_indices).asnumpy()
    height, width = frames.shape[1:3]
    resized_h, resized_w = smart_resize(height, width, factor=32, min_pixels=128 * 32 * 32, max_pixels=768 * 32 * 32)
    clips, offset = [], 0
    for segment, indices in zip(segments, clip_indices):
        count = len(indices)
        original = frames[offset : offset + count]
        pixels = torch.from_numpy(original.copy()).permute(0, 3, 1, 2)
        pixels = resize(pixels, [resized_h, resized_w], interpolation=InterpolationMode.BICUBIC, antialias=True)
        video = pixels.permute(0, 2, 3, 1).contiguous().numpy()
        metadata = {
            "fps": count / (segment[1] - segment[0]),
            "duration": segment[1] - segment[0],
            "total_num_frames": count,
            "frames_indices": list(range(count)),
            "video_backend": "decord_presampled",
            "do_sample_frames": False,
        }
        clips.append({"video": video, "metadata": metadata, "rgb_sha256": hashlib.sha256(video.tobytes()).hexdigest()})
        offset += count
    trace = {
        "vid": row["vid"],
        "source_path": row["video_path"],
        "source_sha256": sha256(row["video_path"]),
        "annotation_duration": row["duration"],
        "decoded_duration": observed,
        "scored_duration": duration,
        "unobserved_tail_seconds": max(0.0, row["duration"] - duration),
        "dropped_residual_seconds": duration - segments[-1][1],
        "segments": segments,
        "frame_indices": clip_indices,
        "requested_times": requested_by_clip,
        "frame_timestamps": [timestamps[indices].tolist() for indices in clip_indices],
        "original_height_width": [height, width],
        "resized_height_width": [resized_h, resized_w],
        "sampled_original_rgb_sha256": hashlib.sha256(frames.tobytes()).hexdigest(),
        "model_rgb_sha256": [c["rgb_sha256"] for c in clips],
        "model_input_frames": sum(len(c["video"]) for c in clips),
        "decode_preprocess_seconds": time.time() - started,
    }
    return clips, trace
