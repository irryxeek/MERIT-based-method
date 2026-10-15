"""Decode time-aligned visual evidence once, retaining source provenance."""

import json
from pathlib import Path

from .core import digest, file_sha256, windows, write_json


def sample_times(start, end, fps, offset):
    import numpy as np

    times = np.arange(start + offset, end, 1.0 / fps).tolist()
    return times or [(start + end) / 2]


def frame_indices(timestamps, requested):
    import numpy as np

    # Decord provides [start, end] for each frame; choose the frame displayed at t.
    starts = np.asarray(timestamps)[:, 0]
    return np.clip(np.searchsorted(starts, requested, side="right") - 1, 0, len(starts) - 1)


def observed_duration(annotation_duration, decoded_duration):
    # QVHighlights labels use 2-second bins; a terminal bin may be only partly visible.
    # Reject a wholly missing bin or an unexpectedly longer source instead of inventing frames.
    delta = annotation_duration - decoded_duration
    if delta >= 2.0 or delta < -1.0 or decoded_duration <= 0:
        raise ValueError(f"Video/annotation duration mismatch: {decoded_duration} vs {annotation_duration}")
    return min(annotation_duration, decoded_duration)


def prepare_frames(row, config, root):
    import decord
    import numpy as np
    from PIL import Image

    source = Path(row["video_path"])
    if not source.is_file():
        raise FileNotFoundError(source)
    source_hash = file_sha256(source)
    identity_data = {"source_sha256": source_hash, "duration": row["duration"], "media": config}
    legacy_identity = digest(identity_data)
    identity = digest({**identity_data, "temporal_schema": 2})
    target = Path(root) / "frames" / row["vid"] / identity
    manifest = target / "frames.json"
    if manifest.exists():
        result = json.loads(manifest.read_text())
        if all(Path(frame["path"]).is_file() for clip in result["clips"] for frame in clip["frames"]):
            return result
    # Reuse already decoded images when only the tail metadata needs correction.
    legacy_manifest = Path(root) / "frames" / row["vid"] / legacy_identity / "frames.json"
    if legacy_manifest.exists():
        result = json.loads(legacy_manifest.read_text())
        end_time = observed_duration(row["duration"], result["decoded_duration"])
        valid = all(
            clip["start"] < end_time
            and all(
                Path(frame["path"]).is_file() and frame["requested_time"] < end_time
                for frame in clip["frames"]
            )
            for clip in result["clips"]
        )
        if valid:
            for clip in result["clips"]:
                clip["end"] = min(clip["end"], end_time)
            result.update(
                identity=identity,
                temporal_schema=2,
                reused_frames_from=str(legacy_manifest),
                unobserved_tail_seconds=max(0.0, row["duration"] - end_time),
            )
            write_json(manifest, result)
            return result
    reader = decord.VideoReader(str(source), ctx=decord.cpu(0), num_threads=1)
    timestamps = np.asarray(reader.get_frame_timestamp(np.arange(len(reader))))
    actual_duration = float(timestamps[-1, 1])
    end_time = observed_duration(row["duration"], actual_duration)
    target.mkdir(parents=True, exist_ok=True)
    clips = []
    for clip_id, (start, end) in enumerate(windows(end_time, config["clip_seconds"])):
        requested = sample_times(start, end, config["fps"], config["sample_offset_seconds"])
        indices = frame_indices(timestamps, requested)
        arrays = reader.get_batch(indices.tolist()).asnumpy()
        frames = []
        for requested_time, frame_id, array in zip(requested, indices, arrays, strict=True):
            frame = Image.fromarray(array)
            frame.thumbnail(
                (config["image_longest_edge"], config["image_longest_edge"]), Image.Resampling.LANCZOS
            )
            path = target / f"{clip_id:03d}_{int(frame_id):06d}.jpg"
            frame.save(path, "JPEG", quality=config["jpeg_quality"])
            frames.append(
                {
                    "requested_time": requested_time,
                    "time": float(timestamps[frame_id, 0]),
                    "frame_index": int(frame_id),
                    "path": str(path),
                    "sha256": file_sha256(path),
                }
            )
        clips.append({"clip_id": clip_id, "start": start, "end": end, "frames": frames})
    result = {
        "identity": identity,
        "source_sha256": source_hash,
        "source_path": str(source),
        "decoded_duration": actual_duration,
        "annotation_duration": row["duration"],
        "temporal_schema": 2,
        "unobserved_tail_seconds": max(0.0, row["duration"] - end_time),
        "clips": clips,
    }
    write_json(manifest, result)
    return result
