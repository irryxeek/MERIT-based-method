"""Select already-observed baseline tensors; never decode extra observations."""

import hashlib
import itertools

import numpy as np
from reze_media import prepare_video


class EvidenceUnavailable(ValueError):
    """A candidate has no eligible previously seen sample."""


def eligible_observations(media, candidate):
    """Keep actual observed times in the target interval, including short residuals."""
    start = 2.0 * candidate
    eligible = []
    for window in range(len(media["requested_times"])):
        for position, requested in enumerate(media["requested_times"][window]):
            if start <= requested < start + 2:
                eligible.append((requested, window, position))
    eligible.sort()
    if len(eligible) > 4:
        nominal = [start + i / 2 for i in range(4)]
        eligible = list(
            min(
                itertools.combinations(eligible, 4),
                key=lambda subset: (sum(abs(item[0] - t) for item, t in zip(subset, nominal)), subset),
            )
        )
    return eligible


def select_frames(clips, media, selected):
    targets, traces = {}, {}
    for candidate in selected:
        observations, video, local_indices = [], [], []
        start = 2.0 * candidate
        relative_times = []
        for requested, window, position in eligible_observations(media, candidate):
            frame = clips[window]["video"][position]
            video.append(frame)
            relative_times.append(requested - start)
            observations.append(
                {
                    "requested_time": requested,
                    "baseline_window": window,
                    "baseline_position": position,
                    "frame_index": media["frame_indices"][window][position],
                    "frame_timestamps": media["frame_timestamps"][window][position],
                    "resized_rgb_sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
                }
            )
        if not video:
            raise EvidenceUnavailable(f"Candidate {candidate} has no eligible baseline frame")
        array = np.stack(video)
        regular = all(abs(t * 2 - round(t * 2)) < 1e-8 for t in relative_times)
        timebase = 2.0 if regular else 1000000.0
        local_indices = [round(t * timebase) for t in relative_times]
        metadata = {
            "fps": timebase,
            "total_num_frames": int(2 * timebase),
            "frames_indices": local_indices,
            "do_sample_frames": False,
        }
        targets[candidate] = {"video": array, "metadata": metadata}
        traces[str(candidate)] = {
            "start": start,
            "end": start + 2,
            "frames": observations,
            "model_rgb_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
            "metadata": metadata,
            "relative_requested_times": relative_times,
            "metadata_timebase_hz": timebase,
            "metadata_timebase_is_source_fps": False,
            "frames_observed": len(video),
            "native_temporal_padding_frames": len(video) % 2,
        }
    return targets, traces


def prepare_candidates(row, baseline, selected):
    clips, media = prepare_video(row)
    for key in (
        "source_sha256",
        "segments",
        "requested_times",
        "frame_indices",
        "frame_timestamps",
        "resized_height_width",
        "model_rgb_sha256",
        "sampled_original_rgb_sha256",
    ):
        # JSON canonicalization avoids list-versus-tuple differences after reload.
        from protocol import digest

        if digest(media[key]) != digest(baseline["media"][key]):
            raise ValueError(f"Baseline evidence changed: {key}")
    targets, traces = select_frames(clips, media, selected)
    return targets, traces, media["decode_preprocess_seconds"]
