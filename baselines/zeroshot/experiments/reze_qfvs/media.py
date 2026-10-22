"""Bounded decoding with the exact QV REZE sampling and resize operations."""

import hashlib
import time

import numpy as np
from shared import segments_for


class StreamVideo:
    def __init__(self, row):
        import decord

        self.row = row
        self.reader = decord.VideoReader(row["video_path"], num_threads=1)
        self.timestamps = np.asarray(self.reader.get_frame_timestamp(np.arange(len(self.reader))))
        self.observed = float(self.timestamps[-1, 1])
        if row["duration"] - self.observed >= 2 or self.observed - row["duration"] > 1:
            raise ValueError(f"Unexplained duration mismatch: {self.observed} vs {row['duration']}")
        self.duration = min(float(row["duration"]), self.observed)
        self.segments = segments_for(self.duration)

    def describe(self):
        return {
            "vid": self.row["vid"],
            "media_sha256": self.row.get("media_sha256"),
            "annotation_duration": self.row["duration"],
            "decoded_duration": self.observed,
            "scored_duration": self.duration,
            "segments": self.segments,
            "first_video_pts": float(self.timestamps[0, 0]),
            "decoded_frame_count": len(self.reader),
        }

    def read(self, start, size):
        import torch
        from qwen_vl_utils.vision_process import smart_resize
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms.functional import resize

        then = time.monotonic()
        segments = self.segments[start : start + size]
        all_indices, clip_indices, requested_times = [], [], []
        for left, right in segments:
            count = max(2, int(round((right - left) * 2)))
            requested = np.linspace(left, right, count, endpoint=False)
            indices = np.searchsorted(self.timestamps[:, 0], requested, side="right") - 1
            indices = np.clip(indices, 0, len(self.reader) - 1)
            if np.any(self.timestamps[indices, 1] <= left):
                raise ValueError("Sample falls outside retained window")
            clip_indices.append(indices.tolist())
            requested_times.append(requested.tolist())
            all_indices.extend(indices.tolist())
        frames = self.reader.get_batch(all_indices).asnumpy()
        height, width = frames.shape[1:3]
        rh, rw = smart_resize(height, width, factor=32, min_pixels=131072, max_pixels=786432)
        clips, offset = [], 0
        for segment, indices in zip(segments, clip_indices):
            count = len(indices)
            pixels = torch.from_numpy(frames[offset : offset + count].copy()).permute(0, 3, 1, 2)
            pixels = resize(pixels, [rh, rw], interpolation=InterpolationMode.BICUBIC, antialias=True)
            video = pixels.permute(0, 2, 3, 1).contiguous().numpy()
            clips.append(
                {
                    "video": video,
                    "metadata": {
                        "fps": count / (segment[1] - segment[0]),
                        "duration": segment[1] - segment[0],
                        "total_num_frames": count,
                        "frames_indices": list(range(count)),
                        "video_backend": "decord_presampled",
                        "do_sample_frames": False,
                    },
                    "rgb_sha256": hashlib.sha256(video.tobytes()).hexdigest(),
                }
            )
            offset += count
        trace = {
            "frame_indices": clip_indices,
            "requested_times": requested_times,
            "frame_timestamps": [self.timestamps[indices].tolist() for indices in clip_indices],
            "original_height_width": [height, width],
            "resized_height_width": [rh, rw],
            "sampled_original_rgb_sha256": hashlib.sha256(frames.tobytes()).hexdigest(),
            "model_rgb_sha256": [clip["rgb_sha256"] for clip in clips],
            "model_input_frames": len(all_indices),
            "decode_preprocess_seconds": time.monotonic() - then,
        }
        return clips, trace
