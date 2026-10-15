from __future__ import annotations

import base64
import bisect
import io
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from .common import atomic_json, digest, file_hash


def metadata(path):
    p = subprocess.run(
        [
            "ffprobe",
            "-threads",
            "1",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=duration,start_time,avg_frame_rate,nb_frames:format=duration,start_time",
            "-of",
            "json",
            str(path),
        ],
        text=True,
        capture_output=True,
        timeout=60,
    )
    if p.returncode:
        raise ValueError("ffprobe failure: " + p.stderr[:300])
    data = json.loads(p.stdout)
    stream = data["streams"][0]
    return {
        "video_duration": float(stream.get("duration", data["format"]["duration"])),
        "container_duration": float(data["format"]["duration"]),
        "origin": float(data["format"].get("start_time", 0)),
        "stream": stream,
    }


def probe(path, annotated_duration, cache_root):
    path = Path(path)
    stat = path.stat()
    cache = Path(cache_root) / (
        digest([str(path.resolve()), stat.st_size, stat.st_mtime_ns]) + ".json"
    )
    if cache.exists():
        result = json.loads(cache.read_text())
    else:
        result = metadata(path)
        p = subprocess.run(
            [
                "ffprobe",
                "-threads",
                "1",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_frames",
                "-show_entries",
                "frame=best_effort_timestamp_time",
                "-of",
                "json",
                str(path),
            ],
            text=True,
            capture_output=True,
            timeout=300,
        )
        if p.returncode or p.stderr.strip():
            raise ValueError("frame probe decode errors: " + p.stderr[:300])
        result["pts"] = [
            float(f["best_effort_timestamp_time"]) - result["origin"]
            for f in json.loads(p.stdout)["frames"]
        ]
        if not result["pts"] or any(
            b <= a for a, b in zip(result["pts"], result["pts"][1:])
        ):
            raise ValueError("invalid or nonmonotonic PTS")
        result["media_sha256"] = file_hash(path)
        result["timeline_sha256"] = digest(result["pts"])
        result["vfr"] = (
            len({round(b - a, 5) for a, b in zip(result["pts"], result["pts"][1:])}) > 1
        )
        reader = FrameReader(path, result)
        try:
            reader.at(result["pts"][0])
            reader.at(result["pts"][-1])
        finally:
            reader.close()
        result["first_last_decode"] = True
        atomic_json(cache, result)
    gap = annotated_duration - result["video_duration"]
    # Dataset duration is integral and occasionally extends over a partial last clip.
    # Accept identity mapping only if every official clip has a distinct real PTS.
    # Do not alter the media, labels or number of clips, and report the partial tail.
    if not -1 <= gap < 2 or result["pts"][0] < -0.001:
        raise ValueError(
            f"unresolved mapping: media={result['video_duration']} annotation={annotated_duration}"
        )
    for i in range(int(annotated_duration / 2)):
        at = bisect.bisect_left(result["pts"], 2 * i)
        if at >= len(result["pts"]) or result["pts"][at] >= 2 * (i + 1):
            raise ValueError(
                f"official clip {i} lacks a real frame; no padding permitted"
            )
    result["mapping_audit"] = {
        "policy": "identity_pts_with_real_frame_in_every_official_clip_v1",
        "duration_gap_seconds": gap,
        "partially_available_tail": gap > 0.25,
        "media_excess_seconds": max(0, -gap),
        "offset_applied": 0,
        "gt_or_media_modified": False,
    }
    return result


class FrameReader:
    def __init__(self, path, info):
        import av

        self.container = av.open(str(path))
        self.stream = self.container.streams.video[0]
        self.stream.thread_type = "NONE"
        self.info = info

    def close(self):
        self.container.close()

    def at(self, relative_pts):
        absolute = relative_pts + self.info["origin"]
        self.container.seek(
            int(absolute / float(self.stream.time_base)),
            stream=self.stream,
            backward=True,
        )
        for frame in self.container.decode(self.stream):
            if frame.pts is None:
                continue
            t = float(frame.pts * self.stream.time_base) - self.info["origin"]
            if abs(t - relative_pts) < 0.001:
                return frame.to_image().convert("RGB")
            if t > relative_pts + 0.001:
                break
        raise ValueError(f"cannot decode exact PTS {relative_pts}")

    def sample(self, start, end, count=24, *, clip_ids=None, output_dir=None):
        pts = self.info["pts"]
        valid = pts[bisect.bisect_left(pts, start) : bisect.bisect_left(pts, end)]
        if not valid:
            raise ValueError("requested interval has no valid frames")
        observed_start = max(start, valid[0])
        observed_end = min(end, self.info["video_duration"])
        count = min(count, len(valid))
        targets = (
            [2 * i + 1 for i in clip_ids]
            if clip_ids is not None
            else np.linspace(observed_start, observed_end, count, endpoint=False)
            + (observed_end - observed_start) / (2 * count)
        )
        output = []
        for k, target in enumerate(targets):
            candidates = valid
            if clip_ids is not None:
                candidates = [
                    t for t in valid if 2 * clip_ids[k] <= t < 2 * (clip_ids[k] + 1)
                ]
                if not candidates:
                    raise ValueError(
                        f"clip {clip_ids[k]} has no frame; no padding permitted"
                    )
            t = min(candidates, key=lambda t: abs(t - target))
            image = self.at(t)
            image.thumbnail((672, 672), Image.Resampling.LANCZOS)
            buff = io.BytesIO()
            image.save(buff, format="JPEG", quality=90)
            data = buff.getvalue()
            image_id = digest([self.info["media_sha256"], t])
            image_path = None
            if output_dir:
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                image_path = str(Path(output_dir) / f"{image_id}.jpg")
                Path(image_path).write_bytes(data)
            output.append(
                {
                    "image_id": image_id,
                    "pts": t,
                    "position": k,
                    "clip_id": clip_ids[k] if clip_ids is not None else None,
                    "path": image_path,
                    "jpeg_sha256": digest(base64.b64encode(data).decode()),
                    "url": "data:image/jpeg;base64," + base64.b64encode(data).decode(),
                }
            )
        return output


def mapping(frames):
    return [{k: v for k, v in f.items() if k != "url"} for f in frames]


def visual_message(text, frames):
    content = [{"type": "text", "text": text}]
    for f in frames:
        caption = (
            f"Picture position {f['position']}; segment-relative PTS={f['pts']:.6f}s"
        )
        if f["clip_id"] is not None:
            caption += f"; clip_id={f['clip_id']}; interval=[{2 * f['clip_id']},{2 * (f['clip_id'] + 1)})s"
        content.extend(
            [
                {"type": "text", "text": caption},
                {"type": "image_url", "image_url": {"url": f["url"]}},
            ]
        )
    return {"role": "user", "content": content}
