"""Shared, label-free inference utilities for QVHighlights baselines."""

import base64
import hashlib
import json
import math
import re
import subprocess
import threading
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image
from qwen_vl_utils import fetch_video
from torchvision.transforms.functional import to_pil_image

SYSTEM_PROMPT = (
    "You are a video highlight assessor. Follow the instruction for the current stage exactly. "
    "Scores must be based only on the provided video, query, and tool observations."
)

CROP_VIDEO_TOOL = {
    "type": "function",
    "function": {
        "name": "crop_video",
        "description": "Inspect a local time interval of the current video more closely.",
        "parameters": {
            "type": "object",
            "properties": {
                "video_path": {"type": "string", "description": "Current video path"},
                "start_time": {"type": "number", "description": "Local start time in seconds"},
                "end_time": {"type": "number", "description": "Local end time in seconds"},
            },
            "required": ["video_path", "start_time", "end_time"],
        },
    },
}


def media_duration(path: str) -> float:
    command = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", path]
    data = json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)
    return float(data["format"]["duration"])


def _to_image_contents(video: torch.Tensor) -> list[dict]:
    video = video.to(torch.uint8)
    contents = []
    for frame in video:
        image: Image.Image = to_pil_image(frame)
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        contents.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}})
    return contents


def image_content_sha256(contents: list[dict]) -> list[str]:
    hashes = []
    for content in contents:
        encoded = content["image_url"]["url"].split(",", 1)[1]
        hashes.append(hashlib.sha256(base64.b64decode(encoded)).hexdigest())
    return hashes


def save_image_contents(contents: list[dict], output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, content in enumerate(contents):
        encoded = content["image_url"]["url"].split(",", 1)[1]
        path = output_dir / f"frame_{index:03d}.png"
        path.write_bytes(base64.b64decode(encoded))
        paths.append(str(path.resolve()))
    return paths


def encode_global_video(video_path: str, nframes: int, max_pixels: int) -> tuple[list[dict], list[float]]:
    element = {
        "type": "video", "video": f"file://{video_path}", "nframes": nframes,
        "min_pixels": 28 * 28, "max_pixels": max_pixels,
    }
    video, metadata = fetch_video(element, return_video_metadata=True)
    fps = float(metadata["fps"])
    timestamps = [float(index) / fps for index in metadata["frames_indices"]]
    return _to_image_contents(video), timestamps


def encode_crop(
    video_path: str, start_time: float, end_time: float, max_frames: int, max_pixels: int
) -> tuple[list[dict], list[float], dict]:
    duration = media_duration(video_path)
    if not all(math.isfinite(value) for value in (start_time, end_time)):
        raise ValueError("Crop times must be finite")
    executed_start = max(0.0, float(start_time))
    executed_end = min(duration, float(end_time))
    if executed_start >= executed_end or executed_start >= duration:
        raise ValueError(f"Invalid crop interval {start_time}-{end_time} for duration {duration}")
    element = {
        "type": "video", "video": f"file://{video_path}", "fps": 1,
        "min_frames": 2, "max_frames": max_frames,
        "min_pixels": 28 * 28, "max_pixels": max_pixels,
        "video_start": executed_start, "video_end": executed_end,
    }
    video, metadata = fetch_video(element, return_video_metadata=True)
    fps = float(metadata["fps"])
    timestamps = [float(index) / fps for index in metadata["frames_indices"]]
    return _to_image_contents(video), timestamps, {
        "duration": duration, "executed_start_time": executed_start, "executed_end_time": executed_end,
    }


def format_timestamps(timestamps: list[float]) -> str:
    return ", ".join(f"{value:.3f}" for value in timestamps)


def build_prompt(
    sample: dict, tool_enabled: bool, global_timestamps: list[float], max_tool_calls: int = 2
) -> str:
    n = sample["num_clips"]
    duration = sample["duration"]
    tool_text = (
        "You may autonomously call crop_video when closer inspection is needed. Do not call it unless useful; "
        f"at most {max_tool_calls} calls are available. A crop is only an observation and never directly "
        "determines scores. "
        if tool_enabled else "No tools are available; score from the uniform global observations. "
    )
    return (
        f'Query: {sample["query"]}\n'
        f"Video local duration: {duration:.3f} seconds. Video path: {sample['video_path']}\n"
        f"The global images are chronological and correspond to local timestamps (seconds): "
        f"[{format_timestamps(global_timestamps)}].\n"
        f"Divide the local video timeline into exactly {n} non-overlapping 2-second clips. "
        f"Clip i covers [2*i, min(2*(i+1), {duration:.3f})) seconds for i=0..{n - 1}. "
        "Assign every clip a query-conditioned highlight saliency score from 0 to 4: "
        "0 is irrelevant, 1 weakly related, 2 relevant but ordinary, 3 strong representative highlight, "
        "and 4 the most salient and representative evidence. Use graded scores, considering both relevance "
        "and prominence; binary relevance alone is insufficient. Score the complete timeline, including clips "
        "outside any inspected crop. " + tool_text +
        "This is the evidence-analysis stage. Concisely identify the relevant and strongest time intervals, "
        "including weaker or irrelevant parts needed to distinguish score levels. Do not output JSON or the "
        "final score vector yet. When enough evidence has been gathered, return only a concise temporal "
        "assessment for the later scoring stage."
    )


def build_finalization_prompt(sample: dict, max_override_segments: int = 5) -> str:
    n = sample["num_clips"]
    duration = sample["duration"]
    example_end = min(4.0, duration)
    return (
        "Now convert the temporal assessment into scores for the complete timeline. Predict one integer "
        "background_score from 0 to 4 for every clip not covered by an override. Add sorted, non-overlapping "
        f"override segments as half-open local-time ranges in seconds within [0,{duration:.3f}]; start_time is "
        "inclusive, end_time is exclusive, and score is an integer from 0 to 4. Do not use clip indices. "
        f"The adapter maps these ranges to all {n} official 2-second clips by clip midpoint. The background "
        "score is a model prediction for all other clips, "
        "not an automatic zero for uninspected regions. Use graded scores based on relevance and prominence. "
        f"Use at most {max_override_segments} overrides and merge adjacent ranges with the same score. "
        'Respond strictly as JSON, for example: {"background_score":0,"segments":'
        f'[{{"start_time":0,"end_time":{example_end:.1f},"score":4}}]}}. '
        "Use exactly these keys and no prose. If ranges overlap, their maximum score applies."
    )


def parse_scores(
    text: str,
    expected: int,
    score_min: float,
    score_max: float,
    max_override_segments: int = 5,
    duration: float | None = None,
    clip_seconds: float = 2.0,
    cleanup_actions: list[str] | None = None,
) -> list[float]:
    cleanup_actions = cleanup_actions if cleanup_actions is not None else []
    matches = re.findall(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    payload = matches[-1].strip() if matches else text.strip()
    if payload.startswith("```") and payload.endswith("```"):
        payload = re.sub(r"^```(?:json)?\s*", "", payload, flags=re.IGNORECASE)
        payload = re.sub(r"\s*```$", "", payload)
        cleanup_actions.append("removed_markdown_fence")
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        if (
            payload.startswith("{")
            and payload.endswith("]")
            and payload.count("{") == payload.count("}") + 1
            and payload.count("[") == payload.count("]")
        ):
            payload += "}"
            cleanup_actions.append("appended_single_missing_closing_object_brace")
            data = json.loads(payload)
        else:
            raise
    if not isinstance(data, dict) or set(data) != {"background_score", "segments"}:
        raise ValueError("invalid_schema")
    background = data["background_score"]
    segments = data["segments"]
    if isinstance(background, bool) or not isinstance(background, int):
        raise ValueError("invalid_background_score")
    if background < score_min or background > score_max:
        raise ValueError(f"score_out_of_range_{background}")
    if not isinstance(segments, list):
        raise ValueError("invalid_segments")
    if len(segments) > max_override_segments:
        raise ValueError(f"too_many_segments_{len(segments)}_maximum_{max_override_segments}")
    if duration is None:
        duration = expected * clip_seconds
    scores = [float(background)] * expected
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict) or set(segment) != {"start_time", "end_time", "score"}:
            raise ValueError(f"invalid_segment_schema_{index}")
        start, end, score = segment["start_time"], segment["end_time"], segment["score"]
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in (start, end)):
            raise ValueError(f"invalid_segment_type_{index}")
        if isinstance(score, bool) or not isinstance(score, int):
            raise ValueError(f"invalid_segment_type_{index}")
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or start >= end or end > duration:
            raise ValueError(f"invalid_segment_bounds_{index}")
        if score < score_min or score > score_max:
            raise ValueError(f"score_out_of_range_{score}")
        for clip_index in range(expected):
            clip_end = min((clip_index + 1) * clip_seconds, duration)
            midpoint = (clip_index * clip_seconds + clip_end) / 2
            if start <= midpoint < end:
                scores[clip_index] = max(scores[clip_index], float(score))
    return scores


def response_record(response) -> dict:
    choice = response.choices[0]
    return {
        "id": response.id, "model": response.model,
        "usage": response.usage.model_dump() if response.usage else None,
        "choice": choice.model_dump(),
        "message": choice.message.model_dump(),
        "finish_reason": choice.finish_reason,
        "stop_reason": getattr(choice, "stop_reason", None),
    }


class GpuMemoryMonitor:
    def __init__(self, gpu_indices=(6, 7), interval=0.5):
        self.gpu_indices = tuple(gpu_indices)
        self.interval = interval
        self.maximum = {str(index): 0 for index in self.gpu_indices}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                output = subprocess.run(
                    ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                    check=True, capture_output=True, text=True,
                ).stdout
                for line in output.splitlines():
                    index, used = [part.strip() for part in line.split(",")]
                    if int(index) in self.gpu_indices:
                        self.maximum[index] = max(self.maximum[index], int(used))
            except Exception:
                pass
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join(timeout=2)
