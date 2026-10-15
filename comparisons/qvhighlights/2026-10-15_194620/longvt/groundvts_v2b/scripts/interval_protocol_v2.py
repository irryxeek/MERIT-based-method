"""Regex-constrained 2-second clip-span protocol."""

from __future__ import annotations

import json
import math
import re
from typing import Any


LEVEL_TO_SCORE = {
    "very_important": 4.0,
    "important": 3.0,
    "less_important": 2.0,
}


def valid_clip_spans(num_clips: int) -> list[str]:
    return [f"{start}:{end}" for start in range(num_clips) for end in range(start + 1, num_clips + 1)]


def _integer_range_regex(low: int, high: int) -> str:
    """Compact regular expression for a small positive integer range."""
    parts = []
    for tens in range(low // 10, high // 10 + 1):
        lo = max(low, tens * 10) - tens * 10
        hi = min(high, tens * 10 + 9) - tens * 10
        if tens == 0:
            prefix = ""
            if lo == hi:
                atom = str(lo)
            elif lo == 0:
                atom = f"[0-{hi}]"
            elif lo == 1 and hi == 9:
                atom = "[1-9]"
            else:
                atom = f"[{lo}-{hi}]"
        else:
            prefix = str(tens)
            if lo == hi:
                atom = str(lo)
            elif hi - lo == 1:
                atom = f"[{lo}{hi}]"
            else:
                atom = f"[{lo}-{hi}]"
        parts.append(prefix + atom)
    return parts[0] if len(parts) == 1 else "(?:" + "|".join(parts) + ")"


def valid_clip_span_pattern(num_clips: int) -> str:
    alternatives = [
        f"{start}:{_integer_range_regex(start + 1, num_clips)}"
        for start in range(num_clips)
    ]
    return "^(?:" + "|".join(alternatives) + ")$"


def interval_response_format(duration: float) -> dict[str, Any]:
    """Constrain the endpoint relation with a regex matching legal half-open clip spans."""
    num_clips = math.floor(duration / 2.0)
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "qvhighlights_clip_spans",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "segments": {
                        "type": "array",
                        "maxItems": num_clips,
                        "items": {
                            "type": "object",
                            "properties": {
                                "clip_span": {"type": "string", "pattern": valid_clip_span_pattern(num_clips)},
                                "level": {"type": "string", "enum": list(LEVEL_TO_SCORE)},
                            },
                            "required": ["clip_span", "level"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["segments"],
                "additionalProperties": False,
            },
        },
    }


def parse_intervals(text: str, duration: float, num_clips: int) -> tuple[list[dict[str, Any]], bool]:
    """Parse valid enumerated spans and sort them losslessly by their first clip."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("empty_response")
    payload = text.strip()
    if payload.startswith("```"):
        match = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\n?```", payload, flags=re.DOTALL | re.IGNORECASE)
        if not match:
            raise ValueError("invalid_markdown_fence")
        payload = match.group(1).strip()
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid_json:{exc.msg}") from exc
    if not isinstance(data, dict) or set(data) != {"segments"}:
        raise ValueError("invalid_top_level_schema")
    segments = data["segments"]
    if not isinstance(segments, list):
        raise ValueError("segments_not_array")
    if len(segments) > num_clips:
        raise ValueError("too_many_segments")
    legal_spans = set(valid_clip_spans(num_clips))
    clean: list[dict[str, Any]] = []
    for i, segment in enumerate(segments):
        if not isinstance(segment, dict) or set(segment) != {"clip_span", "level"}:
            raise ValueError(f"invalid_segment_schema:{i}")
        span, level = segment["clip_span"], segment["level"]
        if not isinstance(span, str) or span not in legal_spans:
            raise ValueError(f"invalid_clip_span:{i}")
        if not isinstance(level, str) or level not in LEVEL_TO_SCORE:
            raise ValueError(f"invalid_level:{i}")
        start, end = map(int, span.split(":"))
        clean.append({"clip_span": span, "start_clip": start, "end_clip_exclusive": end, "level": level})
    sorted_clean = sorted(clean, key=lambda item: (item["start_clip"], item["end_clip_exclusive"], item["level"]))
    order_canonicalized = clean != sorted_clean
    return sorted_clean, order_canonicalized


def intervals_to_scores(
    segments: list[dict[str, Any]], duration: float, num_clips: int, clip_seconds: float = 2.0
) -> list[float]:
    expected = math.floor(duration / clip_seconds)
    if clip_seconds <= 0 or duration <= 0 or num_clips != expected:
        raise ValueError("official_clip_count_mismatch")
    scores = [0.0] * num_clips
    for segment in segments:
        start, end = segment["start_clip"], segment["end_clip_exclusive"]
        if start < 0 or start >= end or end > num_clips:
            raise ValueError("invalid_clip_span_after_parse")
        value = LEVEL_TO_SCORE[segment["level"]]
        for index in range(start, end):
            scores[index] = max(scores[index], value)
    return scores


def clamp_crop_to_official_duration(start_time: float, end_time: float, duration: float) -> tuple[float, float, bool]:
    """Reject invalid crop starts and clamp crop end to the annotated local clip."""
    if not all(math.isfinite(value) for value in (start_time, end_time, duration)) or duration <= 0:
        raise ValueError("non_finite_crop_time_or_invalid_duration")
    if start_time < 0 or start_time >= duration:
        raise ValueError("crop_start_outside_official_local_duration")
    if end_time <= start_time:
        raise ValueError("invalid_crop_interval")
    return start_time, min(end_time, duration), end_time > duration


def eligible_frame_indices(timestamps: list[float], duration: float) -> list[int]:
    return [index for index, stamp in enumerate(timestamps) if math.isfinite(stamp) and 0 <= stamp < duration]


def validate_qid_coverage(rows: list[dict[str, Any]], expected_qids: list[Any]) -> None:
    qids = [row["qid"] for row in rows]
    if len(qids) != len(set(qids)) or set(qids) != set(expected_qids):
        raise ValueError("qid_coverage_mismatch")


def completed_qids(traces: list[dict[str, Any]]) -> set[Any]:
    transient = {"timeout", "request_failure", "oom", "context_overflow"}
    return {row["qid"] for row in traces if row.get("status") not in transient}


def append_tool_observation(messages: list[dict[str, Any]], call_id: str, content: list[dict[str, Any]]) -> None:
    messages.append({"role": "tool", "tool_call_id": call_id, "content": content})
