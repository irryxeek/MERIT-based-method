"""Pure-Python interval output validation and QVHighlights score mapping."""

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


def interval_response_format(duration: float) -> dict[str, Any]:
    """Build a strict JSON schema with a duration-dependent segment bound."""
    max_items = math.floor(duration / 2.0)
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "qvhighlights_intervals",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "segments": {
                        "type": "array",
                        "maxItems": max_items,
                        "items": {
                            "type": "object",
                            "properties": {
                                "start": {"type": "number", "minimum": 0, "maximum": duration},
                                "end": {"type": "number", "minimum": 0, "maximum": duration},
                                "level": {"type": "string", "enum": list(LEVEL_TO_SCORE)},
                            },
                            "required": ["start", "end", "level"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["segments"],
                "additionalProperties": False,
            },
        },
    }


def parse_intervals(text: str, duration: float, num_clips: int) -> list[dict[str, Any]]:
    """Parse and semantically validate a strict interval object."""
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

    clean: list[dict[str, Any]] = []
    previous_start = -math.inf
    for i, segment in enumerate(segments):
        if not isinstance(segment, dict) or set(segment) != {"start", "end", "level"}:
            raise ValueError(f"invalid_segment_schema:{i}")
        start, end, level = segment["start"], segment["end"], segment["level"]
        if isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            raise ValueError(f"invalid_time_type:{i}")
        if not math.isfinite(start) or not math.isfinite(end):
            raise ValueError(f"non_finite_time:{i}")
        if start < 0 or start >= end or end > duration:
            raise ValueError(f"invalid_time_bounds:{i}")
        if not isinstance(level, str) or level not in LEVEL_TO_SCORE:
            raise ValueError(f"invalid_level:{i}")
        if start < previous_start:
            raise ValueError("segments_not_sorted")
        previous_start = start
        clean.append({"start": float(start), "end": float(end), "level": level})
    return clean


def intervals_to_scores(
    segments: list[dict[str, Any]], duration: float, num_clips: int, clip_seconds: float = 2.0
) -> list[float]:
    """Map half-open ranges to clips with positive-length overlap; ties use max."""
    if clip_seconds <= 0 or duration <= 0:
        raise ValueError("duration_and_clip_length_must_be_positive")
    expected = math.floor(duration / clip_seconds)
    if num_clips != expected:
        raise ValueError(f"clip_count_mismatch:{num_clips}!={expected}")
    scores = [0.0] * num_clips
    for segment in segments:
        start, end = segment["start"], segment["end"]
        score = LEVEL_TO_SCORE[segment["level"]]
        first = max(0, math.floor(start / clip_seconds))
        stop = min(num_clips, math.ceil(end / clip_seconds))
        for clip_idx in range(first, stop):
            clip_start = clip_idx * clip_seconds
            clip_end = min((clip_idx + 1) * clip_seconds, duration)
            if max(start, clip_start) < min(end, clip_end):
                scores[clip_idx] = max(scores[clip_idx], score)
    return scores


def validate_qid_coverage(rows: list[dict[str, Any]], expected_qids: list[Any]) -> None:
    qids = [row["qid"] for row in rows]
    if len(qids) != len(set(qids)):
        raise ValueError("duplicate_qid")
    if set(qids) != set(expected_qids):
        missing = sorted(set(expected_qids) - set(qids))
        extra = sorted(set(qids) - set(expected_qids))
        raise ValueError(f"qid_set_mismatch:missing={missing[:5]}:extra={extra[:5]}")


def completed_qids(traces: list[dict[str, Any]]) -> set[Any]:
    """Return qids with a terminal prediction; transient service errors stay resumable."""
    transient = {"timeout", "request_failure", "oom", "context_overflow"}
    return {row["qid"] for row in traces if row.get("status") not in transient}


def append_tool_observation(messages: list[dict[str, Any]], call_id: str, content: list[dict[str, Any]]) -> None:
    """Append a tool response without dropping earlier visual or assistant context."""
    messages.append({"role": "tool", "tool_call_id": call_id, "content": content})
