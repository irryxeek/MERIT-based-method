"""Backend-owned temporal provenance; source binding is not semantic verification."""

import math
from copy import deepcopy

from schema import Facts


def interval(value):
    if len(value) != 2 or any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in value):
        raise ValueError("Expected two finite source times")
    start, end = map(float, value)
    if not math.isfinite(start + end) or not 0 <= start < end:
        raise ValueError("Invalid source interval")
    return [start, end]


def temporal_scope(frame_pts, target):
    """Half-open display intervals; boundary frames are not confidently inside."""
    start, end = interval(target)
    regions = []
    for value in frame_pts.values():
        left, right = interval(value)
        if start <= left and right <= end:
            regions.append("inside")
        elif right <= start or left >= end:
            regions.append("outside")
        else:
            regions.append("boundary")
    if not regions:
        return "unknown"
    return regions[0] if len(set(regions)) == 1 else "mixed"


def ground_observation(result, trace, candidate, target, namespace):
    """Never accept generated timestamps or infer event identity from an anchor."""
    values = Facts.model_validate(result).model_dump()
    target = interval(target)
    frames = {f["id"]: f for f in trace["frames"]}
    if len(frames) != len(trace["frames"]):
        raise ValueError("Duplicate source frame identities")
    for frame in frames.values():
        interval(frame["pts"])
        if frame["id"] != f"s{frame['frame_index']}":
            raise ValueError("Source ID/index mismatch")
    facts = values["facts"] + [{"kind": "unknown", "text": t, "frames": []} for t in values["unknown"]]
    output = {}
    for index, fact in enumerate(facts):
        refs = fact["frames"]
        if len(set(refs)) != len(refs) or not set(refs) <= set(frames):
            raise ValueError("Duplicate or unobserved source citation")
        pts = {ref: interval(frames[ref]["pts"]) for ref in refs}
        scope = temporal_scope(pts, target)
        identity = f"{namespace}_f{index}"
        output[identity] = {
            "id": identity,
            **fact,
            "candidate": candidate if scope == "inside" and fact["kind"] != "unknown" else None,
            "grounding": {
                "anchor_candidate": candidate,
                "target_interval": target,
                "frame_pts": pts,
                "scope": scope,
                "event_assignment": "unverified",
                "semantic_verification": "unreviewed",
            },
            "frame_receipts": {ref: deepcopy(frames[ref]) for ref in refs},
        }
    return output


def validate_source(source):
    """Reject corrupted scoped memory rather than silently restoring legacy attribution."""
    if "grounding" not in source:
        return
    info = source["grounding"]
    if set(info["frame_pts"]) != set(source["frames"]):
        raise ValueError("Source frame/time mapping mismatch")
    scope = temporal_scope(info["frame_pts"], info["target_interval"])
    if info["scope"] != scope:
        raise ValueError("Source scope contradicts its frame times")
    expected = info["anchor_candidate"] if scope == "inside" and source["kind"] != "unknown" else None
    if source["candidate"] != expected:
        raise ValueError("Context cannot be assigned to the candidate")
    if info["event_assignment"] != "unverified":
        raise ValueError("Temporal provenance cannot establish event identity")


def memory_source(fact):
    source = {k: deepcopy(fact[k]) for k in ("candidate", "frames", "kind")}
    if "grounding" in fact:
        source["grounding"] = deepcopy(fact["grounding"])
    validate_source(source)
    return source


def direct_source(source, candidate):
    validate_source(source)
    return source["candidate"] == candidate and source["kind"] != "unknown"


def source_anchor(source):
    return source.get("grounding", {}).get("anchor_candidate", source["candidate"])
