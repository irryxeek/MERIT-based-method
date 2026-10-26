"""Lossless, self-contained source encoding for the frozen memory experiment.

The entire returned object, including its legend and shared tables, is model input
and must be tokenized by the caller. This module never loads external evidence
or chooses which entries to retain. An external receipt authenticates the full
encoded input; structural validation alone cannot detect a coherent rewrite.
"""

import json
import re
from copy import deepcopy

from evidence_grounding import interval, validate_source
from io_utils import digest

VERSION = "source_tables_v1"
LEGEND = (
    "Zero-based table indices. items=[text,[[fact_id,source_index],...]]. "
    "sources=[candidate,kind,frame_set_index] or "
    "[candidate,kind,frame_set_index,target_index,state_index]. "
    "frame_sets contain source frame IDs; pts maps IDs to exact seconds [start,end). "
    "targets=[anchor_candidate,target_interval]; states=[scope,event_assignment,semantic_verification]. "
    "candidate=null forbids direct candidate attribution. Time overlap does not verify event identity or text."
)
FIELDS = {"format", "legend", "items", "sources", "frame_sets", "pts", "targets", "states"}
KINDS = {"event", "phase", "outcome", "reaction", "identity", "context", "unknown"}
SCOPES = {"inside", "outside", "boundary", "mixed", "unknown"}
SOURCE_FIELDS = {"candidate", "frames", "kind"}
GROUNDING_FIELDS = {
    "anchor_candidate",
    "target_interval",
    "frame_pts",
    "scope",
    "event_assignment",
    "semantic_verification",
}
FACT_ID = re.compile(r"(?:c(?P<old>\d+)_(?:f|u)\d+|obs\d+_c(?P<observed>\d+)_f\d+)")
FRAME_ID = re.compile(r"s\d+")


def _json(value, *, sort_keys=False):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False, sort_keys=sort_keys)


def _fields(value, fields, label):
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"Unexpected {label} fields")


def _candidate(value, *, nullable=False):
    if nullable and value is None:
        return
    if type(value) is not int or value < 0:
        raise ValueError("Candidate must be a nonnegative integer")


def _interval(value):
    if not isinstance(value, list):
        raise ValueError("Source interval must be a JSON array")
    interval(value)  # Validate without normalizing/rounding the original numbers.


def _frames(value):
    if not isinstance(value, list) or any(not isinstance(f, str) or not FRAME_ID.fullmatch(f) for f in value):
        raise ValueError("Invalid source frame IDs")
    if len(set(value)) != len(value):
        raise ValueError("Duplicate source frame IDs")


def _validate_items(items):
    if not isinstance(items, list):
        raise ValueError("Memory items must be a list")
    seen, known_pts, known_targets = set(), {}, {}
    for item in items:
        _fields(item, {"text", "fact_ids", "sources"}, "item")
        if not isinstance(item["text"], str):
            raise ValueError("Memory text must be a string")
        ids = item["fact_ids"]
        if not isinstance(ids, list) or not ids or any(not isinstance(f, str) or not FACT_ID.fullmatch(f) for f in ids):
            raise ValueError("Invalid ReVALUE fact IDs")
        if len(set(ids)) != len(ids) or seen.intersection(ids):
            raise ValueError("Duplicate fact IDs")
        seen.update(ids)
        _fields(item["sources"], set(ids), "source mapping")
        for identity in ids:
            source = item["sources"][identity]
            if not isinstance(source, dict):
                raise ValueError("Source must be an object")
            grounded = "grounding" in source
            _fields(source, SOURCE_FIELDS | ({"grounding"} if grounded else set()), "source")
            _candidate(source["candidate"], nullable=grounded)
            if not isinstance(source["kind"], str) or source["kind"] not in KINDS:
                raise ValueError("Invalid source kind")
            _frames(source["frames"])
            if source["kind"] != "unknown" and not source["frames"]:
                raise ValueError("Visible facts require source frames")
            match = FACT_ID.fullmatch(identity)
            anchor = int(match["old"] if match["old"] is not None else match["observed"])
            if match["observed"] is not None and not grounded:
                raise ValueError("Observation source cannot lose temporal grounding")
            if grounded:
                info = source["grounding"]
                _fields(info, GROUNDING_FIELDS, "grounding")
                _candidate(info["anchor_candidate"])
                if info["anchor_candidate"] != anchor:
                    raise ValueError("Fact ID and source anchor disagree")
                _interval(info["target_interval"])
                target_key = _json(info["target_interval"])
                if known_targets.setdefault(anchor, target_key) != target_key:
                    raise ValueError("Candidate has conflicting target intervals")
                if not isinstance(info["scope"], str) or info["scope"] not in SCOPES:
                    raise ValueError("Invalid temporal scope")
                if info["event_assignment"] != "unverified":
                    raise ValueError("Invalid temporal or event status")
                if info["semantic_verification"] != "unreviewed":
                    raise ValueError("Codec cannot invent semantic verification")
                _fields(info["frame_pts"], set(source["frames"]), "frame/time mapping")
                for frame, pts in info["frame_pts"].items():
                    _interval(pts)
                    pts_key = _json(pts)
                    if known_pts.setdefault(frame, pts_key) != pts_key:
                        raise ValueError("Source frame has conflicting PTS")
            elif source["candidate"] != anchor:
                raise ValueError("Fact ID and source candidate disagree")
            validate_source(source)
    _json(items)  # Reject values outside finite JSON, including unsupported types.


def _intern(table, value):
    key = _json(value)
    for index, existing in enumerate(table):
        if _json(existing) == key:
            return index
    table.append(deepcopy(value))
    return len(table) - 1


def _add(encoded, item):
    pairs = []
    for identity in item["fact_ids"]:
        source = item["sources"][identity]
        record = [source["candidate"], source["kind"], _intern(encoded["frame_sets"], source["frames"])]
        if "grounding" in source:
            info = source["grounding"]
            record.extend(
                [
                    _intern(encoded["targets"], [info["anchor_candidate"], info["target_interval"]]),
                    _intern(
                        encoded["states"], [info["scope"], info["event_assignment"], info["semantic_verification"]]
                    ),
                ]
            )
            for frame, pts in info["frame_pts"].items():
                encoded["pts"].setdefault(frame, deepcopy(pts))
        pairs.append([identity, _intern(encoded["sources"], record)])
    encoded["items"].append([item["text"], pairs])


def encode(items):
    """Return deterministic visible JSON; preserve entries, IDs, text and PTS."""
    _validate_items(items)
    encoded = {
        "format": VERSION,
        "legend": LEGEND,
        "items": [],
        "sources": [],
        "frame_sets": [],
        "pts": {},
        "targets": [],
        "states": [],
    }
    for item in items:
        _add(encoded, item)
    return encoded


def _at(table, index):
    if type(index) is not int or not 0 <= index < len(table):
        raise ValueError("Invalid source table index")
    return table[index]


def _row(value, lengths, label):
    if not isinstance(value, list) or len(value) not in lengths:
        raise ValueError(f"Invalid {label} row")


def decode(encoded, *, expected_sha256=None):
    """Validate the complete visible object and reconstruct original items.

    Pass a trusted external receipt to detect arbitrary changes, including PTS
    changes that do not cross a temporal boundary. Without it this only checks
    structure, canonicalization and source consistency, not authenticity/text.
    """
    _fields(encoded, FIELDS, "codec")
    if expected_sha256 is not None and digest(encoded) != expected_sha256:
        raise ValueError("Memory receipt mismatch")
    if encoded["format"] != VERSION or encoded["legend"] != LEGEND:
        raise ValueError("Unknown codec format or altered legend")
    for name in ("items", "sources", "frame_sets", "targets", "states"):
        if not isinstance(encoded[name], list):
            raise ValueError(f"Invalid {name} table")
    if not isinstance(encoded["pts"], dict):
        raise ValueError("Invalid PTS table")
    for frames in encoded["frame_sets"]:
        _frames(frames)
    for frame, pts in encoded["pts"].items():
        _frames([frame])
        _interval(pts)
    for target in encoded["targets"]:
        _row(target, {2}, "target")
        _candidate(target[0])
        _interval(target[1])
    for state in encoded["states"]:
        _row(state, {3}, "state")
        if not isinstance(state[0], str) or state[0] not in SCOPES or state[1:] != ["unverified", "unreviewed"]:
            raise ValueError("Invalid grounding state")
    sources = []
    for record in encoded["sources"]:
        _row(record, {3, 5}, "source")
        candidate, kind, frames_index = record[:3]
        frames = deepcopy(_at(encoded["frame_sets"], frames_index))
        source = {"candidate": candidate, "frames": frames, "kind": kind}
        if len(record) == 5:
            anchor, target = _at(encoded["targets"], record[3])
            scope, event, semantic = _at(encoded["states"], record[4])
            if not set(frames) <= set(encoded["pts"]):
                raise ValueError("Missing source-frame PTS")
            source["grounding"] = {
                "anchor_candidate": anchor,
                "target_interval": deepcopy(target),
                "frame_pts": {f: deepcopy(encoded["pts"][f]) for f in frames},
                "scope": scope,
                "event_assignment": event,
                "semantic_verification": semantic,
            }
        sources.append(source)
    items = []
    for row in encoded["items"]:
        _row(row, {2}, "item")
        text, pairs = row
        if not isinstance(pairs, list):
            raise ValueError("Invalid item fact/source pairs")
        ids, mapped = [], {}
        for pair in pairs:
            _row(pair, {2}, "fact/source")
            identity, index = pair
            if not isinstance(identity, str):
                raise ValueError("Invalid fact ID")
            ids.append(identity)
            mapped[identity] = deepcopy(_at(sources, index))
        items.append({"text": text, "fact_ids": ids, "sources": mapped})
    _validate_items(items)
    # Reject unused/duplicate table entries or hidden payloads. Encoding is
    # first-use ordered, so exact canonicalization also preserves append indices.
    if _json(encoded, sort_keys=True) != _json(encode(items), sort_keys=True):
        raise ValueError("Noncanonical source tables")
    return items


def append_entry(encoded, item, *, expected_sha256=None):
    """Return an add-only encoding; the caller decides whether its tokens fit."""
    items = decode(encoded, expected_sha256=expected_sha256)
    combined = items + [deepcopy(item)]
    _validate_items(combined)
    result = deepcopy(encoded)
    _add(result, item)
    return result
