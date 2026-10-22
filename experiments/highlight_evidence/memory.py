"""Bounded, visually grounded memory extraction with configurable value facets."""

import json
import math

from merit_qvh.core import digest

SYSTEM = (
    "Use only supplied images and original timestamps. Return one compact JSON object. "
    "Record visible facts, not a reasoning transcript. Unseen actions are unknown. "
    "Temporal order does not prove causality. Do not infer audio, feelings, or offscreen outcomes."
)


def number(x):
    return type(x) in (int, float) and math.isfinite(x)


def text(x, limit):
    return isinstance(x, str) and 0 < len(x.strip()) <= limit


def check_values(values):
    if not isinstance(values, list) or not 1 <= len(values) <= 8:
        raise ValueError("Supply an explicit value vocabulary (1-8 dimensions)")
    ids = []
    for v in values:
        if set(v) != {"id", "definition", "visible_cues"}:
            raise ValueError("Each value needs id, definition and visible_cues")
        if not all(
            text(v[k], limit) for k, limit in (("id", 60), ("definition", 500), ("visible_cues", 500))
        ):
            raise ValueError("Invalid value definition")
        ids.append(v["id"])
    if len(set(ids)) != len(ids):
        raise ValueError("Repeated value ID")
    return ids


def frame_index(frames, duration):
    if not number(duration) or duration <= 0 or not frames:
        raise ValueError("Positive duration and real frames required")
    indexed = {}
    for f in frames:
        if not text(f.get("image_id"), 128) or not text(f.get("jpeg_sha256"), 128):
            raise ValueError("Frame identity and pixel hash required")
        if not number(f.get("pts")) or not 0 <= f["pts"] < duration:
            raise ValueError("Frame outside original video timeline")
        if f["image_id"] in indexed:
            raise ValueError("Duplicate frame identity")
        indexed[f["image_id"]] = f
    return indexed


def references(ids, indexed, *, required=True):
    if not isinstance(ids, list) or any(not isinstance(i, str) for i in ids):
        raise ValueError("Evidence IDs must be a list of strings")
    if len(set(ids)) != len(ids) or not set(ids) <= set(indexed) or (required and not ids):
        raise ValueError("Evidence must cite unique supplied frame IDs")


def validate_memory(payload, frames, values, duration):
    """Validate provenance/shape, not the semantic truth of model observations."""
    vocabulary = check_values(values)
    indexed = frame_index(frames, duration)
    if not isinstance(payload, dict) or set(payload) != {"events", "links"}:
        raise ValueError("Memory requires events and links")
    if not isinstance(payload["events"], list) or len(payload["events"]) > 4:
        raise ValueError("At most four event nodes per extraction")
    events = {}
    for e in payload["events"]:
        if not isinstance(e, dict) or set(e) != {
            "id",
            "start",
            "end",
            "description",
            "evidence_ids",
            "values",
        }:
            raise ValueError("Invalid event schema")
        if not text(e["id"], 80) or e["id"] in events or not text(e["description"], 240):
            raise ValueError("Invalid event identity or description")
        if not number(e["start"]) or not number(e["end"]) or not 0 <= e["start"] < e["end"] <= duration:
            raise ValueError("Invalid event bounds")
        references(e["evidence_ids"], indexed)
        if any(not e["start"] <= indexed[i]["pts"] < e["end"] for i in e["evidence_ids"]):
            raise ValueError("Cited event frame outside event bounds")
        if not isinstance(e["values"], list) or len(e["values"]) != len(vocabulary):
            raise ValueError("All configured facets need an explicit status")
        seen = []
        for v in e["values"]:
            if not isinstance(v, dict) or set(v) != {"dimension", "status", "level", "cue", "evidence_ids"}:
                raise ValueError("Invalid value observation")
            seen.append(v["dimension"])
            if v["status"] not in {"observed", "unknown", "not_applicable"} or not text(v["cue"], 160):
                raise ValueError("Invalid observation status or cue")
            observed = v["status"] == "observed"
            references(v["evidence_ids"], {i: indexed[i] for i in e["evidence_ids"]}, required=observed)
            if observed and (type(v["level"]) is not int or not 0 <= v["level"] <= 3):
                raise ValueError("Observed level must be 0-3, an ordinal descriptor, not probability")
            if not observed and v["level"] is not None:
                raise ValueError("Unknown/inapplicable is null, not zero intensity")
        if sorted(seen) != sorted(vocabulary):
            raise ValueError("Value vocabulary mismatch or duplicate dimension")
        events[e["id"]] = e
    if not isinstance(payload["links"], list) or len(payload["links"]) > 8:
        raise ValueError("At most eight links")
    links = set()
    for edge in payload["links"]:
        if not isinstance(edge, dict) or set(edge) != {"source", "target", "relation", "evidence_ids"}:
            raise ValueError("Invalid edge schema")
        a, b = edge["source"], edge["target"]
        if (
            a not in events
            or b not in events
            or a == b
            or edge["relation"] not in {"before", "possible_link"}
        ):
            raise ValueError("Invalid graph edge; causal claims are not supported")
        identity = (a, b, edge["relation"])
        if identity in links:
            raise ValueError("Duplicate edge")
        links.add(identity)
        allowed = set(events[a]["evidence_ids"]) | set(events[b]["evidence_ids"])
        references(edge["evidence_ids"], {i: indexed[i] for i in allowed})
        if not all(set(edge["evidence_ids"]) & set(events[e]["evidence_ids"]) for e in (a, b)):
            raise ValueError("An edge must cite both endpoint events")
        if edge["relation"] == "before" and events[a]["end"] > events[b]["start"]:
            raise ValueError("Temporal edge contradicts event bounds")
    return payload


def extract_memory(client, frames, values, duration, *, namespace):
    """A live-client hook, never invoked by the CPU demo. No query/labels input."""
    check_values(values)
    frame_index(frames, duration)
    if len(frames) > 24 or not text(namespace, 60):
        raise ValueError("At most 24 frames per extraction and an explicit namespace")
    schema = {
        "events": [
            {
                "id": "e0",
                "start": 0.0,
                "end": 2.0,
                "description": "brief visible event",
                "evidence_ids": ["supplied image_id"],
                "values": [
                    {
                        "dimension": v["id"],
                        "status": "unknown",
                        "level": None,
                        "cue": "not observable in these samples",
                        "evidence_ids": [],
                    }
                    for v in values
                ],
            }
        ],
        "links": [{"source": "e0", "target": "e1", "relation": "before", "evidence_ids": []}],
    }
    instruction = (
        "Build query-independent event memory, at most 4 events and 8 links. "
        "For every event provide every configured value dimension. Values: observed/unknown/not_applicable; "
        "observed levels 0 absent, 1 mild, 2 moderate, 3 strong, with a concrete visible cue and frame citations. "
        "A high level alone does not mean a query highlight. Keep description <=240 and cue <=160 characters. "
        "Event intervals bound cited frames and use original video seconds. Do not fill unsampled actions "
        "from common sense. before means nonoverlapping temporal order; possible_link is an unverified "
        "association, never a causal fact. Cite both endpoint events in links. Empty events/links allowed.\n"
        + json.dumps({"duration": duration, "value_definitions": values, "schema_example": schema})
    )
    content = [{"type": "text", "text": instruction}]
    for f in frames:
        content.extend(
            [
                {"type": "text", "text": json.dumps({"image_id": f["image_id"], "original_pts": f["pts"]})},
                {"type": "image_url", "image_url": {"url": f["url"]}},
            ]
        )
    result = client.complete(
        "evidence_memory",
        content,
        4096,
        lambda p: validate_memory(p, frames, values, duration),
        system=SYSTEM,
    )
    result = json.loads(json.dumps(result))
    for e in result["events"]:
        e["id"] = namespace + ":" + e["id"]
    for edge in result["links"]:
        for key in ("source", "target"):
            edge[key] = namespace + ":" + edge[key]
    return {
        **result,
        "values_sha256": digest(values),
        "frames": [{k: f[k] for k in ("image_id", "pts", "jpeg_sha256")} for f in frames],
        "status": "model_observations_unreviewed",
    }


def retrieval_keys(memory):
    """Separate factual and value-cue keys; caller keeps query relevance first."""
    keys = []
    for e in memory["events"]:
        keys.append({"event_id": e["id"], "kind": "event", "text": e["description"]})
        for v in e["values"]:
            if v["status"] == "observed":
                keys.append({"event_id": e["id"], "kind": v["dimension"], "text": v["cue"]})
    return keys
