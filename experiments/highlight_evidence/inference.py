"""Optional live-client hooks for targeted verification and two-candidate comparison."""

import json

from .memory import SYSTEM, frame_index, references, text


def visual_content(instruction, frames):
    content = [{"type": "text", "text": instruction}]
    for f in frames:
        content.extend(
            [
                {"type": "text", "text": json.dumps({"image_id": f["image_id"], "original_pts": f["pts"]})},
                {"type": "image_url", "image_url": {"url": f["url"]}},
            ]
        )
    return content


def verify_view(client, view, frames, duration):
    indexed = frame_index(frames, duration)
    if sorted(f["pts"] for f in frames) != view["pts"]:
        raise ValueError("Evidence frames must exactly match the planned view")
    if not 1 <= len(frames) <= 24:
        raise ValueError("Verification view exceeds frame budget")

    def validate(p):
        if not isinstance(p, dict) or set(p) != {"event_id", "criterion", "answer", "evidence_ids"}:
            raise ValueError("Invalid verification result")
        if p["event_id"] != view["event_id"] or p["criterion"] != view["criterion"]:
            raise ValueError("Verification changed event or criterion")
        if p["answer"] not in {"yes", "no", "unknown"}:
            raise ValueError("Unknown answer category")
        references(p["evidence_ids"], indexed, required=p["answer"] != "unknown")
        return p

    prompt = (
        "Verify this specific highlight-related question in the context of the ORIGINAL query. "
        "Do not replace it with generic scene description. Negative visible evidence permits no; "
        "unseen or ambiguous evidence requires unknown. Report visible expression/motion, not internal "
        "mental states. Original timestamps are unchanged by dense frame sampling. "
        "For a duration-dependent answer cite enough distributed frames to support the interval; "
        "an unobserved interval remains unknown. "
        "Return exactly {event_id,criterion,answer:yes|no|unknown,evidence_ids:[supplied image_ids]}.\n"
        + json.dumps(
            {
                k: view[k]
                for k in (
                    "original_query",
                    "event_id",
                    "criterion",
                    "question",
                    "focus",
                    "start",
                    "end",
                    "min_frames",
                    "max_gap_seconds",
                )
                if k in view
            }
        )
    )
    return client.complete("highlight_verify", visual_content(prompt, frames), 512, validate, system=SYSTEM)


def compare_candidates(client, query, dimension, left, right, frames, duration):
    """Comparison is query-specific evidence, never a query-independent memory fact.

    Inputs: each candidate has id and evidence_ids. Results do not automatically
    overwrite dense scores; this avoids arbitrary score mixing across batches.
    """
    indexed = frame_index(frames, duration)
    if not text(query, 2000) or not text(dimension, 500) or left["id"] == right["id"]:
        raise ValueError("Explicit query, comparison criterion and distinct events required")
    if len(frames) > 24:
        raise ValueError("Comparison requires at most 24 total images")
    for e in (left, right):
        references(e["evidence_ids"], indexed)
    if set(left["evidence_ids"]) & set(right["evidence_ids"]):
        raise ValueError("Compare distinct event evidence, not duplicate views")

    def validate(p):
        if not isinstance(p, dict) or set(p) != {"winner", "evidence_ids", "basis"}:
            raise ValueError("Invalid comparison schema")
        if p["winner"] not in {"left", "right", "tie", "unknown"}:
            raise ValueError("winner must be exactly left, right, tie or unknown")
        if not text(p["basis"], 240):
            raise ValueError(
                "basis must be 1-240 characters. Shorten to one brief visible difference, not paragraphs."
            )
        allowed = set(left["evidence_ids"]) | set(right["evidence_ids"])
        references(p["evidence_ids"], {i: indexed[i] for i in allowed}, required=p["winner"] != "unknown")
        if p["winner"] != "unknown" and not all(
            set(p["evidence_ids"]) & set(e["evidence_ids"]) for e in (left, right)
        ):
            raise ValueError("A comparison must cite evidence on both sides")
        return p

    prompt = (
        "Compare two candidate highlights for the original query using only the requested dimension. "
        "Compare observable behavior: amplitude, duration, participants and visible context, "
        "where relevant; visual scale and camera cuts can confound intensity. Do not infer audio. "
        "Allow ties; answer unknown when viewpoints or evidence do not support comparison. "
        "Cite 1-3 frames from EACH event for a decision. Keep basis under 160 characters (one short sentence). "
        "Do not prefer a candidate merely because it has more supplied frames. Return exactly "
        "{winner:left|right|tie|unknown,evidence_ids:[image_ids],basis:brief observable difference}.\n"
        + json.dumps({"original_query": query, "dimension": dimension, "left": left, "right": right})
    )
    result = client.complete(
        "highlight_compare", visual_content(prompt, frames), 768, validate, system=SYSTEM
    )
    return {"query": query, "dimension": dimension, "left_id": left["id"], "right_id": right["id"], **result}
