"""Frozen-pilot adapters: contextual cues, query policy, retrieval and HD scoring."""

import copy
import json
import math
from dataclasses import asdict

import numpy as np

from .inference import visual_content
from .memory import SYSTEM, check_values, number, references, text, validate_memory
from .planner import Need


def brief(event):
    return {k: event[k] for k in ("id", "start", "end", "description", "evidence_ids", "values")}


def validate_context_link(edge, available, current, seen):
    if not isinstance(edge, dict) or set(edge) != {"source", "target", "relation", "evidence_ids"}:
        raise ValueError("Link needs source,target,relation,evidence_ids only")
    a, b = edge["source"], edge["target"]
    if a not in available or b not in available or a == b or not {a, b} & set(current):
        raise ValueError("Link needs two supplied event IDs, at least one from current chunk")
    if edge["relation"] not in {"before", "possible_link"}:
        raise ValueError("Only before or possible_link; temporal order does not prove causality")
    identity = (a, b, edge["relation"])
    if identity in seen:
        raise ValueError("Duplicate link")
    allowed = set(available[a]["evidence_ids"]) | set(available[b]["evidence_ids"])
    references(edge["evidence_ids"], {i: None for i in allowed})
    if not all(set(edge["evidence_ids"]) & set(available[i]["evidence_ids"]) for i in (a, b)):
        raise ValueError(f"Link {a}->{b} must cite at least one frame from EACH endpoint")
    if edge["relation"] == "before" and available[a]["end"] > available[b]["start"]:
        raise ValueError(f"before edge is backwards or overlaps: {a}->{b}")
    return identity


def extract_context_memory(client, frames, values, duration, start, end, namespace, context):
    """Query-independent chunk memory with explicit references to observed context.

    Context is restricted to the preceding two chunks. It is model-produced
    observation, not truth; unsupported novelty/importance must stay unknown.
    """
    cue_ids = check_values(values)
    previous = {e["id"]: e for e in context}

    def validate(payload):
        if not isinstance(payload, dict) or set(payload) != {"events", "links"}:
            raise ValueError("Expected events and links")
        stripped = copy.deepcopy(payload)
        current = {e["id"]: e for e in payload["events"]}
        available = {**previous, **current}
        if set(previous) & set(current):
            raise ValueError("Use local event IDs e0/e1, not an earlier event ID")
        for e in stripped["events"]:
            if not start <= e["start"] < e["end"] <= end:
                raise ValueError("Current event outside the supplied chunk")
            for v in e["values"]:
                refs = v.pop("reference_event_ids")
                try:
                    references(refs, available, required=False)
                except ValueError as error:
                    raise ValueError(
                        f"{e['id']} {v.get('dimension')} reference_event_ids={refs} is invalid. "
                        f"Cite ONLY event IDs from {sorted(available)}. Do not copy references from older "
                        "records when those events are not supplied here. Use unknown with null level "
                        "if novelty/importance lacks a supplied reference."
                    ) from error
                if e["id"] in refs:
                    raise ValueError("Self-reference is not contextual evidence")
                if (
                    v["status"] == "observed"
                    and v["dimension"] in {"novelty_cue", "importance_cue"}
                    and not refs
                ):
                    raise ValueError("Observed novelty/importance needs another contextual event")
                if not text(v.get("cue"), 160):
                    raise ValueError(
                        f"{e['id']} {v.get('dimension')} cue must be 1-160 characters; "
                        f"received {len(v.get('cue', ''))}. Shorten the visible observation."
                    )
        # Validate current nodes against current pixels. Previous nodes have already
        # passed the same citation checks on their own source frames.
        stripped["links"] = []
        validate_memory(stripped, frames, values, duration)
        if not isinstance(payload["links"], list) or len(payload["links"]) > 8:
            raise ValueError("At most eight links per chunk")
        seen, accepted, rejected = set(), [], []
        for edge in payload["links"]:
            try:
                identity = validate_context_link(edge, available, current, seen)
            except (ValueError, TypeError, KeyError) as error:
                # Optional graph edges are omitted, never reversed or repaired into facts.
                # The original response and exact rejected edge remain auditable.
                rejected.append({"edge": edge, "reason": str(error)})
            else:
                accepted.append(edge)
                seen.add(identity)
        return {"events": payload["events"], "links": accepted, "rejected_links": rejected}

    example = {
        "events": [
            {
                "id": "e0",
                "start": start,
                "end": end,
                "description": "short visible event",
                "evidence_ids": [frames[0]["image_id"]],
                "values": [
                    {
                        "dimension": c,
                        "status": "unknown",
                        "level": None,
                        "cue": "insufficient observation",
                        "evidence_ids": [],
                        "reference_event_ids": [],
                    }
                    for c in cue_ids
                ],
            }
        ],
        "links": [],
    }
    prompt = (
        "Build query-independent evidence memory for the CURRENT chunk. Return at most 4 events and "
        "8 links between supplied events. Use local IDs e0,e1,etc for new nodes; preserve prior IDs. "
        "Every event has all four cue dimensions exactly once. "
        "Status is observed/unknown/not_applicable. Observed levels: 0 absent, 1 mild, 2 moderate, 3 strong; "
        "otherwise level=null. Cite current supplied frame IDs; event [start,end) must contain cited frames. "
        "Description <=240 characters; each cue <=160. Do not fill missing actions from common sense. "
        "Prior events below are imperfect model observations, not ground truth. Novelty requires an "
        "explicit observed reference, importance requires event-goal/outcome context. If observed, these "
        "two dimensions MUST cite at least one other current/prior event in reference_event_ids. "
        "A prior event's existence alone does not establish a contrast or importance. Otherwise use unknown. "
        "Links use {source,target,relation:before|possible_link,evidence_ids}; cite BOTH endpoints. "
        "For a link to a prior event, include a frame ID from that prior event AND from the current event. "
        "before is directed EARLIER source -> LATER target with nonoverlapping intervals; never reverse it. "
        "At least one endpoint must be current. No causes edges. Empty events/links are allowed.\n"
        + json.dumps(
            {
                "video_duration": duration,
                "current_interval": [start, end],
                "value_definitions": values,
                "prior_observations": [brief(e) for e in context],
                "allowed_prior_reference_event_ids": sorted(previous),
                "output_example": example,
            }
        )
    )
    result = client.complete("cue_memory", visual_content(prompt, frames), 4096, validate, system=SYSTEM)
    result = copy.deepcopy(result)
    current_ids = {e["id"] for e in result["events"]}
    for e in result["events"]:
        e["id"] = namespace + ":" + e["id"]
        for v in e["values"]:
            v["reference_event_ids"] = [
                namespace + ":" + i if i in current_ids else i for i in v["reference_event_ids"]
            ]
    for link in result["links"]:
        for key in ("source", "target"):
            if link[key] in current_ids:
                link[key] = namespace + ":" + link[key]
    return result


def compile_policy(client, query, values):
    allowed = set(check_values(values))

    def validate(p):
        if not isinstance(p, dict) or set(p) != {"facets"} or not isinstance(p["facets"], list):
            raise ValueError("Return facets only")
        if any(not isinstance(x, str) for x in p["facets"]) or not 1 <= len(p["facets"]) <= 4:
            raise ValueError("One to four cue IDs required")
        if (
            len(set(p["facets"])) != len(p["facets"])
            or not set(p["facets"]) <= allowed
            or "event_cue" not in p["facets"]
        ):
            raise ValueError("Unique configured cues required, including event_cue for relevance")
        return p

    prompt = (
        "Select cue dimensions needed to judge highlights for the original query. Always include event_cue "
        "for relevance. Add emotion, novelty or importance only when the query or its event goal warrants "
        "it; do not assume all highlights require emotion. Return {facets:[configured cue IDs]}.\n"
        + json.dumps({"original_query": query, "cue_definitions": values})
    )
    return client.complete("query_policy", [{"type": "text", "text": prompt}], 256, validate, system=SYSTEM)


def choose_windows(chunks, initial, query, policy, embedder, limit=3):
    """Round-robin semantic, selected-cue and D0 rankings, capped at three chunks."""
    texts, owners, kinds = [], [], []
    for chunk in chunks:
        for e in chunk.get("events", []):
            texts.append(e["description"])
            owners.append(chunk["window_id"])
            kinds.append("event")
            for v in e["values"]:
                if v["status"] == "observed" and v["dimension"] in policy["facets"]:
                    texts.append(e["description"] + "; " + v["dimension"] + ": " + v["cue"])
                    owners.append(chunk["window_id"])
                    kinds.append("cue")
    semantic, cue = {}, {}
    if texts:
        vectors = embedder.encode(texts)
        q = embedder.encode([query])[0]
        for owner, kind, score in zip(owners, kinds, vectors @ q, strict=True):
            target = semantic if kind == "event" else cue
            target[owner] = max(target.get(owner, -math.inf), float(score))
    rankings = [sorted(scores, key=lambda i: (-scores[i], i)) for scores in (semantic, cue)]
    d0 = {chunk["window_id"]: max(initial[i] for i in chunk["bin_ids"]) for chunk in chunks}
    rankings.append(sorted(d0, key=lambda i: (-d0[i], i)))
    selected = []
    for depth in range(len(chunks)):
        for ranked in rankings:
            if depth < len(ranked) and ranked[depth] not in selected:
                selected.append(ranked[depth])
                if len(selected) == min(limit, len(chunks)):
                    return sorted(selected), {"semantic": semantic, "cue": cue, "D0_peak": d0}
    return sorted(selected), {"semantic": semantic, "cue": cue, "D0_peak": d0}


def context_for(chunks, selected):
    all_events = {e["id"]: e for c in chunks for e in c.get("events", [])}
    chosen = [e for c in chunks if c["window_id"] in selected for e in c.get("events", [])]
    references_needed = {i for e in chosen for v in e["values"] for i in v["reference_event_ids"]}
    links = [edge for c in chunks if c["window_id"] in selected for edge in c.get("links", [])]
    references_needed.update(i for edge in links for i in (edge["source"], edge["target"]))
    return {
        "selected_events": chosen,
        "reference_events": [
            all_events[i]
            for i in sorted(references_needed)
            if i in all_events and i not in {e["id"] for e in chosen}
        ],
        "links": links,
        "caution": "Model observations, not truth. Unknown/inapplicable cues are not zero highlight value.",
    }


def select_needs(client, query, policy, chunks, selected, duration):
    context = context_for(chunks, selected)
    allowed = {e["id"]: e for e in context["selected_events"]}
    if not allowed:
        return []
    options = {}
    for e in allowed.values():
        intervals = sorted(
            {(e["start"], min(e["end"], e["start"] + 6)), (max(e["start"], e["end"] - 6), e["end"])}
        )
        for start, end in intervals:
            if not 0 <= start < end <= duration:
                raise ValueError("Candidate event exceeds video")
            options[f"v{len(options)}"] = {"event_id": e["id"], "start": start, "end": end}

    def validate(p):
        if (
            not isinstance(p, dict)
            or set(p) != {"needs"}
            or not isinstance(p["needs"], list)
            or len(p["needs"]) > 2
        ):
            raise ValueError("Return needs, at most two")
        seen, result = set(), []
        for v in p["needs"]:
            if not isinstance(v, dict) or set(v) != {
                "view_id",
                "criterion",
                "question",
                "reason",
            }:
                raise ValueError(
                    "Each need must contain view_id,criterion,question,reason ONLY; no timestamps or focus"
                )
            if v["view_id"] not in options:
                raise ValueError("Choose a view_id from: " + ",".join(options))
            if not text(v["criterion"], 60) or not text(v["reason"], 200):
                raise ValueError("criterion must be 1-60 characters; reason must be 1-200 characters")
            if v["view_id"] in seen:
                raise ValueError("Select each view_id at most once")
            n = Need(
                **options[v["view_id"]],
                criterion=v["criterion"] + ":" + v["view_id"],
                question=v["question"],
                focus="context",
            )
            seen.add(v["view_id"])
            result.append({**asdict(n), "reason": v["reason"]})
        return {"needs": result}

    prompt = (
        "Identify at most two unanswered, HIGHLIGHT-RELATED checks that could change the ranking of selected "
        "events for the original query. Use concrete yes/no visual questions, not generic descriptions. "
        "If existing evidence is adequate return needs:[]. CHOOSE supplied view_id options; timestamps "
        "and source event are fixed by the controller. Ask only about the selected view's interval, not "
        "about an entire longer event. Name the visible subject/group/action in the question. "
        "Do not try to prove subjective novelty from a single image or infer missing audio. Missing event "
        "outcome, sparse temporal detail, visible reactions, or a concrete contextual difference can justify "
        "a view. Return exactly {needs:[{view_id:'v0',criterion:'outcome',question:'Is the outcome visible?',"
        "reason:'brief missing evidence'}]}. Do not output event_id,start,end,focus or any extra fields.\n"
        + json.dumps(
            {
                "original_query": query,
                "policy": policy,
                "duration": duration,
                "memory": context,
                "allowed_views": options,
            }
        )
    )
    return client.complete(
        "evidence_gaps", [{"type": "text", "text": prompt}], 1024, validate, system=SYSTEM
    )["needs"]


def score_evidence(
    client, stage, row, ids, frames, memory, *, observations=None, gaps=None, comparisons=None
):
    if len(frames) > 80:
        raise ValueError("Scoring image budget exceeded")

    def validate(p):
        if not isinstance(p, dict) or set(p) != {"clip_scores"} or not isinstance(p["clip_scores"], list):
            raise ValueError("Return clip_scores only")
        if len(p["clip_scores"]) != len(ids):
            raise ValueError("Complete requested bin coverage required")
        for item, expected in zip(p["clip_scores"], ids, strict=True):
            if not isinstance(item, dict) or set(item) != {"clip_id", "score"}:
                raise ValueError("Invalid score row")
            if type(item["clip_id"]) is not int or item["clip_id"] != expected:
                raise ValueError("Score IDs must match the requested order")
            if not number(item["score"]) or not 0 <= item["score"] <= 4:
                raise ValueError("HD score must be finite in [0,4]")
        return p

    instruction = (
        "Score EVERY requested 2-second clip for relevance and highlight suitability for the ORIGINAL QUERY. "
        "Use the same 0-4 ranking scale as dense HD: higher is more relevant and salient, decimals allowed. "
        "Do not normalize this batch or force a winner. Memory cues and comparisons are supporting model "
        "observations, not labels or probabilities; never sum cue levels. Prioritize direct visual evidence. "
        "Unknown does not mean low saliency. Contextual evidence can explain an event without making every "
        "context clip a highlight. Do not raise unrelated emotional footage. Respect true timestamps and "
        "output only supplied clip IDs in order. Return {clip_scores:[{clip_id:0,score:2.0},...]}, no prose.\n"
        + json.dumps(
            {
                "original_query": row["query"],
                "vid": row["vid"],
                "duration": row["duration"],
                "requested_bins": [{"clip_id": i, "start": 2 * i, "end": 2 * (i + 1)} for i in ids],
                "memory": memory,
                "verification": observations or [],
                "unresolved": gaps or [],
                "comparisons": comparisons or [],
            }
        )
    )
    return client.complete(stage, visual_content(instruction, frames), 1024, validate, system=SYSTEM)[
        "clip_scores"
    ]


def representative_frames(frames, start, end, limit=12):
    candidates = sorted((f for f in frames if start <= f["pts"] < end), key=lambda f: f["pts"])
    if len(candidates) <= limit:
        return candidates
    return [candidates[i] for i in np.linspace(0, len(candidates) - 1, limit).round().astype(int)]
