"""One bounded plan for both observation arms, using actionable uncertainty only."""

from copy import deepcopy
from typing import Literal

from evidence_grounding import ground_observation, memory_source, source_anchor
from io_utils import compact, digest
from pydantic import Field
from schema import Contract


class LocalStep(Contract):
    gap_id: str
    gap: Literal["attribution", "value", "boundary"]
    question: str = Field(min_length=8, max_length=160)
    candidate: int
    start: float
    end: float


class LocalPlan(Contract):
    actions: list[LocalStep] = Field(max_length=2)


PLAN = (
    "Plan at most two local observations to resolve the supplied actionable evidence gaps for this request. "
    "Do not revisit supported ties just to force a winner. For each action choose a listed gap_id and one of "
    "its candidate anchors, a gap type (attribution, value, boundary), a short concrete visual question, "
    "then a specific source-video "
    "start/end subinterval inside that anchor's allowed "
    "range. Choose where the missing action, outcome, reaction, identity or context could be checked; do not "
    "claim the unseen answer. Two actions are planned now, before any new observation. Return no action if "
    "none can address a concrete gap. Avoid repeated actions. Source times are seconds, not frame indices. "
    "A direction_conflict means reversed comparisons disagreed, not a supported tie. Use the bounded memory "
    "and compressed reasons to ask what visible action, outcome, reaction, event attribution or context could "
    "resolve that conflict. A generic 'which is better' is not an evidence question. Do not force an action "
    "when the conflict cannot be checked locally. Both observation arms will receive this exact same question. "
    "Task: {task}\nCandidates and permitted ranges: {candidates}\n"
    "Bounded active memory: {memory}\nCompressed unresolved comparisons: {gaps}"
)


def actionable_gaps(arm):
    comparisons = {(x["a"], x["b"]): x["result"] for x in arm["comparisons"]}
    gaps = []
    for edge in arm["ranking"]["edges"]:
        if edge["state"] != "unresolved":
            continue
        a, b = sorted([edge["a"], edge["b"]])
        questions = []
        for pair in ((a, b), (b, a)):
            item = comparisons[pair]
            question = (item.get("missing_question") or "").strip()
            if item["preference"] == "insufficient" and question and "no actionable question" not in question.lower():
                if question not in questions:
                    questions.append(question)
        for i, question in enumerate(questions):
            gaps.append(
                {"gap_id": f"g{a}_{b}_{i}", "candidates": [a, b], "question": question, "origin": "insufficient"}
            )
        if not questions and all(comparisons[p]["preference"] != "insufficient" for p in ((a, b), (b, a))):
            gaps.append(
                {
                    "gap_id": f"g{a}_{b}_conflict",
                    "candidates": [a, b],
                    "question": None,
                    "origin": "direction_conflict",
                    "judgments": [
                        {
                            "a": x,
                            "b": y,
                            "preference": comparisons[x, y]["preference"],
                            "reason": comparisons[x, y]["reason"][:160],
                        }
                        for x, y in ((a, b), (b, a))
                    ],
                }
            )
    return sorted(gaps, key=lambda x: (x["candidates"], x["gap_id"]))


def candidate_ranges(selected, duration):
    return [
        {
            "candidate": c,
            "target": [2.0 * c, min(2.0 * c + 2, duration)],
            "allowed": [max(0.0, 2.0 * c - 4), min(duration, 2.0 * c + 6)],
        }
        for c in sorted(selected)
    ]


def validate_plan(value, gaps, candidates):
    value = LocalPlan.model_validate(value).model_dump()
    allowed = {v["candidate"]: v["allowed"] for v in candidates}
    indexed = {v["gap_id"]: v for v in gaps}
    seen = set()
    actions = []
    for item in value["actions"]:
        if not item["question"].strip():
            raise ValueError("Observation requires a concrete visual question")
        gap = indexed.get(item["gap_id"])
        if not gap or item["candidate"] not in gap["candidates"] or item["candidate"] not in allowed:
            raise ValueError("Plan invented a gap or unrelated candidate")
        lo, hi = allowed[item["candidate"]]
        if not lo <= item["start"] < item["end"] <= hi:
            raise ValueError("R2 plan outside candidate +/-4-second range")
        identity = (item["gap_id"], item["candidate"], item["start"], item["end"])
        if identity in seen:
            raise ValueError("Duplicate planned action")
        seen.add(identity)
        actions.append(
            {**item, "gap_origin": gap.get("origin", "insufficient"), "allowed": [lo, hi], "tool": "Segment"}
        )
    return {"actions": actions}


def resolve_plan(plan, gaps, candidates, task):
    """An abstaining planner cannot silently discard a concrete attribution conflict.

    This deterministic controller amendment is NOT an LLM-authored action or a
    retry. It checks both anchors of one conflicting pair, never invents a winner,
    and never triggers on supported ties or uncheckable insufficient questions.
    """
    if plan["actions"]:
        return {"actions": plan["actions"], "source": "model", "rule": None}
    conflicts = [g for g in gaps if g.get("origin") == "direction_conflict"]
    if not conflicts:
        return {"actions": [], "source": "none", "rule": None}
    degree = {}
    for gap in conflicts:
        for c in gap["candidates"]:
            degree[c] = degree.get(c, 0) + 1
    chosen = min(conflicts, key=lambda g: (-max(degree[c] for c in g["candidates"]), g["candidates"], g["gap_id"]))
    indexed = {c["candidate"]: c for c in candidates}
    cue = task["query"]
    if len(cue) > 80:
        # A literal query prefix is a cue, not a rewritten task; full TaskSpec
        # remains available to the later comparison.
        cue = cue[:80].rsplit(" ", 1)[0]
    actions = []
    for candidate in sorted(chosen["candidates"], key=lambda c: (-degree[c], c)):
        item = indexed[candidate]
        start, end = item["target"]
        lo, hi = item["allowed"]
        actions.append(
            {
                "gap_id": chosen["gap_id"],
                "gap": "attribution",
                "candidate": candidate,
                "question": f"In {start:g}-{end:g}s, is query cue '{cue}' visible here or only in neighboring scenes?",
                "start": max(lo, start - 1),
                "end": min(hi, end + 1),
            }
        )
    resolved = validate_plan({"actions": actions}, gaps, candidates)
    return {
        **resolved,
        "source": "controller_conflict_probe",
        "rule": "highest_conflict_degree_then_time; both_anchors; target_plus_minus_1s",
    }


def update_memory(memory, observations, tokenizer, *, target_intervals=None):
    """Retain new facts first; old content is accessible only through its bounded view."""
    from memory import retrieve

    old = retrieve(memory, tokenizer)
    entries, new_ids, targets = [], [], set()
    for number, observation in enumerate(observations):
        candidate = observation["action"]["candidate"]
        targets.add(candidate)
        if target_intervals is not None:
            grounded = ground_observation(
                observation["result"], observation["trace"], candidate,
                target_intervals[candidate], f"obs{number}_c{candidate}",
            )
            for identity, fact in grounded.items():
                new_ids.append(identity)
                entries.append({
                    "text": fact["text"], "fact_ids": [identity],
                    "sources": {identity: memory_source(fact)},
                })
            continue
        result = observation["result"]
        facts = result["facts"] + [{"kind": "unknown", "text": text, "frames": []} for text in result["unknown"]]
        for index, fact in enumerate(facts):
            identity = f"obs{number}_c{candidate}_f{index}"
            new_ids.append(identity)
            entries.append(
                {
                    "text": "Context near target; event attribution unverified. " + fact["text"],
                    "fact_ids": [identity],
                    "sources": {identity: {"candidate": candidate, "frames": fact["frames"], "kind": fact["kind"]}},
                }
            )
    old_entries = sorted(
        deepcopy(old["items"]),
        key=lambda x: (
            not any(source_anchor(s) in targets for s in x["sources"].values()),
            min(source_anchor(s) for s in x["sources"].values()),
            sorted(x["fact_ids"]),
        ),
    )
    retained, removed = [], []
    for entry in entries + old_entries:
        if len(tokenizer.encode(compact({"items": retained + [entry]}), add_special_tokens=False)) <= 1024:
            retained.append(entry)
        else:
            removed.extend(entry["fact_ids"])
    view = {"items": retained}
    kept = sorted({i for item in retained for i in item["fact_ids"]})
    return {
        "view": view,
        "view_sha256": digest(view),
        "capacity": 1024,
        "tokens": len(tokenizer.encode(compact(view), add_special_tokens=False)),
        "retained_fact_ids": kept,
        "capacity_evictions": sorted(set(removed) - set(kept)),
        "deleted_fact_ids": sorted((set(memory["retained_fact_ids"]) | set(new_ids)) - set(kept)),
        "fact_store_version": digest({"parent": memory["fact_store_version"], "observations": observations}),
        "update_rule": "new observation facts, relevant retained old entries, remaining retained old entries",
        **({"evidence_interface": "temporal_v1"} if target_intervals is not None else {}),
        "no_raw_fact_bypass": True,
    }
