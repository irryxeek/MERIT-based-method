"""Deterministic evidence-gap, budget and result-selection primitives."""

from dataclasses import asdict, dataclass

from .memory import frame_index, number, references, text


@dataclass(frozen=True)
class Need:
    event_id: str
    criterion: str
    question: str
    start: float
    end: float
    focus: str = "action"
    min_frames: int = 2
    max_gap_seconds: float = 0.5

    def __post_init__(self):
        if not all(text(s, n) for s, n in ((self.event_id, 80), (self.criterion, 80), (self.question, 400))):
            raise ValueError("A need requires event ID, criterion and a concrete question")
        if not number(self.start) or not number(self.end) or not 0 <= self.start < self.end:
            raise ValueError("Invalid verification interval")
        if self.focus not in {"action", "subject", "group", "context"}:
            raise ValueError("Unknown evidence focus")
        if type(self.min_frames) is not int or not 1 <= self.min_frames <= 24:
            raise ValueError("Invalid minimum frame count")
        if not number(self.max_gap_seconds) or self.max_gap_seconds <= 0:
            raise ValueError("Positive maximum temporal gap required")


@dataclass(frozen=True)
class Budget:
    max_views: int = 2
    max_new_frames: int = 48
    frames_per_view: int = 24
    max_comparisons: int = 3

    def __post_init__(self):
        if any(type(v) is not int or v < 0 for v in asdict(self).values()):
            raise ValueError("Budgets must be nonnegative integers")
        if not 1 <= self.frames_per_view <= 24:
            raise ValueError("At most 24 distinct frames per view")


def unresolved(needs, observations, frames, duration):
    """Structural sufficiency is necessary, not a semantic correctness guarantee.

    A negative answer with evidence is resolved; unknown is not a negative.
    For motion, require temporal support throughout the requested interval.
    """
    indexed = frame_index(frames, duration) if frames else {}
    keys = [(n.event_id, n.criterion) for n in needs]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate verification criterion")
    observed = {}
    for o in observations:
        if not isinstance(o, dict) or set(o) != {"event_id", "criterion", "answer", "evidence_ids"}:
            raise ValueError("Invalid verification observation")
        key = (o["event_id"], o["criterion"])
        if key not in keys or key in observed or o["answer"] not in {"yes", "no", "unknown"}:
            raise ValueError("Unknown/duplicate criterion or invalid answer")
        references(o["evidence_ids"], indexed, required=o["answer"] != "unknown")
        observed[key] = o
    gaps = []
    for n in needs:
        if n.end > duration:
            raise ValueError("Verification interval exceeds video")
        o = observed.get((n.event_id, n.criterion))
        reason = "missing_observation"
        if o is not None:
            pts = sorted({indexed[i]["pts"] for i in o["evidence_ids"]})
            if any(not n.start <= p < n.end for p in pts):
                raise ValueError("Evidence outside verification interval")
            if o["answer"] == "unknown":
                reason = "explicitly_unknown"
            elif len(pts) < n.min_frames:
                reason = "insufficient_distinct_frames"
            elif max(b - a for a, b in zip([n.start] + pts, pts + [n.end], strict=True)) > n.max_gap_seconds:
                reason = "temporal_coverage_gap"
            else:
                continue
        gaps.append({**asdict(n), "reason": reason})
    return gaps


def unique_pts(pts, start, end, count, *, seen=()):
    """Nearest real PTS at uniform targets, deduplicated; never interpolate/pad."""
    if not number(start) or not number(end) or not 0 <= start < end:
        raise ValueError("Invalid sampling interval")
    if type(count) is not int or count < 0 or any(not number(p) for p in pts):
        raise ValueError("Invalid frame budget or timestamp")
    if any(b <= a for a, b in zip(pts, pts[1:], strict=False)):
        raise ValueError("PTS must be strictly increasing")
    excluded = set(seen)
    available = [p for p in pts if start <= p < end and p not in excluded]
    count = min(count, len(available))
    selected = []
    for i in range(count):
        target = start + (i + 0.5) * (end - start) / count
        p = min(available, key=lambda p: (abs(p - target), p))
        selected.append(p)
        available.remove(p)
    return sorted(selected)


def plan_views(query, gaps, pts, duration, budget=None, *, seen=(), used_views=0, used_frames=0):
    """Ordered needs are provided by the query policy; no hidden GT or confidence threshold."""
    budget = Budget() if budget is None else budget
    if not text(query, 2000) or not number(duration) or duration <= 0:
        raise ValueError("Original query and video duration required")
    if type(used_views) is not int or type(used_frames) is not int or min(used_views, used_frames) < 0:
        raise ValueError("Invalid cumulative budget counters")
    views, deferred = [], []
    selected = set(seen)
    for gap in gaps:
        n = Need(**{k: gap[k] for k in Need.__dataclass_fields__})
        if n.end > duration:
            raise ValueError("Need exceeds source video")
        remaining = budget.max_new_frames - used_frames - sum(len(v["pts"]) for v in views)
        if used_views + len(views) >= budget.max_views or remaining <= 0:
            deferred.append({**gap, "deferred_reason": "budget_exhausted"})
            continue
        chosen = unique_pts(pts, n.start, n.end, min(budget.frames_per_view, remaining), seen=selected)
        if not chosen:
            deferred.append({**gap, "deferred_reason": "no_unseen_source_frames"})
            continue
        views.append(
            {
                "original_query": query,
                "event_id": n.event_id,
                "criterion": n.criterion,
                "question": n.question,
                "focus": n.focus,
                "start": n.start,
                "end": n.end,
                "pts": chosen,
                "operation": "sample_new_original_frames",
                "timeline": "original_segment_seconds",
                "min_frames": n.min_frames,
                "max_gap_seconds": n.max_gap_seconds,
            }
        )
        selected.update(chosen)
    return {"views": views, "deferred": deferred, "unresolved_before_views": len(gaps)}


def comparison_pairs(events, budget=None):
    """Spend on adjacent score-ranked relevant candidates; no all-pairs expansion."""
    budget = Budget() if budget is None else budget
    select_events(events, mode="all", threshold=0)  # Shared validity checks.
    relevant = sorted((e for e in events if e["relevant"]), key=lambda e: (-e["score"], e["start"], e["id"]))
    return [(a["id"], b["id"]) for a, b in zip(relevant, relevant[1:], strict=False)][
        : budget.max_comparisons
    ]


def select_events(events, *, mode, threshold, k=None):
    """Display selection only. Preserve independent events and expose boundary ties."""
    if mode not in {"all", "topk"} or not number(threshold) or not 0 <= threshold <= 4:
        raise ValueError("Explicit all/topk mode and a frozen [0,4] threshold required")
    if mode == "topk" and (type(k) is not int or k <= 0):
        raise ValueError("topk requires an explicit positive k")
    if mode == "all" and k is not None:
        raise ValueError("all does not accept k")
    identities = set()
    for e in events:
        if e["id"] in identities or type(e["relevant"]) is not bool:
            raise ValueError("Duplicate event or invalid relevance decision")
        identities.add(e["id"])
        if not number(e["score"]) or not 0 <= e["score"] <= 4 or not number(e["start"]):
            raise ValueError("Invalid event score or time")
    ranked = sorted(
        (e for e in events if e["relevant"] and e["score"] >= threshold),
        key=lambda e: (-e["score"], e["start"], e["id"]),
    )
    chosen = ranked if mode == "all" else ranked[:k]
    boundary_tie = bool(
        chosen and len(chosen) < len(ranked) and chosen[-1]["score"] == ranked[len(chosen)]["score"]
    )
    return {
        "selected_ids": [e["id"] for e in chosen],
        "boundary_tie": boundary_tie,
        "scope": "qualified supplied candidates only; not proof of exhaustive video coverage",
    }


def merge_dense_scores(initial, updates):
    """Evaluation keeps the complete 2-second vector, independently of display top-k."""
    if not initial or any(not number(s) or not 0 <= s <= 4 for s in initial):
        raise ValueError("Invalid initial HD vector")
    result, visited = list(initial), set()
    for item in updates:
        if set(item) != {"clip_id", "score"}:
            raise ValueError("Invalid dense update")
        i, s = item["clip_id"], item["score"]
        if type(i) is not int or i in visited or not 0 <= i < len(result):
            raise ValueError("Duplicate or out-of-range update")
        if not number(s) or not 0 <= s <= 4:
            raise ValueError("Invalid new score")
        visited.add(i)
        result[i] = s
    return result
