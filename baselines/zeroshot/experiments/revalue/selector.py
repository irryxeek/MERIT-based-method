"""Purpose-dependent feasible selection; no silent truncation of required content."""

from schema import numeric_limit


def select_events(events, order, task):
    by_id = {e["event_id"]: e for e in events}
    if len(by_id) != len(events) or len(order) != len(set(order)) or set(order) != set(by_id):
        raise ValueError("Event order/identity mismatch")
    count = numeric_limit(task, "count")
    duration_limit = numeric_limit(task, "duration_seconds")
    selected, decisions, groups, total = [], [], set(), 0.0
    unknown = []
    for identity in order:
        event = by_id[identity]
        start, end = event["start"], event["end"]
        if not 0 <= start < end or not event.get("evidence"):
            raise ValueError("An event interval needs positive length and real evidence")
        group = event.get("confirmed_duplicate_group")
        reason = None
        if event["eligibility"] != "satisfied":
            reason = "eligibility_" + event["eligibility"]
        elif event.get("attribution_conflict"):
            reason = "attribution_unresolved"
        elif not event["necessary_contents"] or any(not c["supported"] for c in event["necessary_contents"]):
            reason = "necessary_content_unresolved"
        elif not event.get("boundary_confirmed", False):
            reason = "boundary_unresolved"
        elif group and group in groups:
            reason = "confirmed_duplicate"
        elif count is not None and len(selected) >= count:
            reason = "count_limit"
        elif duration_limit is not None and total + end - start > duration_limit:
            reason = "duration_limit_requires_reselection_not_truncation"
        if reason is None:
            selected.append(event)
            total += end - start
            if group:
                groups.add(group)
        elif "unresolved" in reason or reason == "eligibility_unknown":
            unknown.append({"event_id": identity, "reason": reason, "boundary_range": event.get("boundary_range")})
        decisions.append({"event_id": identity, "accepted": reason is None, "reason": reason or "feasible"})
    missing_count = count is not None and len(selected) < count
    status = "abstain" if not selected else "partial" if missing_count or unknown else "complete"
    return {
        "status": status,
        "events": selected,
        "duration_seconds": total,
        "decisions": decisions,
        "unresolved": unknown,
        "count_satisfied": not missing_count,
        "selection_rule": "deterministic greedy; not globally optimal",
    }
