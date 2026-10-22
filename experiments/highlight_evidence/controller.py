"""Bounded evidence acquisition for caller-specified query criteria.

Candidate generation, criterion compilation and final calibrated scoring are
separate research components. This controller never reads saliency labels.
"""

from .inference import verify_view
from .media import decode_view
from .planner import Budget, plan_views, unresolved


def resolve_evidence(client, query, needs, observations, frames, video_path, info, duration, *, budget=None):
    budget = Budget() if budget is None else budget
    evidence = {f["image_id"]: f for f in frames}
    if len(evidence) != len(frames):
        raise ValueError("Duplicate initial evidence")
    answers = list(observations)
    gaps = unresolved(needs, answers, list(evidence.values()), duration)
    steps, used_views, used_frames = [], 0, 0
    while gaps and used_views < budget.max_views and used_frames < budget.max_new_frames:
        plan = plan_views(
            query,
            gaps,
            info["pts"],
            duration,
            budget,
            seen=[f["pts"] for f in evidence.values()],
            used_views=used_views,
            used_frames=used_frames,
        )
        if not plan["views"]:
            steps.append({"stage": "stop", "deferred": plan["deferred"]})
            break
        for view in plan["views"]:
            # Reserve budget before decode/network I/O; failures also consume it.
            used_views += 1
            used_frames += len(view["pts"])
            record = {"stage": "verify", "view": view}
            try:
                fresh = decode_view(video_path, info, view)
                # Short request-local aliases keep dense temporal citations within
                # the response-token budget; retain canonical media identities.
                fresh = [
                    {**f, "source_image_id": f["image_id"], "image_id": f"r{used_views}f{i}"}
                    for i, f in enumerate(fresh)
                ]
                evidence.update({f["image_id"]: f for f in fresh})
                answer = verify_view(client, view, fresh, duration)
                key = (answer["event_id"], answer["criterion"])
                answers = [a for a in answers if (a["event_id"], a["criterion"]) != key] + [answer]
                record.update(status="ok", observation=answer)
            except (ValueError, RuntimeError, OSError) as error:
                record.update(status="failed", error=str(error))
            steps.append(record)
        gaps = unresolved(needs, answers, list(evidence.values()), duration)
    return {
        "status": "criteria_structurally_resolved" if not gaps else "unresolved",
        "original_query": query,
        "observations": answers,
        "gaps": gaps,
        "steps": steps,
        "used_views": used_views,
        "reserved_new_frames": used_frames,
        "new_frames_decoded": len(evidence) - len(frames),
        "frames": list(evidence.values()),
        "semantic_correctness": "not established by structural sufficiency",
    }
