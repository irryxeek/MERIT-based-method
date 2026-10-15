from .common import finite


class EmptyCandidates(ValueError):
    pass


def windows(values, duration, *, allow_empty=True, canonicalize=False):
    if not isinstance(values, list) or len(values) > 10:
        raise ValueError("MR requires an array of at most ten candidates")
    parsed = []
    for value in values:
        if not isinstance(value, list) or len(value) != 3:
            raise ValueError("MR candidate must be [start,end,score]")
        start, end, score = map(finite, value)
        if not 0 <= start < end <= duration:
            raise ValueError("MR candidate outside segment-relative timeline")
        parsed.append([start, end, score])
    if canonicalize:
        parsed.sort(key=lambda w: -w[2])
        seen = set()
        parsed = [
            w for w in parsed if not ((w[0], w[1]) in seen or seen.add((w[0], w[1])))
        ]
    elif any(a[2] < b[2] for a, b in zip(parsed, parsed[1:])):
        raise ValueError("MR candidates are not in descending confidence order")
    if not parsed and not allow_empty:
        raise EmptyCandidates(
            "empty MR candidates: official R1 cannot score this query"
        )
    return parsed


def clip_scores(payload, expected_ids):
    if not isinstance(payload, dict) or set(payload) != {"clip_scores", "events"}:
        raise ValueError("HD output requires exactly clip_scores and events")
    rows = payload["clip_scores"]
    if not isinstance(rows, list) or len(rows) != len(expected_ids):
        raise ValueError("HD length mismatch")
    if not isinstance(payload["events"], str) or len(payload["events"]) > 2000:
        raise ValueError("HD event summary invalid/too long")
    scores = []
    for row, expected in zip(rows, expected_ids):
        if not isinstance(row, dict) or set(row) != {"clip_id", "score"}:
            raise ValueError("invalid HD score row")
        if (
            isinstance(row["clip_id"], bool)
            or not isinstance(row["clip_id"], int)
            or row["clip_id"] != expected
        ):
            raise ValueError("HD IDs or order mismatch")
        scores.append(finite(row["score"]))
    return scores


def validate_predictions(
    rows, manifest, task, *, allow_partial=False, allow_empty=False
):
    expected = {r["qid"]: r for r in manifest}
    actual = [r.get("qid") for r in rows]
    if any(isinstance(q, bool) or not isinstance(q, int) for q in actual):
        raise ValueError("prediction qids must be integers")
    if len(actual) != len(set(actual)):
        raise ValueError("duplicate prediction qids")
    if set(actual) - set(expected) or (
        not allow_partial and set(actual) != set(expected)
    ):
        raise ValueError("expected and predicted qids differ")
    key = "pred_relevant_windows" if task == "mr" else "pred_saliency_scores"
    for row in rows:
        if set(row) != {"qid", "query", "vid", key}:
            raise ValueError("prediction contains missing or extra fields")
        item = expected[row["qid"]]
        if row["vid"] != item["vid"] or row["query"] != item["query"]:
            raise ValueError("qid/query/video binding mismatch")
        if task == "mr":
            windows(row[key], item["duration"], allow_empty=allow_empty)
        else:
            if not isinstance(row[key], list) or len(row[key]) != item["num_clips"]:
                raise ValueError("HD length differs from floor(duration/2)")
            for v in row[key]:
                finite(v)
    return {
        "planned": len(expected),
        "predicted": len(actual),
        "missing_qids": sorted(set(expected) - set(actual)),
    }
