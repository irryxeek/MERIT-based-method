"""Four-state comparisons and conservative original-score permutation."""

import itertools

from evidence_grounding import direct_source
from schema import Comparison


def allowed_references(candidate, frame_ids, memory_view):
    allowed = set(frame_ids[candidate])
    for item in memory_view.get("items", []):
        for fact_id, source in item["sources"].items():
            if direct_source(source, candidate):
                allowed.add(fact_id)
                allowed.update(source["frames"])
    return sorted(allowed)


def validate_comparison(value, a, b, frame_ids, memory_view):
    result = Comparison.model_validate(value).model_dump()
    for side, candidate in (("a", a), ("b", b)):
        allowed = set(allowed_references(candidate, frame_ids, memory_view))
        refs = result[f"{side}_evidence"]
        if not set(refs) <= allowed:
            raise ValueError("Comparison assigns evidence to the wrong candidate")
        if result[f"{side}_eligibility"] != "unknown" and not refs:
            raise ValueError("Eligibility assertion requires evidence")
    if result["preference"] != "insufficient":
        if not result["a_evidence"] or not result["b_evidence"] or not result["reason"].strip():
            raise ValueError("A winner or supported tie needs evidence on both sides")
    elif not result["missing_question"]:
        raise ValueError("Insufficient requires a question or explicit 'no actionable question'")
    if result["preference"] == "a_better" and result["a_eligibility"] == "violated":
        raise ValueError("An ineligible A cannot win")
    if result["preference"] == "b_better" and result["b_eligibility"] == "violated":
        raise ValueError("An ineligible B cannot win")
    return result


def rerank(original, selected, comparisons):
    expected = set(itertools.permutations(selected, 2))
    actual = [(x["a"], x["b"]) for x in comparisons]
    if len(selected) != 5 or len(set(selected)) != 5 or set(actual) != expected or len(actual) != len(expected):
        raise ValueError("Exactly 20 directed comparisons of five candidates required")
    values = {(x["a"], x["b"]): x["result"]["preference"] for x in comparisons}
    rank = {i: 0 for i in selected}
    edges = []
    for a, b in itertools.combinations(selected, 2):
        pair = values[a, b], values[b, a]
        if pair == ("a_better", "b_better"):
            state, winner = "winner", a
        elif pair == ("b_better", "a_better"):
            state, winner = "winner", b
        else:
            state, winner = ("tie" if pair == ("tie", "tie") else "unresolved"), None
        if winner is not None:
            loser = b if winner == a else a
            rank[winner] += 1
            rank[loser] -= 1
        edges.append({"a": a, "b": b, "state": state, "winner": winner})
    order = sorted(selected, key=lambda i: (-rank[i], -original[i], i))
    scores = list(original)
    for i, score in zip(order, sorted((original[i] for i in selected), reverse=True), strict=True):
        scores[i] = score
    return scores, {
        "edges": edges,
        "order": order,
        "wins_minus_losses": {str(k): v for k, v in rank.items()},
        "probability_calibrated": False,
    }
