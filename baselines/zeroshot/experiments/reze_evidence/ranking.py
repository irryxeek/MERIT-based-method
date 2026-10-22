"""Prespecified candidate selection, position-symmetric preferences, and score permutation."""

import itertools
import math


def candidates(scores, k=5):
    if not scores or not all(math.isfinite(s) for s in scores):
        raise ValueError("Invalid baseline scores")
    return sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k]


def ordered_pairs(selected):
    return [(i, j) for i, j in itertools.combinations(selected, 2) for i, j in ((i, j), (j, i))]


def preference(pij, pji):
    if not all(math.isfinite(p) and 0 <= p <= 1 for p in (pij, pji)):
        raise ValueError("Invalid preference probability")
    supported = (pij >= 0.6 and pji <= 0.4) or (pji >= 0.6 and pij <= 0.4)
    return pij - pji if supported else 0.0


def make_scores(original, selected, records, arm):
    if selected != candidates(original):
        raise ValueError("Unexpected candidates")
    expected = ordered_pairs(selected)
    pairs = [(r["a"], r["b"]) for r in records]
    if len(pairs) != len(set(pairs)) or set(pairs) != set(expected):
        raise ValueError("Missing or duplicated directed comparisons")
    values = {(r["a"], r["b"]): r["pyes"] for r in records}
    if not all(math.isfinite(p) and 0 <= p <= 1 for p in values.values()):
        raise ValueError("Invalid raw probability")
    edges = {(i, j): preference(values[i, j], values[j, i]) for i, j in expected}
    if arm == "P_abs":
        rank_values = {i: sum(values[i, j] for j in selected if j != i) / (len(selected) - 1) for i in selected}
    elif arm == "P_pair":
        rank_values = {i: sum(edges[i, j] for j in selected if j != i) for i in selected}
    else:
        raise ValueError(f"Unsupported arm: {arm}")
    order = sorted(selected, key=lambda i: (-rank_values[i], -original[i], i))
    scores = list(original)
    for index, value in zip(order, sorted((original[i] for i in selected), reverse=True), strict=True):
        scores[index] = value
    cycles = 0
    for i, j, k in itertools.combinations(selected, 3):
        if (edges[i, j] > 0 and edges[j, k] > 0 and edges[k, i] > 0) or (
            edges[i, j] < 0 and edges[j, k] < 0 and edges[k, i] < 0
        ):
            cycles += 1
    return scores, {
        "order": order,
        "rank_values": {str(i): rank_values[i] for i in selected},
        "neutral_unordered_pairs": sum(edges[i, j] == 0 for i, j in itertools.combinations(selected, 2)),
        "both_yes_pairs": sum(
            values[i, j] >= 0.6 and values[j, i] >= 0.6 for i, j in itertools.combinations(selected, 2)
        ),
        "cycles": cycles,
        "score_values_preserved": True,
    }
