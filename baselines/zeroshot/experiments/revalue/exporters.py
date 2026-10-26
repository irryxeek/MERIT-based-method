"""User intervals and benchmark scores are deliberately separate products."""

from io_utils import legacy


def benchmark_prediction(row, scores, original, selected):
    legacy.validate_scores(scores, len(original))
    if sorted(scores) != sorted(original) or any(scores[i] != x for i, x in enumerate(original) if i not in selected):
        raise ValueError("Local HD export may only permute the original five scores")
    return {**{k: row[k] for k in ("qid", "vid", "query")}, "pred_saliency_scores": scores}


def user_delivery(selection, audit_ref):
    return {
        "status": selection["status"],
        "events": selection["events"],
        "unresolved": selection["unresolved"],
        "audit_ref": audit_ref,
        "benchmark_score_vector": None,
        "score_is_calibrated_probability": False,
    }
