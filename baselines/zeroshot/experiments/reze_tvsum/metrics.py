"""TVSum HD metrics, independently checked against the pinned UMT method body."""

import ast
import itertools
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
import torch
from shared import HERE


def ap_of_ranked_binary(label):
    """Author trapezoidal AP; positive denominator is inside the top-k prefix."""
    positives = sum(label)
    if positives == 0:
        return 0.0
    hits = ap = rec = 0.0
    prc = 1.0
    for j, gt in enumerate(label):
        hits += gt
        next_rec = hits / positives
        next_prc = hits / (j + 1)
        ap += (next_rec - rec) * (prc + next_prc) / 2
        rec, prc = next_rec, next_prc
    return ap


@lru_cache(maxsize=None)
def random_expected_ap(n, positives, k=5):
    """Exact expectation over ordered sampling without replacement; at most 32 patterns."""
    if not 0 <= positives <= n or not 1 <= k <= n:
        raise ValueError("Invalid population or cutoff")
    expectation = 0.0
    for pattern in itertools.product((0, 1), repeat=k):
        p, yes, no = 1.0, positives, n - positives
        for j, value in enumerate(pattern):
            count = yes if value else no
            if count == 0:
                p = 0.0
                break
            p *= count / (n - j)
            yes -= value
            no -= 1 - value
        expectation += p * ap_of_ranked_binary(pattern)
    return expectation


def two_second_scores(second_scores, expected_clips):
    scores = np.asarray(second_scores, dtype=float)
    if scores.ndim != 1 or len(scores) // 2 != expected_clips:
        raise ValueError("Curve does not match the author 2-second grid")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Invalid probabilities")
    # Same readout as reze_core.highlight_scores: discard incomplete final 2s bin.
    return scores[: 2 * expected_clips].reshape(expected_clips, 2).mean(axis=1).tolist()


def video_metrics(scores, annotations):
    pred = torch.tensor(scores, dtype=torch.float32)
    labels = torch.tensor(annotations, dtype=torch.float32)
    if pred.ndim != 1 or labels.shape != (len(pred), 20) or len(pred) < 5:
        raise ValueError("Expected >=5 clips and 20 annotators with matching length")
    if not torch.isfinite(pred).all() or not torch.isfinite(labels).all():
        raise ValueError("Nonfinite prediction or label")
    if torch.any((pred < 0) | (pred > 1)) or torch.any((labels < 1) | (labels > 5)):
        raise ValueError("Prediction or label outside its declared range")
    order = torch.argsort(pred, descending=True)
    constant_order = torch.argsort(torch.zeros_like(pred), descending=True)
    aps, controls, randoms, positive_counts = [], [], [], []
    for i in range(20):
        label = (labels[:, i] > labels[:, i].median()).to(torch.float32)
        positives = int(label.sum())
        aps.append(ap_of_ranked_binary(label[order][:5].tolist()))
        controls.append(ap_of_ranked_binary(label[constant_order][:5].tolist()))
        randoms.append(random_expected_ap(len(label), positives))
        positive_counts.append(positives)
    return {
        "Top5_AP": 100 * float(np.mean(aps)),
        "constant_Top5_AP": 100 * float(np.mean(controls)),
        "random_expected_Top5_AP": 100 * float(np.mean(randoms)),
        "annotator_AP": [100 * a for a in aps],
        "positive_clip_counts": positive_counts,
        "top5_indices": order[:5].tolist(),
        "top5_boundary_tie": bool(len(pred) > 5 and pred[order[4]] == pred[order[5]]),
        "unique_scores": len(torch.unique(pred)),
        "n_2s_clips": len(pred),
    }


def author_top5_map(predictions, annotations):
    """Execute the unchanged author's evaluate body without importing training dependencies."""
    path = HERE / "vendor/umt_tvsum.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TVSum")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "evaluate")
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    namespace = {"torch": torch, "nncore": SimpleNamespace(to_dict_of_list=lambda x: x)}
    exec(compile(module, str(path), "exec"), namespace)
    ids = list(predictions)
    fake = SimpleNamespace(label=annotations, get_video_id=lambda index: ids[index])
    blob = {"saliency": [[torch.tensor(predictions[vid], dtype=torch.float32)] for vid in ids]}
    return namespace["evaluate"](fake, blob)["mAP"] * 100


def bootstrap_mean(values):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Empty/nonfinite metric values")
    rng = np.random.default_rng(42)
    samples = rng.integers(0, len(values), size=(10000, len(values)))
    ci = np.percentile(values[samples].mean(axis=1), [2.5, 97.5])
    return {"mean": float(values.mean()), "bootstrap_95ci": ci.tolist()}


def summarize(details):
    metrics = {
        name: bootstrap_mean([d[name] for d in details])
        for name in ("Top5_AP", "constant_Top5_AP", "random_expected_Top5_AP")
    }
    metrics["Top5_AP_minus_random_expectation"] = bootstrap_mean(
        [d["Top5_AP"] - d["random_expected_Top5_AP"] for d in details]
    )
    return metrics


def merge_details(expected_ids, shard_details):
    combined = [row for part in shard_details for row in part]
    ids = [row["vid"] for row in combined]
    if len(ids) != len(set(ids)) or len(expected_ids) != len(set(expected_ids)) or set(ids) != set(expected_ids):
        raise ValueError("Expected exact disjoint coverage of frozen video IDs")
    by_id = {row["vid"]: row for row in combined}
    return [by_id[vid] for vid in expected_ids]
