"""Independent scorer. This is the only inference-project entry point that reads test GT."""

import _bootstrap  # noqa: F401 -- establish the isolated implementation import
import argparse
import contextlib
import io
import json
import math
import sys
from pathlib import Path

import numpy as np

from videoarm.qvhighlights.common import (
    ARTIFACTS,
    atomic_json,
    file_hash,
    load_manifest,
    read_jsonl,
    source_id,
    write_jsonl,
)
from videoarm.qvhighlights.contracts import validate_predictions


def official_module():
    repo = ARTIFACTS / "external/moment_detr"
    lock = json.loads((ARTIFACTS / "checks/source_lock.json").read_text())
    for rel, expected in lock["evaluator_sha256"].items():
        if file_hash(repo / rel) != expected:
            raise ValueError("official evaluator drift")
    sys.path.insert(0, str(repo))
    from standalone_eval import eval as evaluator

    return evaluator


def metric_keys(task):
    if task == "mr":
        return [
            "MR-full-mAP",
            "MR-full-mAP@0.5",
            "MR-full-mAP@0.75",
            "MR-full-R1@0.5",
            "MR-full-R1@0.7",
        ]
    return [
        f"HL-min-{name}-{metric}"
        for name in ("Fair", "Good", "VeryGood")
        for metric in ("mAP", "Hit1")
    ]


def per_query(predictions, gt, task, official):
    truth = {r["qid"]: r for r in gt}
    result = []
    for pred in predictions:
        r = truth[pred["qid"]]
        metrics = {}
        if task == "mr":
            qid = r["qid"]
            triples = [
                qid,
                [
                    {"video-id": qid, "t-start": w[0], "t-end": w[1]}
                    for w in r["relevant_windows"]
                ],
                [
                    {"video-id": qid, "t-start": w[0], "t-end": w[1], "score": w[2]}
                    for w in pred["pred_relevant_windows"]
                ],
            ]
            th = np.array([float(f"{t:.2f}") for t in np.linspace(0.5, 0.95, 10)])
            _, ap = official.compute_average_precision_detection_wrapper(
                triples, tiou_thresholds=th
            )
            candidates = np.array([pred["pred_relevant_windows"][0][:2]], dtype=float)
            cross = official.compute_temporal_iou_batch_cross(
                candidates, np.array(r["relevant_windows"])
            )[0]
            best_gt = np.array(
                [r["relevant_windows"][int(cross.argmax())]], dtype=float
            )
            # Preserve official R1's paired utility, including its existing definition.
            iou = official.compute_temporal_iou_batch_paired(candidates, best_gt)[0]
            metrics = {
                "MR-full-mAP": float(ap.mean()),
                "MR-full-mAP@0.5": float(ap[0]),
                "MR-full-mAP@0.75": float(ap[5]),
                "MR-full-R1@0.5": float(iou >= 0.5),
                "MR-full-R1@0.7": float(iou >= 0.7),
            }
        else:
            matrix = official.mk_gt_scores(r)
            scores = np.array(pred["pred_saliency_scores"])
            for threshold, name in ((2, "Fair"), (3, "Good"), (4, "VeryGood")):
                labels = (matrix >= threshold).astype(float)
                aps = [
                    official.compute_ap_from_tuple((0, w, labels[:, w], scores))[2]
                    for w in range(3)
                ]
                metrics[f"HL-min-{name}-mAP"] = float(np.mean(aps))
                metrics[f"HL-min-{name}-Hit1"] = float(
                    np.max(labels[int(np.argmax(scores))])
                )
        result.append(
            {"qid": r["qid"], "source_id": source_id(r["vid"])[0], "metrics": metrics}
        )
    return result


def clean_nonfinite(v):
    if isinstance(v, dict):
        return {k: clean_nonfinite(x) for k, x in v.items()}
    if isinstance(v, list):
        return [clean_nonfinite(x) for x in v]
    return None if isinstance(v, float) and not math.isfinite(v) else v


def evaluate_official(predictions, gt, task, official):
    if task == "hd":
        return official.eval_submission(
            predictions, gt, verbose=False, match_number=True
        )
    ranges = [
        ("short", [0, 10]),
        ("middle", [10, 30]),
        ("long", [30, 150]),
        ("full", [0, 150]),
    ]
    if all(
        official.get_data_by_range(predictions, gt, bounds)[1] for _, bounds in ranges
    ):
        return official.eval_submission(
            predictions, gt, verbose=False, match_number=True
        )
    # Original entry point crashes for an absent duration stratum. Call the SAME
    # official functions on each nonempty stratum, leaving absent strata undefined.
    result = {}
    for name, bounds in ranges:
        p, g = official.get_data_by_range(predictions, gt, bounds)
        result[name] = (
            {
                "MR-mAP": official.compute_mr_ap(p, g, num_workers=1),
                "MR-R1": official.compute_mr_r1(p, g),
            }
            if g
            else {"MR-mAP": None, "MR-R1": None}
        )
    full = result["full"]
    result["brief"] = {
        "MR-full-mAP": full["MR-mAP"]["average"],
        "MR-full-mAP@0.5": full["MR-mAP"]["0.5"],
        "MR-full-mAP@0.75": full["MR-mAP"]["0.75"],
        "MR-full-R1@0.5": full["MR-R1"]["0.5"],
        "MR-full-R1@0.7": full["MR-R1"]["0.7"],
    }
    for name in ("short", "middle", "long"):
        result["brief"][f"MR-{name}-mAP"] = (
            result[name]["MR-mAP"]["average"] if result[name]["MR-mAP"] else None
        )
    return result


def score(manifest, predictions, task, gt, *, partial=False):
    coverage = validate_predictions(predictions, manifest, task, allow_partial=partial)
    expected = {r["qid"]: r for r in manifest}
    truth = {r["qid"]: r for r in gt}
    if len(truth) != len(gt) or set(expected) - set(truth):
        raise ValueError("GT qid identity/coverage mismatch")
    for qid, row in expected.items():
        if any(truth[qid][key] != row[key] for key in ("query", "vid", "duration")):
            raise ValueError("GT/inference manifest binding mismatch")
    if not predictions:
        return {
            "coverage": coverage,
            "task": task,
            "official_metrics": None,
            "status": "no_successful_predictions",
            "custom_full_plan_failure_zero_metrics": {
                key: 0.0 for key in metric_keys(task)
            },
            "custom_policy": "all planned queries failed; per-query failure metrics are zero; no official submission or fabricated prediction",
        }, []
    subset = [truth[r["qid"]] for r in predictions]
    official = official_module()
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        metrics = evaluate_official(predictions, subset, task, official)
    individual = per_query(predictions, subset, task, official)
    for key in individual[0]["metrics"]:
        aggregate = 100 * np.mean([r["metrics"][key] for r in individual])
        if abs(round(aggregate, 2) - metrics["brief"][key]) > 0.011:
            raise ValueError(
                f"per-query aggregation differs from official evaluator: {key}"
            )
    conservative = {
        k: 100 * sum(r["metrics"][k] for r in individual) / len(manifest)
        for k in individual[0]["metrics"]
    }
    return {
        "status": "complete_official_submission"
        if not coverage["missing_qids"]
        else "official_success_subset_only",
        "task": task,
        "coverage": coverage,
        "official_metrics": clean_nonfinite(metrics),
        "undefined_duration_groups": [
            k
            for k, v in metrics.items()
            if k != "brief"
            and (
                any(d is None for d in v.values())
                or any(
                    isinstance(x, float) and not math.isfinite(x)
                    for d in v.values()
                    if isinstance(d, dict)
                    for x in d.values()
                )
            )
        ],
        "custom_full_plan_failure_zero_metrics": conservative,
        "custom_policy": "sum successful official per-query unrounded metrics / original planned query count; missing queries contribute metric zero, never fabricated predictions",
        "official_log": output.getvalue(),
    }, individual


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--task", choices=["mr", "hd"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--gt")
    a = ap.parse_args()
    manifest = load_manifest(a.manifest)
    split = manifest[0]["split"]
    if split == "test" and not (ARTIFACTS / "checks/formal_freeze.json").exists():
        raise ValueError("test scoring requires the same formal protocol freeze")
    gt_path = (
        Path(a.gt)
        if a.gt
        else ARTIFACTS
        / "external/moment_detr/data"
        / (
            "highlight_test_with_gt.jsonl"
            if split == "test"
            else f"highlight_{split}_release.jsonl"
        )
    )
    result, individual = score(
        manifest,
        read_jsonl(a.predictions),
        a.task,
        read_jsonl(gt_path),
        partial=a.allow_partial,
    )
    result.update(
        manifest_sha256=file_hash(a.manifest),
        predictions_sha256=file_hash(a.predictions),
        gt_sha256=file_hash(gt_path),
        predictions_path=str(Path(a.predictions).resolve()),
        gt_path=str(gt_path.resolve()),
    )
    atomic_json(a.out, result)
    write_jsonl(Path(a.out).with_suffix(".per_query.jsonl"), individual)
    print(
        json.dumps(
            {
                "status": result["status"],
                "coverage": result["coverage"],
                "metrics": (result["official_metrics"] or {}).get("brief"),
            }
        )
    )


if __name__ == "__main__":
    main()
