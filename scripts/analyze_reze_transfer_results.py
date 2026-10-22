"""Read-only checks of completed REZE transfers and MR.HiSum summary diagnostics."""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

BASE = Path(os.environ.get("ZEROSHOT_ARTIFACT_ROOT", "artifacts/zeroshot"))


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def describe(values):
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError("Nonfinite diagnostic value")
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p95": float(np.quantile(array, 0.95)),
    }


def verify_completed(name):
    root = BASE / name
    completion, evaluation = read_json(root / "complete.json"), read_json(root / "evaluation.json")
    if completion["evaluation_sha256"] != digest(root / "evaluation.json"):
        raise ValueError("Completed evaluation hash changed")
    verified = {
        "complete.json": digest(root / "complete.json"),
        "evaluation.json": digest(root / "evaluation.json"),
    }
    for key, value in evaluation.items():
        if key.endswith("_sha256"):
            artifact = root / (key.removesuffix("_sha256") + (".json" if key == "plan_sha256" else ".jsonl"))
            if key in {
                "plan_sha256",
                "per_video_metrics_sha256",
                "hd_predictions_sha256",
                "summary_predictions_sha256",
            }:
                if digest(artifact) != value:
                    raise ValueError(f"Artifact hash mismatch: {artifact}")
                verified[artifact.name] = value
    plan = read_json(root / "plan.json")
    details = read_rows(root / "per_video_metrics.jsonl")
    ids = [d["vid"] for d in details]
    if len(ids) != len(set(ids)) or set(ids) != set(plan["selected_ids"]):
        raise ValueError("Combined results do not cover the frozen video set")
    shard_ids = []
    for shard in plan["shards"]:
        sub = root / shard["directory"]
        complete = read_json(sub / "complete.json")
        result = read_json(sub / "evaluation.json")
        binding = read_json(sub / "binding.json")
        if complete["evaluation_sha256"] != digest(sub / "evaluation.json"):
            raise ValueError("Shard evaluation changed")
        if result["binding_sha256"] != digest(sub / "binding.json"):
            raise ValueError("Shard binding changed")
        if binding["selected_qids"] != shard["video_ids"]:
            raise ValueError("Shard membership changed")
        if {p.stem for p in (sub / "predictions").glob("*.json")} != set(shard["video_ids"]):
            raise ValueError("Missing or unexpected predictions")
        for filename, checksum in binding["source_sha256"].items():
            if digest(sub / "source_snapshot" / filename) != checksum:
                raise ValueError("Frozen source changed")
        for filename, checksum in binding["labels_sha256"].items():
            if digest(sub / "evaluation_inputs" / filename) != checksum:
                raise ValueError("Frozen labels changed")
        shard_ids.extend(shard["video_ids"])
    if len(shard_ids) != len(set(shard_ids)) or set(shard_ids) != set(ids):
        raise ValueError("Shard overlap or missing IDs")
    return evaluation, details, {"run": str(root), "verified_files": verified, "videos": len(ids)}


def tvsum_analysis(evaluation, rows):
    result = {
        "metrics_percent": evaluation["metrics_percent"],
        "cost": evaluation["cost"],
        "diagnostics": evaluation["diagnostics"],
        "per_category": evaluation["per_category"],
        "scopes": {},
    }
    for name, split in (("umt_val10", "val"), ("umt_train40", "train"), ("all50_zero_shot", None)):
        subset = [d for d in rows if split is None or d["umt_split"] == split]
        stats = {
            "n": len(subset),
            "mean_AP": float(np.mean([d["Top5_AP"] for d in subset])),
            "videos_above_random_expectation": sum(
                d["Top5_AP"] > d["random_expected_Top5_AP"] for d in subset
            ),
            "videos_above_constant": sum(d["Top5_AP"] > d["constant_Top5_AP"] for d in subset),
        }
        if name in evaluation["metrics_percent"]:
            assert abs(stats["mean_AP"] - evaluation["metrics_percent"][name]["Top5_AP"]["mean"]) < 1e-10
        result["scopes"][name] = stats
    return result


def mrhisum_analysis(evaluation, metrics):
    root = BASE / "reze_mrhisum_full1888_v2"
    sys.path.insert(0, str(root / "source_snapshot"))
    from evaluate import summary_metric

    summaries = {d["vid"]: d for d in read_rows(root / "summary_predictions.jsonl")}
    source_by_id = {}
    for index in (0, 1):
        for path in (root / f"shard_{index}/predictions").glob("*.json"):
            source_by_id[path.stem] = path
    if set(summaries) != set(source_by_id) or set(summaries) != {d["vid"] for d in metrics}:
        raise ValueError("Summary and raw curve scopes disagree")
    diagnostics = []
    with h5py.File(root / "shard_0/evaluation_inputs/mr_hisum.h5") as source:
        for index, row in enumerate(metrics, 1):
            vid = row["vid"]
            group = source[vid]
            gt = group["gtscore"][:]
            cps = group["change_points"][:]
            gt_summary = group["gt_summary"][:]
            pred = np.asarray(read_json(source_by_id[vid])["second_scores"])
            if len(pred) != len(gt) or len(gt) != len(gt_summary):
                raise ValueError("Annotation and prediction grids disagree")
            # Label-aware diagnostic ONLY: test whether the official GT scores
            # passed through the existing decoder reconstruct released gt_summary.
            oracle = summary_metric(gt, cps, gt_summary)
            oracle_mask = np.pad(oracle["summary"], (0, len(gt_summary) - len(oracle["summary"])))
            weights = cps[:, 1] - cps[:, 0]
            boundaries = np.concatenate(([0], np.cumsum(weights)))
            entry = {
                "vid": vid,
                "oracle_decoder_F1": oracle["F1"],
                "oracle_reconstructs_gt_mask": bool(np.array_equal(oracle_mask, gt_summary.astype(int))),
                "mean_all_shot_seconds": float(weights.mean()),
                "prediction_coefficient_variation": float(pred.std() / (pred.mean() + 1e-12)),
                "gt_coefficient_variation": float(gt.std() / (gt.mean() + 1e-12)),
            }
            for method, mask in (
                ("reze", summaries[vid]["predicted_summary"]),
                ("constant", summaries[vid]["constant_summary"]),
                ("gt", gt_summary),
            ):
                mask = np.asarray(mask, dtype=int)
                selected = []
                for shot, length in enumerate(weights):
                    if length == 0:
                        continue
                    contents = mask[boundaries[shot] : boundaries[shot + 1]]
                    if not ((contents == 0).all() or (contents == 1).all()):
                        raise ValueError("Mask does not match the frozen shot grid")
                    if contents[0] == 1:
                        selected.append(int(length))
                entry[method + "_budget_fraction"] = float(mask.sum() / len(gt))
                entry[method + "_selected_shot_count"] = len(selected)
                entry[method + "_mean_selected_shot_seconds"] = float(np.mean(selected)) if selected else 0.0
            diagnostics.append(entry)
            if index % 400 == 0:
                print(f"Checked summary decoder {index}/{len(metrics)}", flush=True)
    aggregate = {
        k: describe([d[k] for d in diagnostics])
        for k in diagnostics[0]
        if k not in {"vid", "oracle_reconstructs_gt_mask"}
    }
    aggregate["oracle_exact_gt_mask_count"] = sum(d["oracle_reconstructs_gt_mask"] for d in diagnostics)
    return {
        "metrics_percent": evaluation["metrics_percent"],
        "cost": evaluation["cost"],
        "diagnostics": evaluation["diagnostics"],
        "summary_analysis": aggregate,
        "warning": "GT-score decoder audit uses labels and is not a method result; diagnostics are post hoc",
        "per_video_summary_diagnostics": diagnostics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve the previous analysis")
    tv, tv_rows, tv_check = verify_completed("reze_tvsum_full50_v1")
    mr, mr_rows, mr_check = verify_completed("reze_mrhisum_full1888_v2")
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_calls": 0,
        "original_predictions_modified": False,
        "script_sha256": digest(__file__),
        "completion_verification": {"tvsum": tv_check, "mrhisum": mr_check},
        "tvsum": tvsum_analysis(tv, tv_rows),
        "mrhisum": mrhisum_analysis(mr, mr_rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as target:
        json.dump(result, target, ensure_ascii=False, indent=2, allow_nan=False)
        target.write("\n")
    print(
        json.dumps(
            {"output": str(args.output), "summary_analysis": result["mrhisum"]["summary_analysis"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
