"""Apply pinned MR.HiSum HD/summary metrics in a separate CPU-only process."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# The CPU subprocess starts fresh; it cannot depend on another entry point
# having imported shared and installed the pinned author helpers first.
sys.path.insert(0, str(Path(__file__).resolve().parent / "vendor"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from model.utils.evaluate_map import generate_mrhisum_seg_scores, top15_summary, top50_summary  # noqa: E402
from model.utils.evaluation_metrics import evaluate_summary  # noqa: E402
from model.utils.generate_summary import generate_summary  # noqa: E402
from shared import atomic_json, load_query_chunks, load_run, second_scores, sha256, utc_now, write_jsonl  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402


def expected_random_ap(n, positives):
    if not 0 < positives <= n:
        raise ValueError("Invalid AP positive count")
    if n == 1:
        return 1.0
    harmonic = sum(1 / rank for rank in range(1, n + 1))
    return (positives - 1) / (n - 1) + (n - positives) * harmonic / (n * (n - 1))


def hd_metrics(scores, ground_truth):
    pred, gt = torch.tensor(scores, dtype=torch.float32), torch.tensor(ground_truth, dtype=torch.float32)
    if pred.ndim != 1 or gt.shape != pred.shape or not torch.isfinite(pred).all() or not torch.isfinite(gt).all():
        raise ValueError("Invalid scores or annotation length")
    pred_shots = generate_mrhisum_seg_scores(pred, uniform_clip=5)
    gt_shots = generate_mrhisum_seg_scores(gt, uniform_clip=5)
    normalized = torch.exp(pred_shots) / (torch.exp(pred_shots).sum() + 1e-7)
    labels = {"50": top50_summary(gt_shots), "15": top15_summary(gt_shots)}
    result = {}
    order = torch.argsort(gt_shots, descending=True)
    for fraction, targets in labels.items():
        k = sum(targets)
        if k == 0:
            raise ValueError("No positive labels in native AP grid")
        result[f"AP{fraction}"] = 100 * float(average_precision_score(targets, normalized.numpy()))
        result[f"constant_AP{fraction}"] = 100 * k / len(targets)
        result[f"random_expected_AP{fraction}"] = 100 * expected_random_ap(len(targets), k)
        result[f"gt_positive_count_{fraction}"] = k
        result[f"gt_boundary_tie_{fraction}"] = bool(k < len(targets) and gt_shots[order[k - 1]] == gt_shots[order[k]])
    result.update(
        n_5s_shots=len(gt_shots),
        gt_5s_scores=gt_shots.tolist(),
        gt_binary=labels,
        pred_5s_scores=pred_shots.tolist(),
        official_normalized_scores=normalized.tolist(),
        unique_5s_scores=len(torch.unique(pred_shots)),
        unique_normalized_scores=len(torch.unique(normalized)),
    )
    return result


def summary_metric(scores, cps, gt_summary):
    n = len(scores)
    cps = np.asarray(cps, dtype=int)
    gt_summary = np.asarray(gt_summary)
    if gt_summary.shape != (n,) or cps.ndim != 2 or cps.shape[1] != 2:
        raise ValueError("Unexpected summary annotation structure")
    lengths = (cps[:, 1] - cps[:, 0]).tolist()
    # Keep author behavior exactly, including the shared endpoint convention.
    selected = generate_summary(
        np.asarray(scores, dtype=np.float32), cps, np.array([n]), lengths, np.arange(n), proportion=0.15
    )
    if len(selected) > n:
        raise ValueError("Author keyshot summary extends past the annotation timeline")
    # Author F1 already pads S to max(len(pred), len(GT)). Apply the same padding
    # before its unused Spearman/Kendall branch, which otherwise crashes at N-1/N.
    padded = np.pad(selected, (0, n - len(selected)))
    f1, _, _ = evaluate_summary(padded, gt_summary[None, :], eval_method="avg")
    return {
        "F1": float(f1),
        "prediction_length": len(selected),
        "label_length": n,
        "selected_seconds": int(selected.sum()),
        "summary": selected.astype(int).tolist(),
    }


def bootstrap_mean(values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite evaluation metric")
    rng = np.random.default_rng(42)
    draws = rng.integers(0, len(values), size=(10000, len(values)))
    lo, hi = np.percentile(values[draws].mean(axis=1), [2.5, 97.5])
    return {"mean": float(values.mean()), "bootstrap_95ci": [float(lo), float(hi)]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    binding, rows, binding_hash = load_run(run)
    complete = json.loads((run / "inference_complete.json").read_text())
    if complete["binding_sha256"] != binding_hash or complete["queries"] != len(rows):
        raise ValueError("Inference is incomplete or belongs to another run")
    if {p.stem for p in (run / "predictions").glob("*.json")} != set(binding["selected_qids"]):
        raise ValueError("Prediction ID set differs from fixed sample")
    for name, expected in binding["labels_sha256"].items():
        if sha256(run / "evaluation_inputs" / name) != expected:
            raise ValueError("Frozen annotation bytes changed")
    import h5py

    torch.set_num_threads(2)
    details, summaries = [], []
    with h5py.File(run / "evaluation_inputs/mr_hisum.h5", "r") as h5:
        for row in rows:
            vid = row["vid"]
            prediction = json.loads((run / "predictions" / f"{vid}.json").read_text())
            if prediction["binding_sha256"] != binding_hash:
                raise ValueError("Prediction protocol changed")
            media = json.loads((run / "media" / f"{vid}.json").read_text())
            records, _ = load_query_chunks(run, row, binding_hash, media["segments"], binding["batch_size"])
            reconstructed = second_scores(
                [r["pyes"] for r in records], media["segments"], row["duration"], row["n_seconds"]
            )
            if prediction["second_scores"] != reconstructed:
                raise ValueError("Prediction differs from complete raw probability records")
            group = h5[vid]
            gt, cps, gt_summary = group["gtscore"][:], group["change_points"][:], group["gt_summary"][:]
            if len(gt) != row["n_seconds"]:
                raise ValueError("Annotation length differs from label-blind metadata")
            hd = hd_metrics(reconstructed, gt)
            summary = summary_metric(reconstructed, cps, gt_summary)
            constant = summary_metric([0.5] * len(gt), cps, gt_summary)
            details.append(
                {
                    "vid": vid,
                    **hd,
                    "summary_F1": summary["F1"],
                    "constant_summary_F1": constant["F1"],
                    "summary_prediction_length": summary["prediction_length"],
                    "annotation_seconds": len(gt),
                    "summary_selected_seconds": summary["selected_seconds"],
                }
            )
            summaries.append(
                {"vid": vid, "predicted_summary": summary["summary"], "constant_summary": constant["summary"]}
            )
    names = (
        "AP50",
        "AP15",
        "summary_F1",
        "constant_AP50",
        "constant_AP15",
        "constant_summary_F1",
        "random_expected_AP50",
        "random_expected_AP15",
    )
    metrics = {name: bootstrap_mean([d[name] for d in details]) for name in names}
    for fraction in ("50", "15"):
        metrics[f"AP{fraction}_minus_random_expectation"] = bootstrap_mean(
            [d[f"AP{fraction}"] - d[f"random_expected_AP{fraction}"] for d in details]
        )
    requests = [json.loads(p.read_text()) for p in (run / "request_attempts").glob("*.json")]
    query_summaries = [json.loads((run / "query_summaries" / f"{r['vid']}.json").read_text()) for r in rows]
    answers, modes = Counter(), Counter()
    for item in query_summaries:
        answers.update(item["generated_answers"])
        modes.update(item["score_modes"])
    attempts = json.loads((run / "worker_attempts.json").read_text())
    cost = {
        "formal_requests": sum(q["model_requests"] for q in query_summaries),
        "charged_requests_including_smoke_and_failed_attempts": sum(r["requests"] for r in requests),
        "worker_seconds": sum(a["seconds"] for a in attempts),
        "gpu_worker_attempts": len(attempts),
        "formal_input_tokens": sum(q["input_tokens"] for q in query_summaries),
        "formal_frames": sum(q["frames"] for q in query_summaries),
        "formal_model_seconds": sum(q["inference_seconds"] for q in query_summaries),
        "formal_decode_seconds": sum(q["decode_preprocess_seconds"] for q in query_summaries),
    }
    result = {
        "status": "complete",
        "finished_at": utc_now(),
        "binding_sha256": binding_hash,
        "scope": binding["scope"],
        "evaluated_videos": len(rows),
        "official_test_videos": binding["official_test_count"],
        "official_test_coverage": len(rows) / binding["official_test_count"],
        "all_verified_media_selected": binding.get("all_verified_media_selected", False),
        "excluded_status_counts": binding["excluded_status_counts"],
        "metrics_percent": metrics,
        "cost": cost,
        "diagnostics": {
            "answers": dict(answers),
            "score_modes": dict(modes),
            "gt_boundary_tie_videos": {f: sum(d[f"gt_boundary_tie_{f}"] for d in details) for f in ("50", "15")},
            "constant_normalized_prediction_videos": sum(d["unique_normalized_scores"] == 1 for d in details),
            "summary_length_differences": dict(
                Counter(d["summary_prediction_length"] - d["annotation_seconds"] for d in details)
            ),
        },
        "evaluator_revision": binding["evaluator_revision"],
        "labels_sha256": binding["labels_sha256"],
        "not_full_test_result": len(rows) != binding["official_test_count"],
        "not_original_REZE_paper_dataset_result": True,
    }
    write_jsonl(run / "per_video_metrics.jsonl", details)
    write_jsonl(run / "summary_predictions.jsonl", summaries)
    result["per_video_metrics_sha256"] = sha256(run / "per_video_metrics.jsonl")
    result["summary_predictions_sha256"] = sha256(run / "summary_predictions.jsonl")
    atomic_json(run / "evaluation.json", result)
    lines = [
        f"# REZE on MR.HiSum: {binding['scope']}",
        "",
        f"Completed: {len(rows)} videos; official test split has {binding['official_test_count']} videos. Query: `{binding['query']}`.",
        "",
        "This is the local QV B0 scorer transferred to a query-free task, with Qwen3-VL-8B-Instruct.",
        f"Selection: {binding['selection_rule']}; verified pool: {binding['media_eligible_count']} videos. No labels selected videos or prompts.",
        f"Official test coverage: {100 * len(rows) / binding['official_test_count']:.2f}%. Media exclusions: {binding['excluded_status_counts']}.",
        "",
        "| Method / diagnostic | mAP50 (%) | mAP15 (%) | Summary F1 (%) |",
        "|---|---:|---:|---:|",
        f"| REZE | {metrics['AP50']['mean']:.4f} | {metrics['AP15']['mean']:.4f} | {metrics['summary_F1']['mean']:.4f} |",
        f"| Constant scores | {metrics['constant_AP50']['mean']:.4f} | {metrics['constant_AP15']['mean']:.4f} | {metrics['constant_summary_F1']['mean']:.4f} |",
        f"| Uniform random ranking, exact expected AP | {metrics['random_expected_AP50']['mean']:.4f} | {metrics['random_expected_AP15']['mean']:.4f} | — |",
        "",
        "## Evaluation and limitations",
        "",
        "Pinned author helpers use 5-second means, top-50%/15% GT labels, float32 exponent normalization and sklearn AP.",
        "CPU torch.argsort tie behavior is retained and boundary ties are counted in evaluation.json.",
        "Summary F1 preserves author KTS weights (end-start) and the evaluator padding for the one-second short vector.",
        "No prompt tuning, memory, reranking, training or new visual verification was added.",
        "The fixed generic query is an interface choice for this dataset and may limit the original action-verification prompt.",
        "Download and duration checks do not prove perfect semantic alignment with the historical YouTube video.",
        "Metrics cover only the selected available videos and must be reported with their official-test coverage.",
        "Incomplete official-test coverage prevents direct comparison with published full-test scores.",
        "",
        "## Cost",
        "",
        f"{cost['formal_requests']} formal requests; {cost['charged_requests_including_smoke_and_failed_attempts']} charged including probes/retries.",
        f"Single-GPU worker wall time: {cost['worker_seconds'] / 60:.2f} minutes across {cost['gpu_worker_attempts']} attempt(s).",
        "",
        "## Sources",
        "",
        "[Official MR.HiSum repository](https://github.com/MRHiSum/MR.HiSum/tree/aeb667f71239887034c2974062a341d312762867)",
        "[REZE Appendix A](https://arxiv.org/html/2608.04480v1)",
        "",
    ]
    (run / "REPORT.md").write_text("\n".join(lines))
    atomic_json(
        run / "complete.json",
        {
            "status": "complete",
            "finished_at": utc_now(),
            "evaluation_sha256": sha256(run / "evaluation.json"),
            "binding_sha256": binding_hash,
        },
    )
    atomic_json(
        run / "status.json",
        {
            "status": "complete",
            "updated_at": utc_now(),
            "completed_queries": len(rows),
            "expected_queries": len(rows),
            "metrics_percent": {k: metrics[k]["mean"] for k in ("AP50", "AP15", "summary_F1")},
        },
    )
    print(json.dumps({"evaluated": len(rows), "metrics": metrics, "cost": cost}))


if __name__ == "__main__":
    main()
