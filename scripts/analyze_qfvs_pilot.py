"""Analyze frozen QFVS predictions with label-independent, equal-budget controls."""

import argparse
import hashlib
import importlib.metadata
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def semantic_weights(tags, reference):
    a = tags.astype(np.int32)
    b = a[reference]
    intersection = a @ b.T
    union = a.sum(axis=1)[:, None] + b.sum(axis=1)[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union > 0)


def fast_metrics(weights, selected):
    matrix = weights[selected]
    rows, columns = linear_sum_assignment(-matrix)
    total = float(matrix[rows, columns].sum())
    k, m = matrix.shape
    return {"P": 100 * total / k, "R": 100 * total / m, "F1": 200 * total / (k + m)}


def mean_metrics(values):
    return {key: float(np.mean([x[key] for x in values])) for key in ("P", "R", "F1")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--dense-tags", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--random-trials", type=int, default=1000)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Analysis already exists; do not overwrite it")
    sys.path.insert(0, str(args.run / "source_snapshot"))
    import evaluate
    import shared

    binding, rows, binding_hash = shared.load_run(args.run)
    tags = evaluate.official.load_videos_tag(str(args.run / "evaluation_inputs/Tags.mat"))
    names = [
        line.strip().strip("'") for line in (args.dense_tags / "Dictionary.txt").read_text().splitlines()
    ]
    previous = json.loads((args.run / "evaluation.json").read_text())
    original_f1 = {r["qid"]: r["metrics_percent"]["oracle"]["F1"] for r in previous["per_query"]}
    all_trials, results, checks = [], [], 0
    tag_differences = []
    for item in rows:
        index = int(item["vid"][1:]) - 1
        native_tags = tags[index]
        tag_path = args.dense_tags / item["vid"] / f"{item['vid']}.txt"
        text_tags = [set(line.strip().split(",")) for line in tag_path.read_text().splitlines()]
        reconstructed = np.array([[name in labels for name in names] for labels in text_tags])
        if reconstructed.shape != native_tags.shape:
            raise ValueError("Dictionary/text tag dimensions differ from the frozen evaluation matrix")
        columns = [names.index(name) for name in item["concepts"]]
        if not np.array_equal(reconstructed[:, columns], native_tags[:, columns]):
            raise ValueError("Selected query concept columns differ between text tags and frozen MAT")
        for shot, column in np.argwhere(reconstructed != native_tags):
            tag_differences.append(
                {
                    "vid": item["vid"],
                    "shot_index_zero_based": int(shot),
                    "concept": names[column],
                    "dense_text": int(reconstructed[shot, column]),
                    "frozen_mat": int(native_tags[shot, column]),
                }
            )
        prediction_path = args.run / "predictions" / f"{item['qid']}.json"
        prediction = json.loads(prediction_path.read_text())
        scores = np.array(prediction["shot_scores"])
        selected = prediction["selected_shot_indices_zero_based"]
        n, k = len(scores), len(selected)
        uniform = np.floor((np.arange(k) + 0.5) * n / k).astype(int).tolist()
        if len(set(uniform)) != k or selected != shared.select_summary(scores, binding["summary_fraction"]):
            raise ValueError("Selection or control budget differs")
        reference_path = args.run / "evaluation_inputs/oracle" / item["vid"] / f"{item['query_name']}.txt"
        reference = [int(line) - 1 for line in reference_path.read_text().split()]
        weights = semantic_weights(native_tags, reference)
        rng = np.random.default_rng(np.random.SeedSequence([42, index]))
        trials, verified_samples = [], [selected, uniform]
        for trial in range(args.random_trials):
            sample = rng.choice(n, size=k, replace=False).tolist()
            trials.append(fast_metrics(weights, sample))
            if trial == 0:
                verified_samples.append(sample)
        for sample in verified_samples:
            official = evaluate.semantic_metrics(sample, reference, tags, index)
            fast = fast_metrics(weights, sample)
            if any(abs(official[key] - fast[key]) > 1e-10 for key in official):
                raise ValueError("Assignment solver differs from frozen official evaluator")
            checks += 1
        reze = fast_metrics(weights, selected)
        if abs(reze["F1"] - original_f1[item["qid"]]) > 1e-10:
            raise ValueError("Original F1 does not reproduce")
        all_trials.append([trial["F1"] for trial in trials])
        relevant = np.any(native_tags[:, columns], axis=1)
        ordered = sorted(selected)
        runs, length = [], 0
        for i, shot in enumerate(ordered):
            if i and shot != ordered[i - 1] + 1:
                runs.append(length)
                length = 0
            length += 1
        runs.append(length)
        records = []
        description = json.loads((args.run / "media" / f"{item['vid']}.json").read_text())
        records, _ = shared.load_query_chunks(
            args.run, item, binding_hash, description["segments"], binding["batch_size"]
        )
        row = {
            "qid": item["qid"],
            "n_shots": n,
            "summary_shots": k,
            "oracle_shots": len(reference),
            "reze": reze,
            "uniform": fast_metrics(weights, uniform),
            "random_mean": mean_metrics(trials),
            "random_f1_std": float(np.std([x["F1"] for x in trials], ddof=1)),
            "random_f1_interval_2_5_97_5": np.quantile([x["F1"] for x in trials], [0.025, 0.975]).tolist(),
            "reze_minus_random_mean": reze["F1"] - np.mean([x["F1"] for x in trials]),
            "reze_minus_uniform": reze["F1"] - fast_metrics(weights, uniform)["F1"],
            "summary_budget_only_f1_upper_bound": 200 * min(k, len(reference)) / (k + len(reference)),
            "score_quantiles_0_50_90_98_99_100": np.quantile(scores, [0, 0.5, 0.9, 0.98, 0.99, 1]).tolist(),
            "score_at_selection_cutoff": float(min(scores[selected])),
            "window_answer_yes": sum(r["generated_text"].strip() == "Yes" for r in records),
            "window_answer_no": sum(r["generated_text"].strip() == "No" for r in records),
            "window_score_modes": sorted(set(r["score_mode"] for r in records)),
            "selected_runs": len(runs),
            "longest_selected_run_shots": max(runs),
            "selected_adjacent_pairs": k - len(runs),
            "unique_selected_tag_vectors": len(np.unique(native_tags[selected], axis=0)),
            "query_relevance": {
                "all_shots_fraction": float(relevant.mean()),
                "selected_fraction": float(relevant[selected].mean()),
                "oracle_fraction": float(relevant[reference].mean()),
                "uniform_fraction": float(relevant[uniform].mean()),
                "concept_counts": {
                    name: {
                        "all": int(native_tags[:, column].sum()),
                        "selected": int(native_tags[selected, column].sum()),
                        "oracle": int(native_tags[reference, column].sum()),
                    }
                    for name, column in zip(item["concepts"], columns, strict=True)
                },
            },
            "top_10_selected_examples": [
                {
                    "shot_index_zero_based": shot,
                    "start_seconds": shot * 5,
                    "score": float(scores[shot]),
                    "released_tags": [names[j] for j in np.flatnonzero(native_tags[shot])],
                }
                for shot in selected[:10]
            ],
            "input_sha256": {
                "prediction": digest(prediction_path),
                "oracle": digest(reference_path),
                "dense_text_tags": digest(tag_path),
            },
        }
        results.append(row)
        print(json.dumps(row), flush=True)
    macro_trials = np.array(all_trials).mean(axis=0)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run": str(args.run),
        "binding_sha256": binding_hash,
        "scope": "Post-hoc CPU-only analysis of the frozen four-query pilot; not full180 evaluation",
        "random_control": {
            "trials_per_query": args.random_trials,
            "seed_sequence": "[42, zero_based_video_index]",
            "sampling": "Uniform without replacement; same fixed per-video 2% budget",
            "interval_interpretation": "Random-selector variability on these four fixed queries, not a dataset CI",
        },
        "uniform_control": "floor((i + 0.5) * n_shots / k), i = 0..k-1; no query or label input",
        "oracle_control_metrics_percent": {
            "REZE": mean_metrics([r["reze"] for r in results]),
            "Uniform": mean_metrics([r["uniform"] for r in results]),
            "Random_mean": mean_metrics([r["random_mean"] for r in results]),
        },
        "random_macro_f1_interval_2_5_97_5": np.quantile(macro_trials, [0.025, 0.975]).tolist(),
        "official_evaluator_equivalence_checks": checks,
        "selected_query_concept_columns_match_dense_text_tags": True,
        "dense_text_vs_frozen_mat_differences": tag_differences,
        "evaluation_uses_unchanged_frozen_mat": True,
        "per_query": results,
        "script_sha256": digest(Path(__file__)),
        "versions": {name: importlib.metadata.version(name) for name in ["numpy", "scipy", "networkx"]},
        "model_calls_added": 0,
        "original_predictions_modified": False,
        "no_threshold_or_prompt_tuning": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary["oracle_control_metrics_percent"]), flush=True)


if __name__ == "__main__":
    main()
