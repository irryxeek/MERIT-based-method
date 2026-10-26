"""Compare two frozen full180 runs after inference; use bounded per-query memory."""

import argparse
import json
from pathlib import Path

import numpy as np
from adaptation import ACTION_MODE, CONCEPT_MODE, SUPPRESSION_RADIUS_SHOTS, temporal_summary
from compare_adaptation import checked_scores, coverage
from evaluate import mean_metrics, semantic_metrics
from pairing import check_paired_trace, paired_receipt
from shared import atomic_json, load_run, select_summary, sha256, utc_now

# isort: split
import qfvs_eval as official

CONDITIONS = ("B0_topk", "B0_temporal", "Concept_topk", "Concept_temporal", "Uniform")


def validate_pair(bindings, manifests):
    baseline, concept = bindings
    if baseline["prompt_mode"] != ACTION_MODE or concept["prompt_mode"] != CONCEPT_MODE:
        raise ValueError("Expected original B0 and the frozen concept prompt")
    for binding, rows in zip(bindings, manifests, strict=True):
        if binding["scope"] != "full180" or len(rows) != 180 or len({r["qid"] for r in rows}) != 180:
            raise ValueError("Full comparison requires all 180 unique queries")
        if any(sum(row["vid"] == vid for row in rows) != 45 for vid in ("P01", "P02", "P03", "P04")):
            raise ValueError("Full comparison requires exactly 45 queries per source video")
        if len(set(binding["previously_diagnosed_qids"])) != 4:
            raise ValueError("Development-query partition changed")
    for key in (
        "selected_qids",
        "labels_sha256",
        "model_path",
        "model_revision",
        "media_verification_sha256",
        "window_seconds",
        "sampling_fps",
        "frames_per_full_window",
        "batch_size",
        "readout",
        "summary_fraction",
        "evaluator_revision",
        "previously_diagnosed_qids",
        "pixel_reference_sha256",
    ):
        if baseline[key] != concept[key]:
            raise ValueError(f"Paired protocol differs: {key}")
    if concept["selector_ablation"]["suppression_radius_shots"] != SUPPRESSION_RADIUS_SHOTS:
        raise ValueError("Frozen temporal suppression radius changed")
    for left, right in zip(*manifests, strict=True):
        if {k: v for k, v in left.items() if k != "query"} != {k: v for k, v in right.items() if k != "query"}:
            raise ValueError("Query identities or video inputs differ")
    if not set(baseline["previously_diagnosed_qids"]) <= {row["qid"] for row in manifests[0]}:
        raise ValueError("Development queries are missing from the full denominator")


def summarize(rows):
    return {
        condition: {
            ref: mean_metrics([row["conditions"][condition]["metrics_percent"][ref] for row in rows])
            for ref in ("oracle", "user_mean")
        }
        for condition in CONDITIONS
    }


def grouped_results(rows, development_qids):
    groups = {
        "full180": rows,
        "remaining176": [row for row in rows if row["qid"] not in development_qids],
        "previously_diagnosed4": [row for row in rows if row["qid"] in development_qids],
    }
    if [len(group) for group in groups.values()] != [180, 176, 4]:
        raise ValueError("Full / development-excluded denominators changed")
    return {
        name: {"n_queries": len(group), "macro_metrics_percent": summarize(group)} for name, group in groups.items()
    }


def compare(runs):
    loaded = [load_run(run) for run in runs]
    bindings, manifests, hashes = zip(*loaded, strict=True)
    validate_pair(bindings, manifests)
    evaluations = []
    for run, binding_hash in zip(runs, hashes, strict=True):
        done = json.loads((run / "complete.json").read_text())
        if done["status"] != "complete" or done["binding_sha256"] != binding_hash:
            raise ValueError("Cannot compare incomplete inference/evaluation")
        if done["evaluation_sha256"] != sha256(run / "evaluation.json"):
            raise ValueError("Evaluation receipt changed")
        evaluations.append(json.loads((run / "evaluation.json").read_text()))
    references = [paired_receipt(run, binding) for run, binding in zip(runs, bindings, strict=True)]
    if references[0] != references[1] or references[0] is None:
        raise ValueError("Both runs must use the same verified video pixel reference")
    for name, digest in references[0]["files"].items():
        if sha256(Path(references[0]["run"]) / name) != digest:
            raise ValueError(f"Historical pixel reference changed: {name}")
    selections, windows = {}, 0
    for i, row in enumerate(manifests[0]):
        choices = {}
        for label, run, binding, rows, digest, reference in zip(
            ("B0", "Concept"), runs, bindings, manifests, hashes, references, strict=True
        ):
            # One query at a time: full180 must not retain 1.8M logprob records in RAM.
            checked, count = checked_scores(run, binding, [rows[i]], digest)
            scores, chunks = checked[row["qid"]]
            for chunk in chunks:
                check_paired_trace(reference, rows[i], chunk["start"], chunk["segments"], chunk["trace"])
            choices[label + "_topk"] = checked_selection = select_summary(scores, binding["summary_fraction"])
            choices[label + "_temporal"] = temporal_summary(scores, binding["summary_fraction"])
            if len(checked_selection) != int(row["n_shots"] * binding["summary_fraction"]):
                raise ValueError("Summary budget changed")
            windows += count
            del checked, scores, chunks
        n = row["n_shots"]
        k = int(n * bindings[0]["summary_fraction"])
        choices["Uniform"] = np.floor((np.arange(k) + 0.5) * n / k).astype(int).tolist()
        if any(len(set(choice)) != k for choice in choices.values()):
            raise ValueError("Unequal comparison budgets")
        selections[row["qid"]] = choices
        print(f"Verified inputs and scores: {i + 1}/180", flush=True)
    # All choices are fixed before loading evaluation values.
    label_root = runs[1] / "evaluation_inputs"
    for run, binding in zip(runs, bindings, strict=True):
        for name, digest in binding["labels_sha256"].items():
            if sha256(run / "evaluation_inputs" / name) != digest:
                raise ValueError("Frozen evaluation labels changed")
    tags = official.load_videos_tag(str(label_root / "Tags.mat"))
    dictionary = label_root / "Dictionary.txt"
    if sha256(dictionary) != bindings[1]["concept_dictionary_sha256"]:
        raise ValueError("Concept dictionary changed")
    names = [line.strip().strip("'") for line in dictionary.read_text().splitlines()]
    per_query = []
    originals = [{row["qid"]: row for row in ev["per_query"]} for ev in evaluations]
    for row in manifests[0]:
        video_index = int(row["vid"][1:]) - 1
        matrix = tags[video_index]
        if len(matrix) != row["n_shots"] or matrix.shape[1] != len(names):
            raise ValueError("Native grid / tag dictionary mismatch")
        relevant = np.any(matrix[:, [names.index(c) for c in row["concepts"]]], axis=1)
        refs = {
            ref: [
                int(value) - 1
                for value in (label_root / ref / row["vid"] / f"{row['query_name']}.txt").read_text().split()
            ]
            for ref in ("oracle", "user1", "user2", "user3")
        }
        conditions = {}
        for name, selected in selections[row["qid"]].items():
            metrics = {ref: semantic_metrics(selected, ids, tags, video_index) for ref, ids in refs.items()}
            metrics["user_mean"] = mean_metrics([metrics[f"user{i}"] for i in range(1, 4)])
            conditions[name] = {
                "metrics_percent": metrics,
                "selected_shot_indices_zero_based": selected,
                "coverage": coverage(selected, row["n_shots"]),
                "query_concept_shots": int(relevant[selected].sum()),
            }
        for condition, original in zip(("B0_topk", "Concept_topk"), originals, strict=True):
            for ref in ("oracle", "user_mean"):
                for metric in ("P", "R", "F1"):
                    old = original[row["qid"]]["metrics_percent"][ref][metric]
                    if abs(old - conditions[condition]["metrics_percent"][ref][metric]) > 1e-10:
                        raise ValueError("Standalone metric does not reproduce")
        per_query.append({"qid": row["qid"], "vid": row["vid"], "conditions": conditions})
    return {
        "status": "complete",
        "created_at": utc_now(),
        "baseline_binding_sha256": hashes[0],
        "concept_binding_sha256": hashes[1],
        "groups": grouped_results(per_query, set(bindings[0]["previously_diagnosed_qids"])),
        "per_video_metrics_percent": {
            vid: summarize([row for row in per_query if row["vid"] == vid]) for vid in ("P01", "P02", "P03", "P04")
        },
        "per_query": per_query,
        "verified_window_inputs_across_both_prompts": windows,
        "suppression_radius_seconds": SUPPRESSION_RADIUS_SHOTS * 5,
        "summary_fraction": 0.02,
        "selection_uses_labels": False,
        "hyperparameter_search": False,
        "score_generation_cost": dict(zip(("B0", "Concept"), [ev["cost"] for ev in evaluations], strict=True)),
        "limitations": [
            "Full180 includes four diagnosed queries; remaining176 excludes those queries but uses the same videos.",
            "Fixed settings selected before full validation; no training or metric-driven retuning.",
            "QFVS transfer of local REZE, not an originally published REZE benchmark result.",
            "Temporal suppression is a coverage proxy, not semantic deduplication.",
        ],
    }


def markdown(result):
    lines = [
        "# QFVS 全量验证：REZE 与概念提示适配",
        "",
        f"完成时间（UTC）：{result['created_at']}。4 条完整视频、180 个查询；所有条件采用相同输入帧和 2% 摘要预算。",
        "固定 3 秒评分窗口、2 fps、Qwen3-VL-8B-Instruct BF16；时间去重半径固定为 10 秒。",
        "",
        "180 查询包含此前诊断的 4 条；另报其余 176 条。176 条仍来自相同 4 条视频，不属于未见视频验证。",
    ]
    for name, group in result["groups"].items():
        lines += [
            "",
            f"## {name}（{group['n_queries']} 条）",
            "",
            "| 条件 | Oracle P | Oracle R | Oracle F1 | 用户均值 F1 |",
            "|---|---:|---:|---:|---:|",
        ]
        for condition, scores in group["macro_metrics_percent"].items():
            oracle = scores["oracle"]
            lines.append(
                f"| {condition} | {oracle['P']:.2f} | {oracle['R']:.2f} | {oracle['F1']:.2f} | {scores['user_mean']['F1']:.2f} |"
            )
    lines += [
        "",
        "## 成本与证据",
        "",
        f"逐窗口核对两组原始帧、时间戳、模型输入像素与原始概率，共 {result['verified_window_inputs_across_both_prompts']:,} 次窗口评分。",
        "时间去重与均匀选片均不增加模型调用。所有查询均保留；逐查询指标、各视频均值及成本见 comparison.json。",
        "完整查询验证采用冻结配置；未在本轮结果上搜索提示词、抑制半径或摘要预算。",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    args = parser.parse_args()
    root = args.experiment.resolve()
    config_path = root / "experiment.json"
    if sha256(config_path) != (root / "experiment.sha256").read_text().strip():
        raise ValueError("Frozen experiment configuration changed")
    config = json.loads(config_path.read_text())
    if (root / "complete.json").exists() or (root / "stopped.json").exists():
        return
    statuses, runs = {}, []
    for name in ("baseline", "concept"):
        spec = config["runs"][name]
        run = Path(spec["path"])
        if sha256(run / "binding.json") != spec["binding_sha256"]:
            raise ValueError("Run binding changed after launch")
        statuses[name] = json.loads((run / "status.json").read_text())
        runs.append(run)
    state = {"status": "running", "updated_at": utc_now(), "runs": statuses}
    if any((run / "stopped.json").exists() for run in runs):
        state["status"] = "stopped_incomplete"
    elif all((run / "complete.json").exists() for run in runs):
        state["status"] = "evaluating_comparison"
    atomic_json(root / "status.json", state)
    if state["status"] != "evaluating_comparison":
        return
    attempts_path = root / "aggregation_attempts.json"
    attempts = json.loads(attempts_path.read_text()) if attempts_path.exists() else []
    for old in attempts:
        if old["status"] == "running":
            old["status"] = "interrupted"
    if len(attempts) >= 3:
        stopped = {"status": "comparison_stopped_after_three_attempts", "updated_at": utc_now()}
        atomic_json(root / "stopped.json", stopped)
        atomic_json(root / "status.json", stopped)
        return
    attempt = {"number": len(attempts) + 1, "status": "running", "started_at": utc_now()}
    attempts.append(attempt)
    atomic_json(attempts_path, attempts)
    output = root / "comparison.json"
    if output.exists():
        result = json.loads(output.read_text())
        if (result["baseline_binding_sha256"], result["concept_binding_sha256"]) != tuple(
            config["runs"][name]["binding_sha256"] for name in ("baseline", "concept")
        ):
            raise ValueError("Comparison belongs to another run pair")
    else:
        result = compare(runs)
        atomic_json(output, result)
    (root / "REPORT.md").write_text(markdown(result))
    complete = {"status": "complete", "finished_at": utc_now(), "comparison_sha256": sha256(output), "queries": 180}
    atomic_json(root / "complete.json", complete)
    atomic_json(root / "status.json", complete)
    attempt.update(status="complete", finished_at=utc_now())
    atomic_json(attempts_path, attempts)


if __name__ == "__main__":
    main()
