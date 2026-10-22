"""Evaluate the frozen 2x2 QFVS prompt/selection ablation without tuning its parameters."""

import argparse
import json
from pathlib import Path

import numpy as np
from adaptation import CONCEPT_MODE, SUPPRESSION_RADIUS_SHOTS, temporal_summary
from evaluate import mean_metrics, semantic_metrics
from pairing import check_paired_trace, paired_receipt
from shared import atomic_json, load_query_chunks, load_run, qfvs_scores, read_jsonl, select_summary, sha256, utc_now

# isort: split
import qfvs_eval as official


def checked_scores(run, binding, rows, binding_hash):
    result, window_count = {}, 0
    for row in rows:
        description = json.loads((run / "media" / f"{row['vid']}.json").read_text())
        records, chunks = load_query_chunks(run, row, binding_hash, description["segments"], binding["batch_size"])
        scores = qfvs_scores([r["pyes"] for r in records], description["segments"], row["duration"], row["n_shots"])
        stored = json.loads((run / "predictions" / f"{row['qid']}.json").read_text())
        if stored["shot_scores"] != scores or stored["selected_shot_indices_zero_based"] != select_summary(scores):
            raise ValueError("Stored predictions do not reproduce from the committed probabilities")
        result[row["qid"]] = (scores, chunks)
        window_count += len(records)
    return result, window_count


def coverage(selected, n_shots):
    ordered = sorted(selected)
    lengths = [1]
    for left, right in zip(ordered, ordered[1:]):
        if right == left + 1:
            lengths[-1] += 1
        else:
            lengths.append(1)
    return {
        "selected_shots": len(selected),
        "contiguous_runs": len(lengths),
        "longest_run_shots": max(lengths),
        "adjacent_pairs": len(selected) - len(lengths),
        "occupied_time_deciles": len({min(9, int(i * 10 / n_shots)) for i in selected}),
    }


def compare(run, baseline_only=False):
    binding, rows, binding_hash = load_run(run)
    if binding.get("prompt_mode") != CONCEPT_MODE:
        raise ValueError("Expected the concept prompt adaptation binding")
    if binding["selector_ablation"]["suppression_radius_shots"] != SUPPRESSION_RADIUS_SHOTS:
        raise ValueError("Predeclared suppression radius changed")
    receipt = paired_receipt(run, binding)
    parent = Path(receipt["run"])
    for name, digest in receipt["files"].items():
        if sha256(parent / name) != digest:
            raise ValueError(f"Frozen B0 evidence changed: {name}")
    old_binding = json.loads((parent / "binding.json").read_text())
    old_rows = read_jsonl(parent / "inference.jsonl")
    sources = {"B0": checked_scores(parent, old_binding, old_rows, receipt["binding_sha256"])[0]}
    paired_windows = 0
    if not baseline_only:
        completed = json.loads((run / "complete.json").read_text())
        if completed["binding_sha256"] != binding_hash or completed["status"] != "complete":
            raise ValueError("Concept inference and evaluation must complete before comparison")
        if sha256(run / "evaluation.json") != completed["evaluation_sha256"]:
            raise ValueError("Concept evaluation receipt changed")
        scores, paired_windows = checked_scores(run, binding, rows, binding_hash)
        for row in rows:
            for chunk in scores[row["qid"]][1]:
                check_paired_trace(
                    receipt, row, chunk["start"], chunk["segments"], chunk["trace"], binding["batch_size"]
                )
        sources["Concept"] = scores
    # Selection is finished before opening semantic tags or reference shot IDs.
    selections = {}
    for row in rows:
        qid, n = row["qid"], row["n_shots"]
        k = int(n * binding["summary_fraction"])
        choices = {}
        for label, scores in sources.items():
            choices[label + "_topk"] = select_summary(scores[qid][0], binding["summary_fraction"])
            choices[label + "_temporal"] = temporal_summary(scores[qid][0], binding["summary_fraction"])
        choices["Uniform"] = np.floor((np.arange(k) + 0.5) * n / k).astype(int).tolist()
        if any(len(set(selection)) != k for selection in choices.values()):
            raise ValueError("A comparison condition changed the summary budget")
        selections[qid] = choices
    for name, digest in binding["labels_sha256"].items():
        if sha256(run / "evaluation_inputs" / name) != digest:
            raise ValueError(f"Frozen evaluation labels changed: {name}")
    tags = official.load_videos_tag(str(run / "evaluation_inputs/Tags.mat"))
    dictionary = run / "evaluation_inputs/Dictionary.txt"
    if sha256(dictionary) != binding["concept_dictionary_sha256"]:
        raise ValueError("Concept dictionary changed")
    names = [line.strip().strip("'") for line in dictionary.read_text().splitlines()]
    results = []
    for row in rows:
        matrix = tags[int(row["vid"][1:]) - 1]
        if matrix.shape[1] != len(names) or len(matrix) != row["n_shots"]:
            raise ValueError("Concept dictionary and native tag grid differ")
        relevant = np.any(matrix[:, [names.index(c) for c in row["concepts"]]], axis=1)
        conditions = {}
        for name, selected in selections[row["qid"]].items():
            metrics = {}
            for reference in ("oracle", "user1", "user2", "user3"):
                path = run / "evaluation_inputs" / reference / row["vid"] / f"{row['query_name']}.txt"
                indices = [int(value) - 1 for value in path.read_text().split()]
                metrics[reference] = semantic_metrics(selected, indices, tags, int(row["vid"][1:]) - 1)
            metrics["user_mean"] = mean_metrics([metrics[f"user{i}"] for i in range(1, 4)])
            conditions[name] = {
                "metrics_percent": metrics,
                "coverage": coverage(selected, row["n_shots"]),
                "query_concept_shots": int(relevant[selected].sum()),
                "query_concept_fraction": float(relevant[selected].mean()),
                "selected_shot_indices_zero_based": selected,
            }
        results.append({"qid": row["qid"], "conditions": conditions})
    original = json.loads((parent / "evaluation.json").read_text())
    for before, after in zip(original["per_query"], results, strict=True):
        if before["qid"] != after["qid"]:
            raise ValueError("Comparison order differs from the original report")
        for metric in ("P", "R", "F1"):
            value = after["conditions"]["B0_topk"]["metrics_percent"]["oracle"][metric]
            if abs(value - before["metrics_percent"]["oracle"][metric]) > 1e-10:
                raise ValueError("Original reported B0 metric does not reproduce")
    macro = {
        name: {
            reference: mean_metrics([r["conditions"][name]["metrics_percent"][reference] for r in results])
            for reference in ("oracle", "user_mean")
        }
        for name in results[0]["conditions"]
    }
    costs = {"B0": original["cost"]}
    if not baseline_only:
        costs["Concept"] = json.loads((run / "evaluation.json").read_text())["cost"]
    return {
        "status": "baseline_controls_complete_waiting_for_concept" if baseline_only else "complete",
        "created_at": utc_now(),
        "binding_sha256": binding_hash,
        "baseline_binding_sha256": receipt["binding_sha256"],
        "scope": "Fixed previously diagnosed pilot4; development ablation, not full180 or unseen confirmation",
        "summary_fraction": binding["summary_fraction"],
        "suppression_radius_shots": SUPPRESSION_RADIUS_SHOTS,
        "selection_uses_labels": False,
        "hyperparameter_search": False,
        "paired_windows_with_identical_input_trace": paired_windows,
        "temporal_selection_extra_model_calls": 0,
        "macro_metrics_percent": macro,
        "per_query": results,
        "score_generation_cost": costs,
        "source_sha256": {
            "comparison": sha256(Path(__file__)),
            "adaptation": sha256(Path(__file__).with_name("adaptation.py")),
        },
    }


def markdown(result):
    lines = [
        "# QFVS 概念提示与时间去重对照",
        "",
        f"状态：`{result['status']}`；更新：{result['created_at']}。",
        "固定原四查询、原输入帧、3 秒评分窗口、5 秒原生片段和 2% 摘要预算。",
        "概念提示修正动作核验与 Building 歧义；时间选择按分数贪心抑制中心相距不超过 10 秒的片段。",
        "四查询已用于诊断，本次属于开发消融；时间分散并不直接证明语义去重。",
        "",
        "| 条件 | Oracle P / % | Oracle R / % | Oracle F1 / % | 用户均值 F1 / % |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, values in result["macro_metrics_percent"].items():
        o = values["oracle"]
        lines.append(f"| {name} | {o['P']:.2f} | {o['R']:.2f} | {o['F1']:.2f} | {values['user_mean']['F1']:.2f} |")
    lines += [
        "",
        "| 查询 | " + " | ".join(result["macro_metrics_percent"]) + " |",
        "|---|" + "---:|" * len(result["macro_metrics_percent"]),
    ]
    for row in result["per_query"]:
        scores = [f"{c['metrics_percent']['oracle']['F1']:.2f}" for c in row["conditions"].values()]
        lines.append(f"| {row['qid']} | " + " | ".join(scores) + " |")
    lines += [
        "",
        "离线查询概念命中（仅作诊断，标签不参与选片）：",
        "",
        "| 查询 | " + " | ".join(result["macro_metrics_percent"]) + " |",
        "|---|" + "---:|" * len(result["macro_metrics_percent"]),
    ]
    for row in result["per_query"]:
        counts = [f"{c['query_concept_shots']}/{c['coverage']['selected_shots']}" for c in row["conditions"].values()]
        lines.append(f"| {row['qid']} | " + " | ".join(counts) + " |")
    lines += [
        "",
        f"已逐窗口验证配对帧索引、时间戳和像素哈希：{result['paired_windows_with_identical_input_trace']:,} 个窗口。",
        "时间选择复用分数，不增加模型调用；B0 成本为历史运行成本，Concept 成本单独记录。",
        "详细选择、覆盖、成本及来源哈希见同名 JSON。原 B0 报告和预测不修改。",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--baseline-only", action="store_true")
    args = parser.parse_args()
    stem = "comparison_baseline_only" if args.baseline_only else "comparison"
    output = args.run / (stem + ".json")
    if output.exists():
        raise FileExistsError("Comparison already exists; preserve the frozen result")
    result = compare(args.run.resolve(), args.baseline_only)
    atomic_json(output, result)
    (args.run / (stem + ".md")).write_text(markdown(result))
    print(json.dumps(result["macro_metrics_percent"]), flush=True)


if __name__ == "__main__":
    main()
