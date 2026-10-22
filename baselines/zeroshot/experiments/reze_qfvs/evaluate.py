"""Separate CPU evaluation with pinned QFVS semantic matching and fixed budgets."""

import argparse
import json
from pathlib import Path

import numpy as np
from shared import atomic_json, load_query_chunks, load_run, qfvs_scores, select_summary, sha256, utc_now

# isort: split
import qfvs_eval as official


def semantic_metrics(prediction, reference, tags, video_index):
    n = len(tags[video_index])
    for name, values in (("prediction", prediction), ("reference", reference)):
        if not values or len(set(values)) != len(values) or any(i < 0 or i >= n for i in values):
            raise ValueError(f"Invalid or duplicate {name} shot indices")
    # The released helper divides 0/0 if no semantic edge has positive weight.
    a, b = tags[video_index][prediction], tags[video_index][reference]
    if not np.any(a.astype(np.int32) @ b.astype(np.int32).T):
        return {"P": 0.0, "R": 0.0, "F1": 0.0}
    p, r, f = official.calculate_semantic_matching(prediction, reference, tags, video_index)
    if not np.isfinite([p, r, f]).all():
        raise ValueError("Nonfinite native metric")
    return {"P": 100 * float(p), "R": 100 * float(r), "F1": 100 * float(f)}


def mean_metrics(values):
    return {key: float(np.mean([item[key] for item in values])) for key in ("P", "R", "F1")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    binding, rows, binding_hash = load_run(run)
    receipt = json.loads((run / "inference_complete.json").read_text())
    if receipt["binding_sha256"] != binding_hash or receipt["queries"] != len(rows):
        raise ValueError("Inference is incomplete for this protocol")
    if {p.stem for p in (run / "predictions").glob("*.json")} != {r["qid"] for r in rows}:
        raise ValueError("Missing or unexpected predictions")
    predictions = []
    for row in rows:
        description = json.loads((run / "media" / f"{row['vid']}.json").read_text())
        records, _ = load_query_chunks(run, row, binding_hash, description["segments"], binding["batch_size"])
        scores = qfvs_scores([r["pyes"] for r in records], description["segments"], row["duration"], row["n_shots"])
        selected = select_summary(scores, binding["summary_fraction"])
        stored = json.loads((run / "predictions" / f"{row['qid']}.json").read_text())
        expected = {
            "qid": row["qid"],
            "vid": row["vid"],
            "query_name": row["query_name"],
            "shot_scores": scores,
            "selected_shot_indices_zero_based": selected,
            "selected_shot_ids_one_based": sorted(i + 1 for i in selected),
        }
        if stored != expected:
            raise ValueError("Prediction differs from raw probabilities or fixed selection")
        predictions.append(stored)
    # Only this CPU evaluation process opens and parses label values.
    for name, digest in binding["labels_sha256"].items():
        if sha256(run / "evaluation_inputs" / name) != digest:
            raise ValueError(f"Frozen label file changed: {name}")
    tags = official.load_videos_tag(str(run / "evaluation_inputs/Tags.mat"))
    per_query = []
    for row, prediction in zip(rows, predictions):
        video_index = int(row["vid"][1:]) - 1
        if len(tags[video_index]) != row["n_shots"]:
            raise ValueError("Native tag/score count mismatch")
        metrics = {}
        for reference in ("oracle", "user1", "user2", "user3"):
            path = run / "evaluation_inputs" / reference / row["vid"] / f"{row['query_name']}.txt"
            indices = [int(line.strip()) - 1 for line in path.read_text().splitlines() if line.strip()]
            metrics[reference] = semantic_metrics(
                prediction["selected_shot_indices_zero_based"], indices, tags, video_index
            )
        metrics["user_mean"] = mean_metrics([metrics[f"user{i}"] for i in range(1, 4)])
        per_query.append({"qid": row["qid"], "vid": row["vid"], "metrics_percent": metrics})
    summaries = [json.loads((run / "query_summaries" / f"{row['qid']}.json").read_text()) for row in rows]
    cost = {
        key: sum(s[key] for s in summaries)
        for key in [
            "model_requests",
            "frames",
            "input_tokens",
            "output_tokens",
            "decode_preprocess_seconds",
            "inference_seconds",
        ]
    }
    cost["charged_requests_including_smoke_and_interrupted"] = sum(
        json.loads(p.read_text())["requests"] for p in (run / "request_attempts").glob("*.json")
    )
    report = {
        "status": "complete",
        "evaluated_at": utc_now(),
        "binding_sha256": binding_hash,
        "scope": binding["scope"],
        "n_queries": len(rows),
        "n_videos": 4,
        "primary_oracle_percent": mean_metrics([q["metrics_percent"]["oracle"] for q in per_query]),
        "secondary_user_mean_percent": mean_metrics([q["metrics_percent"]["user_mean"] for q in per_query]),
        "per_query": per_query,
        "cost": cost,
        "per_video_oracle_percent": {
            vid: mean_metrics([q["metrics_percent"]["oracle"] for q in per_query if q["vid"] == vid])
            for vid in sorted({r["vid"] for r in rows})
        },
        "score_diagnostics": [
            {
                k: s[k]
                for k in (
                    "qid",
                    "zero_score_fraction",
                    "unique_score_count",
                    "score_modes",
                    "generated_answers",
                )
            }
            for s in summaries
        ],
        "limitations": [
            "Transfer of the local QV REZE baseline; REZE does not report this QFVS protocol.",
            "Two concept queries use OR wording; original strict action verifier prompt remains unchanged.",
            "Fixed 2% summary budget follows the pinned UniVTG default; ties use earlier shot indices.",
            "Primary oracle F1 and secondary mean-user F1 have different reference sets and must not be conflated.",
            "pilot4 is a diagnostic subset, not the complete 180-query benchmark.",
        ],
    }
    atomic_json(run / "predictions.json", predictions)
    atomic_json(run / "evaluation.json", report)
    lines = [
        "# REZE 在 QFVS 上的迁移测试",
        "",
        f"完成：{report['evaluated_at']}；范围：{binding['scope']}，4 条视频、{len(rows)} 个查询。",
        "使用 Qwen3-VL-8B-Instruct BF16，保留原 REZE 的 3 秒窗口与 Yes/No 概率评分。",
        "逐秒分数映射到原生 5 秒片段，选择最高分的 2% 作为摘要；这是跨任务适配。",
        "",
        "| 参考摘要 | 语义 Precision / % | 语义 Recall / % | 语义 F1 / % |",
        "|---|---:|---:|---:|",
    ]
    for label, key in (
        ("Oracle（主指标）", "primary_oracle_percent"),
        ("三位用户均值（辅助）", "secondary_user_mean_percent"),
    ):
        lines.append(f"| {label} | " + " | ".join(f"{report[key][k]:.2f}" for k in ("P", "R", "F1")) + " |")
    lines += [
        "",
        f"正式窗口评分 {cost['model_requests']:,} 次；包括检查与中断批次的计费请求 "
        f"{cost['charged_requests_including_smoke_and_interrupted']:,} 次。",
        "pilot4 的 4 个查询结果不能代表全部 180 查询成绩。逐查询指标和成本见 evaluation.json。",
        f"[原生语义指标](https://github.com/showlab/UniVTG/blob/{binding['evaluator_revision']}/eval/qfvs.py)。",
    ]
    (run / "REPORT.md").write_text("\n".join(lines) + "\n")
    atomic_json(
        run / "complete.json",
        {
            "status": "complete",
            "binding_sha256": binding_hash,
            "finished_at": utc_now(),
            "evaluation_sha256": sha256(run / "evaluation.json"),
        },
    )
    atomic_json(
        run / "status.json",
        {
            "status": "complete",
            "updated_at": utc_now(),
            "completed_queries": len(rows),
            "expected_queries": len(rows),
            "primary_oracle_percent": report["primary_oracle_percent"],
            "cost": cost,
        },
    )
    print(json.dumps(report["primary_oracle_percent"]), flush=True)


if __name__ == "__main__":
    main()
