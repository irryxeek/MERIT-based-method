"""Verify disjoint shards and separately report UMT val10 and all50 transfer."""

import argparse
import json
from collections import Counter
from pathlib import Path

from metrics import merge_details, summarize
from shared import HERE, atomic_json, read_jsonl, sha256, utc_now, write_jsonl

PROTOCOL_FIELDS = (
    "dataset",
    "query",
    "window_seconds",
    "sampling_fps",
    "frames_per_full_window",
    "readout",
    "timeline_policy",
    "evaluation_policy",
    "evaluator_revision",
    "model_path",
    "model_revision",
    "batch_size",
    "labels_sha256",
    "baseline_code_sha256",
    "source_sha256",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    root = args.run.resolve()
    if (root / "complete.json").exists():
        return
    plan = json.loads((root / "plan.json").read_text())
    for name, digest in plan["source_sha256"].items():
        if sha256(HERE / name) != digest:
            raise ValueError("Frozen aggregate source changed")
    states = []
    for shard in plan["shards"]:
        run = root / shard["directory"]
        state = json.loads((run / "status.json").read_text()) if (run / "status.json").exists() else {}
        states.append(
            {
                "index": shard["index"],
                "directory": shard["directory"],
                "state": state,
                "complete": (run / "complete.json").exists(),
                "stopped": (run / "stopped.json").exists(),
                "completed_videos": len(list((run / "predictions").glob("*.json"))),
            }
        )
    if not all(s["complete"] for s in states):
        atomic_json(
            root / "status.json",
            {
                "status": "shard_needs_attention" if any(s["stopped"] for s in states) else "in_progress",
                "updated_at": utc_now(),
                "expected_queries": 50,
                "completed_queries": sum(s["completed_videos"] for s in states),
                "shards": states,
            },
        )
        return
    details_parts, prediction_parts, evaluations, bindings, attempts = [], [], [], [], []
    for shard in plan["shards"]:
        run = root / shard["directory"]
        binding = json.loads((run / "binding.json").read_text())
        completion = json.loads((run / "complete.json").read_text())
        result = json.loads((run / "evaluation.json").read_text())
        if completion["binding_sha256"] != sha256(run / "binding.json"):
            raise ValueError("Completion binding changed")
        if completion["evaluation_sha256"] != sha256(run / "evaluation.json"):
            raise ValueError("Completed evaluation changed")
        if result["binding_sha256"] != sha256(run / "binding.json") or not result["author_metric_parity_passed"]:
            raise ValueError("Evaluation binding or parity check failed")
        for name in ("per_video_metrics", "hd_predictions"):
            if result[name + "_sha256"] != sha256(run / (name + ".jsonl")):
                raise ValueError("Evaluation artifact changed")
        if binding["selected_qids"] != shard["video_ids"] or binding["source_sha256"] != plan["source_sha256"]:
            raise ValueError("Shard membership or code differs from the plan")
        for name, digest in binding["source_sha256"].items():
            if sha256(run / "source_snapshot" / name) != digest:
                raise ValueError("Frozen shard source changed")
        if bindings and any(binding[k] != bindings[0][k] for k in PROTOCOL_FIELDS):
            raise ValueError("Cannot aggregate different protocols")
        bindings.append(binding)
        details_parts.append(merge_details(shard["video_ids"], [read_jsonl(run / "per_video_metrics.jsonl")]))
        prediction_parts.append(merge_details(shard["video_ids"], [read_jsonl(run / "hd_predictions.jsonl")]))
        if result["evaluated_videos"] != len(details_parts[-1]):
            raise ValueError("Shard evaluation coverage mismatch")
        evaluations.append(result)
        attempts.extend(json.loads((run / "worker_attempts.json").read_text()))
    details = merge_details(plan["selected_ids"], details_parts)
    predictions = merge_details(plan["selected_ids"], prediction_parts)
    val = [d for d in details if d["umt_split"] == "val"]
    if len(details) != 50 or {d["vid"] for d in val} != set(plan["umt_val_ids"]):
        raise ValueError("Expected all50 and exact author val10")
    metrics = {"umt_val10": summarize(val), "all50_zero_shot": summarize(details)}
    categories = sorted({d["category"] for d in details})
    per_category = {
        scope: {cat: summarize([d for d in subset if d["category"] == cat]) for cat in categories}
        for scope, subset in (("umt_val10", val), ("all50_zero_shot", details))
    }
    cost = {k: sum(e["cost"][k] for e in evaluations) for k in evaluations[0]["cost"]}
    cost["inference_wall_seconds_including_between_shard_queue"] = max(
        a["started_epoch"] + a["seconds"] for a in attempts
    ) - min(a["started_epoch"] for a in attempts)
    diagnostics = {}
    for key in ("answers", "score_modes"):
        counts = Counter()
        for result in evaluations:
            counts.update(result["diagnostics"][key])
        diagnostics[key] = dict(counts)
    for key in ("top5_boundary_tie_videos", "constant_prediction_videos"):
        diagnostics[key] = sum(e["diagnostics"][key] for e in evaluations)
    write_jsonl(root / "per_video_metrics.jsonl", details)
    write_jsonl(root / "hd_predictions.jsonl", predictions)
    result = {
        "status": "complete",
        "finished_at": utc_now(),
        "dataset": "TVSum",
        "evaluated_videos": 50,
        "primary_scope": "umt_val10",
        "secondary_scope": "all50_zero_shot",
        "metrics_percent": metrics,
        "per_category": per_category,
        "cost": cost,
        "diagnostics": diagnostics,
        "plan_sha256": sha256(root / "plan.json"),
        "per_video_metrics_sha256": sha256(root / "per_video_metrics.jsonl"),
        "hd_predictions_sha256": sha256(root / "hd_predictions.jsonl"),
        "not_original_REZE_paper_dataset_result": True,
        "no_training_or_prompt_tuning": True,
    }
    atomic_json(root / "evaluation.json", result)
    lines = [
        "# REZE / TVSum 高光检测结果",
        "",
        "Qwen3-VL-8B-Instruct，固定通用 query `Highlight of this video.`，3 秒窗口、2 fps、原 REZE 逐秒曲线及 2 秒均值。",
        "这是零样本任务迁移，不是 REZE 原论文 TVSum 成绩的严格复现。没有使用 TVSum 标签训练、选提示词或调参。",
        "",
        "| 范围 | 视频数 | Top-5 mAP (%) | 常数分数 | 随机排序精确期望 |",
        "|---|---:|---:|---:|---:|",
    ]
    for scope, n in (("umt_val10", 10), ("all50_zero_shot", 50)):
        m = metrics[scope]
        lines.append(
            f"| {scope} | {n} | {m['Top5_AP']['mean']:.4f} | {m['constant_Top5_AP']['mean']:.4f} | {m['random_expected_Top5_AP']['mean']:.4f} |"
        )
    lines += [
        "",
        "主指标使用 UMT 固定验证划分 10 条；全 50 条包含 UMT 的 40 条训练划分，仅作零样本全量迁移结果。",
        "20 名标注者分别按严格大于其中位数二值化，预测 top-5 内计算作者梯形 AP，再平均。保留 CPU float32 排序和并列处理。",
        "本地细粒度指标已与未改动的 UMT evaluate 方法体逐分片交叉核验。每视频宏平均、10,000 次视频 bootstrap（seed 42）见 evaluation.json。",
        "不能将全 50 条分数与监督方法的固定 10 条分数混用；当前只评高光检测，未计算摘要 F1。",
        "原始逐帧标注的两个尾帧差异保留；本次直接使用作者发布的 2 秒 HD 标签，不重采样或改写标签。",
        "",
        f"正式模型请求 {cost['formal_requests']}；含 smoke/失败保留请求 {cost['charged_requests_including_smoke_and_failed_attempts']}。",
        f"累计 worker 时间 {cost['worker_seconds'] / 60:.2f} 分钟；两分片首启至末完 {cost['inference_wall_seconds_including_between_shard_queue'] / 60:.2f} 分钟（可能含分片等待）。",
        "",
        "[UMT 评价与固定划分](https://github.com/TencentARC/UMT/tree/bde05e3de39049bf90d8807f6b2a4ed8b42acbb0/datasets)",
        "[TVSum 作者发布](https://github.com/yalesong/tvsum)",
        "[REZE Appendix A](https://arxiv.org/html/2608.04480v1)",
        "",
    ]
    (root / "REPORT.md").write_text("\n".join(lines))
    atomic_json(
        root / "complete.json",
        {
            "status": "complete",
            "finished_at": utc_now(),
            "evaluation_sha256": sha256(root / "evaluation.json"),
            "plan_sha256": sha256(root / "plan.json"),
        },
    )
    atomic_json(
        root / "status.json",
        {
            "status": "complete",
            "updated_at": utc_now(),
            "completed_queries": 50,
            "expected_queries": 50,
            "metrics_percent": {k: v["Top5_AP"]["mean"] for k, v in metrics.items()},
        },
    )
    print(json.dumps({"evaluated": 50, "metrics": metrics, "cost": cost}))


if __name__ == "__main__":
    main()
