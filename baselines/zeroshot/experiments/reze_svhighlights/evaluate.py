"""Separate CPU scorer: official SVH functions, strict coverage, paired baselines."""

import argparse
import json
import math
from pathlib import Path

from shared import atomic_json, load_run, load_video_chunks, sha256, svh_scores, utc_now

# isort: split
# shared loads the pinned vendor directory.
import svhighlights_eval as official


def strict_predictions(predictions, labels, expected_ids):
    if len(predictions) != len(expected_ids) or {r["vid"] for r in predictions} != set(expected_ids):
        raise ValueError("Prediction IDs must exactly match the frozen scope")
    for row in predictions:
        scores = row["pred_saliency_scores"]
        if len(scores) != len(labels[row["vid"]]) or not scores:
            raise ValueError("Prediction/label lengths must match; no padding or truncation")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in scores):
            raise ValueError("Invalid scores")


def metrics(predictions, labels):
    by_id = {r["vid"]: r["pred_saliency_scores"] for r in predictions}
    return {
        "HL-mAP": official.compute_hl_ap(by_id, labels, num_workers=1),
        "HL-Hit1": official.compute_hl_hit1(by_id, labels),
        "HL-Hitk": official.compute_hl_hitk(by_id, labels),
        "HL-IoU": official.compute_hl_iou(by_id, labels),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run
    binding, rows, binding_hash = load_run(run)
    ids = binding["selected_ids"]
    predictions = []
    for row in rows:
        description = json.loads((run / "media" / f"{row['vid']}.json").read_text())
        records, _ = load_video_chunks(run, row, binding_hash, description["segments"], binding["batch_size"])
        scores = svh_scores([r["pyes"] for r in records], description["segments"], row["duration"], row["n_clips"])
        stored = json.loads((run / "predictions" / f"{row['vid']}.json").read_text())
        if stored != {"vid": row["vid"], "pred_saliency_scores": scores}:
            raise ValueError("Predictions disagree with raw log probability checkpoints")
        predictions.append(stored)
    label_path = Path(binding["labels_path_for_separate_evaluation"])
    if sha256(label_path) != binding["labels_sha256"]:
        raise ValueError("Official labels changed")
    labels = {r["vid"]: r["saliency_scores"] for r in json.loads(label_path.read_text())}
    strict_predictions(predictions, labels, ids)
    methods = {"reze_original": predictions}
    for name, source in binding["previous_predictions"].items():
        if sha256(source["path"]) != source["sha256"]:
            raise ValueError(f"Historical baseline changed: {name}")
        selected = [r for r in json.loads(Path(source["path"]).read_text()) if r["vid"] in ids]
        strict_predictions(selected, labels, ids)
        methods[name] = selected
    report = {
        "status": "complete",
        "evaluated_at": utc_now(),
        "binding_sha256": binding_hash,
        "scope": binding["scope"],
        "n_videos": len(ids),
        "selected_ids": ids,
        "official_all320_result": False,
        "missing_sport": "basketball",
        "metrics_percent": {name: metrics(preds, labels) for name, preds in methods.items()},
        "per_video": {},
    }
    for name, preds in methods.items():
        report["per_video"][name] = {r["vid"]: metrics([r], labels) for r in preds}
    summary = [json.loads((run / "video_summaries" / f"{vid}.json").read_text()) for vid in ids]
    report["cost"] = {
        key: sum(r[key] for r in summary)
        for key in [
            "model_requests",
            "frames",
            "input_tokens",
            "output_tokens",
            "decode_preprocess_seconds",
            "inference_seconds",
        ]
    }
    report["cost"]["charged_requests_including_interrupted_batches"] = sum(
        json.loads(p.read_text())["requests"] for p in (run / "request_attempts").glob("*.json")
    )
    report["limitations"] = [
        "Cross-dataset transfer of REZE scoring, not an SVHighlights result reported by the REZE paper.",
        "Seven videos selected for paired development comparison; no statistical significance claim.",
        "Qwen3-VL-8B-Instruct and scoring match local QV B0; final partial clip uses SVHighlights grid.",
        "REZE uses vision only; historical AV methods have different inputs and inference budgets.",
        "Generic sport highlight query follows the dataset, without sport-specific prompt tuning.",
    ]
    atomic_json(run / "predictions.json", predictions)
    atomic_json(run / "evaluation.json", report)
    lines = [
        "# REZE 原方法：SVHighlights 小规模迁移测试",
        "",
        f"完成时间：{report['evaluated_at']}。固定 7 场、7 类运动，共 {binding['source_content_hours']:.2f} 小时视频。",
        "这是固定开发子集的诊断结果，不能称为官方 320 场成绩；篮球原视频缺失。",
        "",
        "| 方法 | HL-mAP | HL-Hit1 | HL-Hitk | HL-IoU |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, values in report["metrics_percent"].items():
        lines.append("| " + name + " | " + " | ".join(f"{v:.2f}" for v in values.values()) + " |")
    lines += [
        "",
        "REZE：Qwen3-VL-8B-Instruct BF16，3 秒窗口、6 帧，原 Yes/No top-20 概率。",
        "保持原始逐秒曲线后取 2 秒均值；SVHighlights 末尾只有 1 秒时按该秒取分。",
        "比较采用完全相同的 7 场；既有 AV 方法与 REZE 视觉输入及预算不同。",
        "",
        f"完成模型请求 {report['cost']['model_requests']:,} 次，读帧 {report['cost']['frames']:,} 次。",
        "逐视频指标、原始概率与成本见 evaluation.json、chunks/ 和 video_summaries/。",
        "",
        "[官方指标实现](https://github.com/leedongkyu2019/SVHighlights/blob/main/eval.py)。",
    ]
    (run / "REPORT.md").write_text("\n".join(lines) + "\n")
    atomic_json(
        run / "complete.json",
        {
            "status": "complete",
            "finished_at": utc_now(),
            "binding_sha256": binding_hash,
            "evaluation_sha256": sha256(run / "evaluation.json"),
        },
    )
    atomic_json(
        run / "status.json",
        {
            "status": "complete",
            "updated_at": utc_now(),
            "completed_videos": 7,
            "metrics_percent": report["metrics_percent"],
            "cost": report["cost"],
        },
    )
    print(json.dumps(report["metrics_percent"]))


if __name__ == "__main__":
    main()
