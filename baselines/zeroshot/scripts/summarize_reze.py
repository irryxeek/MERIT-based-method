"""Archive completed REZE outputs and write an auditable same-val comparison."""

import argparse
import csv
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from common import ROOT, STORE, atomic_json, read_jsonl, sha256, utc_now, validate_scores
from reze_run_io import load_bound_run, read_record
from summarize import archive_run, paired_interval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    assert run == STORE / "runs/reze_val_v2"
    binding, manifest = load_bound_run(run)
    assert not binding["smoke"] and len(manifest) == 1550
    reze_path, reze_metrics, reze_completion = archive_run(run.name)
    destination = ROOT / "results" / run.name
    for name in ["record_receipt.json", "inference.jsonl", "launch_ledger.json", "source_resolution_audit.json"]:
        target = destination / name
        if target.exists():
            assert sha256(target) == sha256(run / name)
        else:
            shutil.copy2(run / name, target)
    protocol = json.loads((ROOT / "configs/protocol_v1.json").read_text())
    audit = json.loads((ROOT / "manifests/data_audit.json").read_text())
    baseline_summary = json.loads((ROOT / "results/summary.json").read_text())
    assert sha256(protocol["labels_path"]) == audit["label_sha256"] == baseline_summary["label_sha256"]
    for name, digest in audit["evaluator_hashes"].items():
        assert sha256(Path(protocol["evaluator_root"]) / "standalone_eval" / name) == digest
    labels = read_jsonl(protocol["labels_path"])
    assert [r["qid"] for r in manifest] == [r["qid"] for r in labels]
    sys.path.insert(0, protocol["evaluator_root"])
    from standalone_eval.eval import compute_ap_from_tuple, mk_gt_scores

    paths = {
        "REZE_Qwen3VL8B": reze_path,
        "CLIP": ROOT / "results/clip_val/predictions.jsonl",
        "VTG_GPT": ROOT / "results/vtg_gpt_val/predictions.jsonl",
        "MERIT_historical": Path(
            (
                os.environ.get("MERIT_ARTIFACT_ROOT", "artifacts/merit")
                + "/runs/merit_qvh_main_v1-15f5f6e4ee00/val_predictions.jsonl"
            )
        ),
    }
    baseline_keys = {"CLIP": "clip_val", "VTG_GPT": "vtg_gpt_val", "MERIT_historical": "merit_historical_val"}
    values, results = {}, {}
    for name, path in paths.items():
        rows = read_jsonl(path)
        predictions = {r["qid"]: r for r in rows}
        assert len(rows) == len(predictions) == len(labels)
        assert set(predictions) == {r["qid"] for r in labels}
        if name in baseline_keys:
            previous = baseline_summary["results"][baseline_keys[name]]
            assert sha256(path) == previous["prediction_sha256"]
            metrics = previous["metrics"]
        else:
            metrics = reze_metrics
        per_query = []
        for row in labels:
            pred = predictions[row["qid"]]
            assert pred["vid"] == row["vid"]
            if name != "MERIT_historical":
                assert pred["query"] == row["query"]
            scores = np.array(pred["pred_saliency_scores"])
            validate_scores(scores.tolist(), int(row["duration"] / 2))
            binary = (mk_gt_scores(row) >= 4).astype(float)
            ap = np.mean([compute_ap_from_tuple((0, a, binary[:, a], scores))[2] for a in range(3)])
            per_query.append([100 * ap, 100 * binary[int(np.argmax(scores))].max()])
        values[name] = np.asarray(per_query)
        expected = metrics["HL-min-VeryGood"]
        np.testing.assert_allclose(
            np.round(values[name].mean(axis=0), 2), [expected["HL-mAP"], expected["HL-Hit1"]], atol=1e-9
        )
        results[name] = {
            "predictions_path": str(path),
            "predictions_sha256": sha256(path),
            "metrics": metrics,
            "unrounded_VeryGood": dict(zip(["mAP", "Hit1"], values[name].mean(axis=0).tolist())),
        }

    groups = np.array([r["vid"].rsplit("_", 2)[0] for r in labels])
    comparisons = {
        f"REZE_minus_{name}": paired_interval(values["REZE_Qwen3VL8B"] - values[name], groups) for name in baseline_keys
    }
    runtime = json.loads((run / "runtime.json").read_text())
    engineering = STORE / "runs/reze_sustained_v3"
    numeric_comparison = json.loads((engineering / "numeric_comparison.json").read_text())
    for name in ["completion.json", "source_binding.json", "processor_equivalence.json", "numeric_comparison.json"]:
        target = destination / "engineering" / name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(engineering / name, target)
    receipt = json.loads((run / "record_receipt.json").read_text())
    raw_scores, query_stats, pixel_sizes, frame_counts = [], [], Counter(), Counter()
    binding_hash = sha256(run / "binding.json")
    for row in manifest:
        assert sha256(run / "records" / f"{row['qid']}.json") == receipt[str(row["qid"])]
        record = read_record(run, row, binding, binding_hash)
        media = record["media"]
        scores = [c["pyes"] for c in record["clips"]]
        raw_scores.extend(scores)
        pixel_sizes["x".join(map(str, media["resized_height_width"]))] += 1
        frame_counts.update(len(indices) for indices in media["frame_indices"])
        query_stats.append(
            {
                "qid": row["qid"],
                "clips": len(scores),
                "all_zero": all(s == 0 for s in scores),
                "all_hard_no": all(c["generated_text"] == "No" for c in record["clips"]),
                "constant_raw": len(set(scores)) == 1,
                "missing_tail_seconds": media["unobserved_tail_seconds"],
                "dropped_residual_seconds": media["dropped_residual_seconds"],
            }
        )
    media_audit = {
        "queries": len(manifest),
        "segments": len(raw_scores),
        "frames": runtime["frames"],
        "resized_shapes_by_query": dict(pixel_sizes),
        "frames_per_segment_counts": dict(frame_counts),
        "probability_quantiles": dict(
            zip(
                ["min", "p01", "p10", "median", "p90", "p99", "max"],
                np.quantile(raw_scores, [0, 0.01, 0.1, 0.5, 0.9, 0.99, 1]).tolist(),
            )
        ),
        "all_zero_queries": sum(s["all_zero"] for s in query_stats),
        "all_hard_no_queries": sum(s["all_hard_no"] for s in query_stats),
        "constant_raw_queries": sum(s["constant_raw"] for s in query_stats),
        "queries_with_unobserved_tail": sum(s["missing_tail_seconds"] > 0 for s in query_stats),
        "queries_with_discarded_residual": sum(s["dropped_residual_seconds"] > 0 for s in query_stats),
        "max_unobserved_tail_seconds": max(s["missing_tail_seconds"] for s in query_stats),
    }
    summary = {
        "created_at": utc_now(),
        "queries": len(labels),
        "video_segments": len({r["vid"] for r in labels}),
        "source_videos": len(set(groups)),
        "split": "validation",
        "label_sha256": audit["label_sha256"],
        "evaluator_hashes": audit["evaluator_hashes"],
        "results": results,
        "paired_comparisons": comparisons,
        "runtime": runtime,
        "engineering_numeric_comparison": numeric_comparison,
        "completion": reze_completion,
        "media_audit": media_audit,
        "bootstrap": {
            "cluster": "source video",
            "repeats": 10000,
            "seed": 42,
            "interval": "95% percentile, exploratory; no multiple-comparison adjustment",
        },
        "summary_script_sha256": sha256(__file__),
        "baseline_summary_sha256": sha256(ROOT / "results/summary.json"),
        "scope": "independent paper-appendix implementation; HD only; full-val comparison, not paper test-score matching",
    }
    atomic_json(ROOT / "results/reze_comparison.json", summary)
    atomic_json(destination / "media_audit.json", media_audit)
    with (ROOT / "results/reze_paired_per_query.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["qid", "vid", "source_video"] + [f"{n}_{m}" for n in values for m in ["mAP", "Hit1"]])
        for i, row in enumerate(labels):
            writer.writerow([row["qid"], row["vid"], groups[i]] + [v for n in values for v in values[n][i]])
    names = {
        "REZE_Qwen3VL8B": "REZE + Qwen3-VL-8B",
        "CLIP": "CLIP ViT-B/32",
        "VTG_GPT": "VTG-GPT",
        "MERIT_historical": "MERIT（历史主实验）",
    }
    table = ["| 方法 | 查询数 | VeryGood mAP | HIT@1 |", "|---|---:|---:|---:|"]
    for name, result in results.items():
        score = result["metrics"]["HL-min-VeryGood"]
        table.append(f"| {names[name]} | 1,550 | {score['HL-mAP']:.2f} | {score['HL-Hit1']:.2f} |")
    pairs = ["| REZE 对比 | ΔmAP（95% CI） | ΔHIT@1（95% CI） |", "|---|---:|---:|"]
    for name in baseline_keys:
        pair = comparisons[f"REZE_minus_{name}"]
        cells = []
        for metric in ["mAP", "Hit1"]:
            lower, upper = pair["ci95_percentage_points"][metric]
            cells.append(f"{pair['delta_percentage_points'][metric]:+.2f} [{lower:+.2f}, {upper:+.2f}]")
        pairs.append(f"| {names[name]} | {' | '.join(cells)} |")
    modes = runtime["score_modes"]
    fair = reze_metrics["HL-min-Fair"]
    good = reze_metrics["HL-min-Good"]
    report = f"""# REZE 复现实验结果（2026-10-19）

## 完成情况

REZE + Qwen3-VL-8B-Instruct 已完成 QVHighlights **val 全部 1,550 条查询**，覆盖 1,519 个视频片段、
474 个源视频。预测失败 {reze_completion["prediction_failures"]} 条，完整分母不删样本。
完成时间（UTC）：`{reze_completion["completed_at"]}`。
这是依据[论文附录](https://arxiv.org/html/2608.04480v1)的独立实现，详细配置与边界见
[复现协议](REZE_PROTOCOL.md)。本轮仅评估 HD，没有 MR 实验。

## 同集合主指标

{chr(10).join(table)}

REZE 的 Fair mAP / HIT@1 为 **{fair["HL-mAP"]:.2f} / {fair["HL-Hit1"]:.2f}**，
Good 为 **{good["HL-mAP"]:.2f} / {good["HL-Hit1"]:.2f}**。
所有分数使用相同 val 与官方评估器；历史 MERIT 的 9 条失败仍按原方案计入完整分母。
VTG-GPT 使用作者已发布的 caption 和 query 改写；模型、信息来源和计算预算未统一。

## 配对差值

{chr(10).join(pairs)}

单位为百分点；按 474 个源视频聚类进行 10,000 次配对 bootstrap，seed 42。
差值由未舍入的逐查询指标计算，可能与表中两位小数直接相减相差 0.01。
区间是探索性统计，不含提示词、实现差异和模型预训练等系统误差，也未做多重比较校正。
这些比较检验本地实现的表现，不证明与论文 test 成绩等价。

## 运行开销与审计

- 四张 RTX 4090（GPU 4–7）各自加载 BF16 原始权重，未量化、未训练。
- 全量运行墙钟时间 **{runtime["wall_seconds_this_launch"] / 60:.2f} 分钟**，含加载、读帧、推理、汇总和官方评估，
  不含前期开发与 smoke。四 worker 记录的运行时间之和为
  **{sum(w["elapsed_seconds"] for w in runtime["worker_runs"]) / 60:.2f} worker-min**，不能解释为纯 GPU 核函数时间。
- 共 **{runtime["clips"]:,} 次 3 秒窗口核验**，读取 **{runtime["frames"]:,} 帧**，
  输入 **{runtime["prompt_tokens"]:,} tokens**、输出 **{runtime["output_tokens"]:,} tokens**；无外部 API 调用。
- 模型批量生成调用累计 **{runtime["generate_seconds_sum"] / 60:.2f} 分钟**；该累计值跨四 worker，含框架处理开销。
- 本轮 v2 worker 失败记录 **{runtime["worker_failures"]} 次**。此前 v1 的 encoder cache OOM、早期设备编号和
  自动显存预估失败均保留在独立目录；v2 从全量重新运行，没有混合 v1 的局部预测。
- top-20 中 Yes/No 均存在：**{modes.get("two_way", 0):,}**；No 缺失：**{modes.get("no_missing_top20", 0):,}**；
  Yes 缺失并按论文取零：**{modes.get("yes_missing_top20", 0):,}**。
- 全部窗口硬回答 No 的查询 **{media_audit["all_hard_no_queries"]} 条**；原始概率全零
  **{media_audit["all_zero_queries"]} 条**，常数曲线 **{media_audit["constant_raw_queries"]} 条**。
  连续概率分数保留了硬回答之外的排序信息。
- 输入帧索引、实际时间戳、像素哈希、视频哈希及原始 top-20 logprobs 均逐查询保存；2 秒得分已由原始概率重算核对。

CLIP 本地全量耗时 16.79 分钟（单卡）；VTG-GPT 的 96.95 秒仅覆盖复用作者文本后的计算。
应分别比较墙钟、设备数及预计算成本，不能把这三个耗时直接当作同预算速度排名。

## 复现边界与产物

论文 Qwen3-VL-8B 的 **test 43.37 / 71.98** 与本轮 val 属于不同划分，不计算“复现差距”。
论文未完整指定帧偏移、残段帧数、分辨率、视频接口细节和模型 revision；本地选择已在全量运行前冻结。
本轮没有按验证集成绩修改提示词或超参数。
缓存修复前后的 1,035 对窗口，平均概率绝对差为 0.000713，最大 0.215123；51 对超过 1e-5。
输入像素与 prompt token 数一致，但 BF16 批处理调度并非逐位不变；该数值敏感性不计入上述 bootstrap 区间。
21 条持续检查的显存稳定在 21,603 MiB，验证记录已归档到 `results/reze_val_v2/engineering/`。

- 预测、指标和冻结代码：`results/reze_val_v2/`。
- 同集合完整对比：`results/reze_comparison.json`。
- 逐查询配对表：`results/reze_paired_per_query.csv`。
- 原始输入与概率记录：`{STORE}/runs/reze_val_v2/records/`。
- 完成凭证：该运行目录的 `completion.json`；预测 SHA256 为 `{reze_completion["predictions_sha256"]}`。

后续将本结果作为密集视觉核验参照。评估 memory 方案时，应同时报告精度、实际读帧量、
模型调用与 token 开销，进一步区分记忆机制的作用和增加视觉计算的作用。
"""
    (ROOT / "docs/REZE_RESULTS_2026-10-19.md").write_text(report)
    print(
        json.dumps({"status": "complete", "metrics": reze_metrics, "paired_comparisons": comparisons}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
