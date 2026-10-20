"""Evaluate only complete stage A; inference never imports this module."""

import json
import sys
from pathlib import Path

import numpy as np
import protocol
from summarize import paired_interval

EVALUATOR = Path("source_dependencies/moment_detr")
LABELS = EVALUATOR / "data/highlight_train_release.jsonl"


def choose_arm(results, comparisons):
    qualifying = []
    checks = {}
    for arm in ("P_abs", "P_pair"):
        delta = comparisons[f"{arm}_minus_B0"]["delta_percentage_points"]
        check = {
            "mAP_gain_at_least_1pp": delta["mAP"] >= 1.0,
            "Hit1_drop_at_most_1pp": delta["Hit1"] >= -1.0 - 1e-9,
            "fallback_queries_at_most_1": results[arm]["fallback_queries"] <= 1,
            "cold_total_tokens_at_most_2_25x_B0": results[arm]["cold_total_token_ratio"] <= 2.25,
        }
        checks[arm] = check
        if all(check.values()):
            qualifying.append(arm)
    if len(qualifying) == 2:
        delta = comparisons["P_pair_minus_P_abs"]["delta_percentage_points"]
        winner = "P_pair" if delta["mAP"] >= 0.5 and delta["Hit1"] >= -1 - 1e-9 else "P_abs"
    else:
        winner = qualifying[0] if qualifying else None
    return {
        "checks": checks,
        "qualified_arms": qualifying,
        "selected_arm": winner,
        "decision": "proceed_to_stage_B" if winner else "stop_v1_no_qualifying_stage_A_arm",
    }


def evaluate(baseline_run, pair_run, destination):
    baseline_run, pair_run, destination = map(Path, (baseline_run, pair_run, destination))
    binding, rows = protocol.load_run(pair_run)
    if len(rows) != 100 or binding["split"] != "explore100":
        raise ValueError("Stage A evaluation requires the complete frozen explore100")
    for run in (baseline_run, pair_run):
        complete = json.loads((run / "completion.json").read_text())
        if complete["status"] != "complete" or complete["queries"] != 100:
            raise ValueError("Incomplete phase cannot be evaluated")
    audit = json.loads((protocol.STORE / "audit/inputs_v1.json").read_text())
    if protocol.sha256(LABELS) != audit["train_labels_sha256"]:
        raise ValueError("Ground truth changed")
    for name, expected in audit["evaluator_sha256"].items():
        if protocol.sha256(EVALUATOR / "standalone_eval" / name) != expected:
            raise ValueError("Official evaluator changed")
    sys.path.insert(0, str(EVALUATOR))
    from standalone_eval.eval import compute_ap_from_tuple, eval_submission, mk_gt_scores

    labels_by_id = {r["qid"]: r for r in protocol.read_jsonl(LABELS)}
    labels = [labels_by_id[row["qid"]] for row in rows]
    for row, gt in zip(rows, labels, strict=True):
        if any(row[key] != gt[key] for key in ("qid", "vid", "query", "duration")):
            raise ValueError("Label identity mismatch")
    groups = np.array([r["vid"].rsplit("_", 2)[0] for r in rows])
    records = [json.loads((pair_run / "records" / f"{r['qid']}.json").read_text()) for r in rows]
    baselines = [protocol.read_baseline(baseline_run, r) for r in rows]
    costs = {
        arm: {k: 0 for k in ("logical_requests", "input_tokens", "output_tokens", "model_frames", "inference_seconds")}
        for arm in ("B0", "P_abs", "P_pair")
    }
    predictions = {"B0": [r["prediction"] for r in baselines], "P_abs": [], "P_pair": []}
    for baseline, record, row in zip(baselines, records, rows, strict=True):
        protocol.validate_pair_record(record, row, binding, protocol.sha256(pair_run / "binding.json"), baseline)
        for key in costs["B0"]:
            costs["B0"][key] += baseline["cost"][key]
        for arm in ("P_abs", "P_pair"):
            predictions[arm].append(
                {
                    "qid": row["qid"],
                    "vid": row["vid"],
                    "query": row["query"],
                    "pred_saliency_scores": record["arms"][arm]["scores"],
                }
            )
            for key in costs[arm]:
                costs[arm][key] += record["arms"][arm]["cost"][key]
    values, results = {}, {}
    baseline_tokens = costs["B0"]["input_tokens"] + costs["B0"]["output_tokens"]
    for arm, preds in predictions.items():
        metrics = eval_submission(preds, labels, verbose=False, match_number=True)
        per_query = []
        for pred, gt in zip(preds, labels, strict=True):
            scores = np.array(pred["pred_saliency_scores"])
            binary = (mk_gt_scores(gt) >= 4).astype(float)
            ap = np.mean([compute_ap_from_tuple((0, w, binary[:, w], scores))[2] for w in range(3)])
            per_query.append([100 * ap, 100 * binary[int(np.argmax(scores))].max()])
        values[arm] = np.asarray(per_query)
        np.testing.assert_allclose(
            np.round(values[arm].mean(axis=0), 2),
            [metrics["HL-min-VeryGood"]["HL-mAP"], metrics["HL-min-VeryGood"]["HL-Hit1"]],
        )
        fallback = sum(r["arms"][arm]["status"] == "fallback" for r in records) if arm != "B0" else 0
        tokens = costs[arm]["input_tokens"] + costs[arm]["output_tokens"]
        results[arm] = {
            "metrics": metrics,
            "unrounded_VeryGood": dict(zip(("mAP", "Hit1"), values[arm].mean(axis=0).tolist())),
            "fallback_queries": fallback,
            "incremental_cost": costs[arm],
            "cold_total_token_ratio": 1.0 if arm == "B0" else 1 + tokens / baseline_tokens,
            "cold_total_tokens": tokens if arm == "B0" else tokens + baseline_tokens,
            "tied_max_queries": sum(p["pred_saliency_scores"].count(max(p["pred_saliency_scores"])) > 1 for p in preds),
        }
        protocol.write_jsonl(destination / f"{arm}_predictions.jsonl", preds)
    comparisons = {
        f"{a}_minus_{b}": paired_interval(values[a] - values[b], groups)
        for a, b in (("P_abs", "B0"), ("P_pair", "B0"), ("P_pair", "P_abs"))
    }
    diagnostics = {}
    for arm in ("P_abs", "P_pair"):
        valid = [r["arms"][arm] for r in records if r["arms"][arm]["status"] == "complete"]
        diagnostics[arm] = {
            "neutral_unordered_pairs": sum(r["ranking"]["neutral_unordered_pairs"] for r in valid),
            "both_yes_pairs": sum(r["ranking"]["both_yes_pairs"] for r in valid),
            "cycles": sum(r["ranking"]["cycles"] for r in valid),
            "query_rank_changed": sum(
                r["arms"][arm]["ranking"]["order"] != r["candidate_ids"]
                for r in records
                if r["arms"][arm]["status"] == "complete"
            ),
            "actual_scores_changed": sum(
                p["pred_saliency_scores"] != b["prediction"]["pred_saliency_scores"]
                for p, b in zip(predictions[arm], baselines, strict=True)
            ),
        }
    # Report unavoidable reuse of already viewed validation/train splits honestly.
    report = {
        "evaluated_at": protocol.utc_now(),
        "split": "explore100_train_exploratory",
        "queries": 100,
        "source_videos": len(set(groups)),
        "results": results,
        "paired_comparisons": comparisons,
        "ranking_diagnostics": diagnostics,
        "gate": choose_arm(results, comparisons),
        "bootstrap": {"unit": "source_video", "resamples": 10000, "seed": 42, "type": "paired_percentile_95pct"},
        "provenance": {
            "pair_binding_sha256": protocol.sha256(pair_run / "binding.json"),
            "baseline_binding_sha256": protocol.sha256(baseline_run / "binding.json"),
            "label_sha256": audit["train_labels_sha256"],
            "evaluator_sha256": audit["evaluator_sha256"],
        },
        "limits": [
            "Exploratory training subset; gates are point-estimate rules, not significance tests.",
            "P_abs sees B as context and is not an isolated one-video absolute baseline.",
            "The two new arms share a batch; half-batch time is an accounting allocation, not isolated latency.",
            "Five-candidate permutation cannot admit missed candidates or resolve equal original scores.",
        ],
    }
    protocol.atomic_json(destination / "evaluation.json", report)
    protocol.write_jsonl(
        destination / "per_query.jsonl",
        [
            {
                "qid": row["qid"],
                "vid": row["vid"],
                **{a: dict(zip(("mAP", "Hit1"), values[a][i].tolist())) for a in predictions},
            }
            for i, row in enumerate(rows)
        ],
    )
    lines = [
        "# REZE Evidence v1：A 阶段实验结果",
        "",
        f"评价时间：{report['evaluated_at']}",
        "",
        "固定 explore100（训练集 100 条查询、100 个源视频），全部完成后统一评价。",
        "",
        "| 方法 | VeryGood mAP | Hit@1 | 冷启动总 token / B0 | 回退查询 |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm, result in results.items():
        m = result["metrics"]["HL-min-VeryGood"]
        lines.append(
            f"| {arm} | {m['HL-mAP']:.2f} | {m['HL-Hit1']:.2f} | "
            f"{result['cold_total_token_ratio']:.3f}× | {result['fallback_queries']} |"
        )
    lines += [
        "",
        "## 预先规定的阶段判定",
        "",
        f"决策：`{report['gate']['decision']}`。",
        f"入选方法：`{report['gate']['selected_arm']}`。",
        "",
        "继续门槛：mAP 至少提高 1 个百分点，Hit@1 下降不超过 1 点，回退不超过 1 条，"
        "含 B0 的总 token 不超过 2.25 倍。两者合格时，成对评分还需比绝对评分高 0.5 点。",
        "",
        "## 配对差异与不确定性",
        "",
    ]
    for name, comparison in comparisons.items():
        delta = comparison["delta_percentage_points"]
        ci = comparison["ci95_percentage_points"]
        lines.append(
            f"- {name}：mAP {delta['mAP']:+.3f} pp，95% CI [{ci['mAP'][0]:.3f}, {ci['mAP'][1]:.3f}]；"
            f"Hit@1 {delta['Hit1']:+.2f} pp，95% CI [{ci['Hit1'][0]:.2f}, {ci['Hit1'][1]:.2f}]。"
        )
    lines += [
        "",
        "源视频聚类配对 bootstrap：10,000 次，seed=42。这是探索结果，阶段门槛不代表显著性检验。",
        "",
        "## 实验边界",
        "",
        "两组输入相同的 A/B 视频帧；P_abs 也看到 B。只对 B0 前 5 个 2s 片段的原分数做置换；"
        "其余分数不变，不增加新候选、不消除原分数并列。新增方法要求 Yes/No 均在原词表 top20；"
        "无效输出明确回退 B0。没有训练、记忆构建或回看。",
        "",
        "计费同时列出新增调用与包含 B0 的冷启动开销。两组在同一批次执行，不能将分摊时间解释为独立延迟。",
        "",
        "详细指标、逐查询结果、哈希和门槛布尔值见本目录 `evaluation.json`、`per_query.jsonl`。",
        "",
    ]
    (destination / "REPORT.md").write_text("\n".join(lines))
    return report
