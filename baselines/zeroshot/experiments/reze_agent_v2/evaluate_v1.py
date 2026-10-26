"""Full-denominator official HD evaluation and prespecified C0/C1 decision."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from core import legacy, load_run, parent_records, validate_record
from summarize import paired_interval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    binding, rows, binding_hash = load_run(run)
    complete = json.loads((run / "inference_complete.json").read_text())
    if complete["binding_sha256"] != binding_hash or complete["queries"] != len(rows):
        raise ValueError("Incomplete formal predictions")
    if complete["record_receipt_sha256"] != legacy.sha256(run / "record_receipt.json"):
        raise ValueError("Record receipt changed")
    receipt = json.loads((run / "record_receipt.json").read_text())
    source = Path(os.environ.get("REZE_QV_EVALUATOR", os.environ.get("QVH_EVALUATOR_ROOT", "third_party/moment_detr")))
    label_path = source / "data/highlight_train_release.jsonl"
    audit = json.loads((run / "evaluation_binding.json").read_text())
    if legacy.sha256(label_path) != audit["train_labels_sha256"]:
        raise ValueError("Ground truth drift")
    for name, expected in audit["evaluator_sha256"].items():
        if legacy.sha256(source / "standalone_eval" / name) != expected:
            raise ValueError("Official evaluator drift")
    sys.path.insert(0, str(source))
    from standalone_eval.eval import compute_ap_from_tuple, eval_submission, mk_gt_scores

    labels_by_id = {r["qid"]: r for r in legacy.read_jsonl(label_path)}
    labels = [labels_by_id[r["qid"]] for r in rows]
    predictions = {arm: [] for arm in ("B0", "C0", "C1")}
    costs = {
        arm: {k: 0 for k in ("logical_requests", "input_tokens", "output_tokens", "model_frames", "inference_seconds")}
        for arm in predictions
    }
    fallback = {arm: 0 for arm in predictions}
    diagnostics = []
    for row, gt in zip(rows, labels, strict=True):
        if any(row[k] != gt[k] for k in ("qid", "vid", "query", "duration")):
            raise ValueError("Evaluation identity mismatch")
        path = run / "records" / f"{row['qid']}.json"
        if legacy.sha256(path) != receipt[str(row["qid"])]:
            raise ValueError("Formal prediction changed after completion")
        record = json.loads(path.read_text())
        validate_record(record, row, binding, binding_hash)
        baseline, pair = parent_records(binding, row)
        candidates = {"C0": pair["candidate_ids"], "C1": record["candidate_ids"]}
        for arm, scores, cost, status in (
            ("B0", baseline["prediction"]["pred_saliency_scores"], baseline["cost"], "complete"),
            ("C0", pair["arms"]["P_pair"]["scores"], pair["arms"]["P_pair"]["cost"], pair["arms"]["P_pair"]["status"]),
            ("C1", record["scores"], record["cost"], record["status"]),
        ):
            predictions[arm].append({**{k: row[k] for k in ("qid", "vid", "query")}, "pred_saliency_scores": scores})
            for key in costs[arm]:
                costs[arm][key] += cost[key]
            fallback[arm] += status == "fallback"
        binary = (mk_gt_scores(gt) >= 4).astype(float)
        item = {"qid": row["qid"], "C1_fill_ids": record["selection"]["fill_ids"], "C1_status": record["status"]}
        for arm, selected in candidates.items():
            ordered = sorted(selected)
            item[arm] = {
                "candidate_ids": selected,
                "contiguous_runs": 1 + sum(b != a + 1 for a, b in zip(ordered, ordered[1:])),
                "candidate_VeryGood_hits": int(sum(binary[i].max() for i in selected)),
                "candidate_oracle_hit1": float(max(binary[i].max() for i in selected)),
            }
        item["C1_ranking"] = record.get("ranking")
        diagnostics.append(item)
    values, results = {}, {}
    for arm, preds in predictions.items():
        metrics = eval_submission(preds, labels, verbose=False, match_number=True)
        per_query = []
        for pred, gt in zip(preds, labels, strict=True):
            scores = np.asarray(pred["pred_saliency_scores"])
            binary = (mk_gt_scores(gt) >= 4).astype(float)
            ap = np.mean([compute_ap_from_tuple((0, w, binary[:, w], scores))[2] for w in range(3)])
            per_query.append([100 * ap, 100 * binary[int(np.argmax(scores))].max()])
        values[arm] = np.asarray(per_query)
        means = values[arm].mean(axis=0)
        np.testing.assert_allclose(
            np.round(means, 2), [metrics["HL-min-VeryGood"]["HL-mAP"], metrics["HL-min-VeryGood"]["HL-Hit1"]]
        )
        token_sum = costs[arm]["input_tokens"] + costs[arm]["output_tokens"]
        results[arm] = {
            "metrics": metrics,
            "unrounded_VeryGood": dict(zip(("mAP", "Hit1"), means.tolist())),
            "fallback_queries": fallback[arm],
            "incremental_cost": costs[arm],
            "end_to_end_token_ratio": 1 if arm == "B0" else 1 + token_sum / binding["baseline_tokens"],
        }
        legacy.write_jsonl(run / "results" / f"{arm}_predictions.jsonl", preds)
    groups = np.array([r["vid"].rsplit("_", 2)[0] for r in rows])
    comparisons = {
        f"{a}_minus_{b}": paired_interval(values[a] - values[b], groups)
        for a, b in (("C1", "C0"), ("C1", "B0"), ("C0", "B0"))
    }
    delta = (values["C1"] - values["C0"]).mean(axis=0)
    checks = {
        "mAP_gain_at_least_1pp": bool(delta[0] >= 1),
        "Hit1_drop_at_most_1pp": bool(delta[1] >= -1),
        "fallback_does_not_increase": fallback["C1"] <= fallback["C0"],
        "end_to_end_token_ratio_at_most_6": results["C1"]["end_to_end_token_ratio"] <= 6,
    }
    decision = {
        "stage": "V1",
        "checks": checks,
        "selected_candidates": "C1" if all(checks.values()) else "C0",
        "delta_unrounded_pp": {"mAP": float(delta[0]), "Hit1": float(delta[1])},
        "next_stage": "V2 engineering then four-arm development; no automatic expansion",
    }
    report = {
        "evaluated_at": legacy.utc_now(),
        "queries": len(rows),
        "source_videos": len(set(groups)),
        "split": "previously used explore100 development",
        "results": results,
        "paired_comparisons": comparisons,
        "decision": decision,
        "bootstrap": {"resamples": 10000, "seed": 42, "unit": "source_video"},
        "binding_sha256": binding_hash,
        "engineering_accuracy_evaluated": False,
    }
    legacy.atomic_json(run / "results/evaluation.json", report)
    legacy.atomic_json(run / "decision.json", decision)
    legacy.write_jsonl(
        run / "results/per_query.jsonl",
        [
            {
                "qid": row["qid"],
                "vid": row["vid"],
                **{a: dict(zip(("mAP", "Hit1"), values[a][i].tolist())) for a in values},
            }
            for i, row in enumerate(rows)
        ],
    )
    legacy.write_jsonl(run / "results/candidate_diagnostics.jsonl", diagnostics)
    lines = [
        "# QVHighlights V1：候选时间分散对照",
        "",
        "固定 explore100，共 100 条查询；完整分母，标签仅用于评价。",
        "",
        "| 条件 | VeryGood mAP | Hit@1 | 回退 | 端到端 token / B0 |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm, result in results.items():
        m = result["unrounded_VeryGood"]
        lines.append(
            f"| {arm} | {m['mAP']:.2f} | {m['Hit1']:.2f} | {result['fallback_queries']} | {result['end_to_end_token_ratio']:.3f} |"
        )
    lines += [
        "",
        f"下一阶段选择：**{decision['selected_candidates']}**。C1−C0：mAP {delta[0]:+.3f} pp，Hit@1 {delta[1]:+.2f} pp。",
        "",
        "C1 固定抑制 clip ID 距离 ≤2，仍使用原 P_pair 比较器，且只置换五候选的原分数。",
        "未达到 mAP +1 pp、Hit@1 降幅 ≤1 pp、回退不增加的门槛时，后续 Agent 使用 C0。",
        "开发集已用于方案探索；不是独立测试。时间分散不等于不同语义事件，候选外分数保持不变。",
        "",
        "原 B0/C0 核验复用；C1 正式调用与工程调用分别保存在账本。该阶段没有生成记忆或新增源视频观察。",
        "",
    ]
    (run / "results/REPORT.md").write_text("\n".join(lines))
    legacy.atomic_json(
        run / "complete.json",
        {
            "status": "complete",
            "finished_at": legacy.utc_now(),
            "evaluation_sha256": legacy.sha256(run / "results/evaluation.json"),
            "binding_sha256": binding_hash,
            "queries": len(rows),
        },
    )
    legacy.atomic_json(
        run / "status.json",
        {"status": "complete", "updated_at": legacy.utc_now(), "completed_queries": len(rows), "decision": decision},
    )
    print(json.dumps(decision))


if __name__ == "__main__":
    main()
