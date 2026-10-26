"""Official full-denominator HD evaluation, cost accounting and blank human review."""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
from budget import Ledger
from comparison import rerank, validate_comparison
from exporters import benchmark_prediction
from io_utils import compact, digest, legacy, load_run, parents, read_json, sha256, utc_now, write_json
from memory import retrieve
from worker import ARMS, frame_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    directory = run / "R1"
    binding, rows, binding_hash = load_run(run)
    complete = read_json(directory / "inference_complete.json")
    if (
        complete["binding_sha256"] != binding_hash
        or complete["queries"] != 20
        or (complete["record_receipt_sha256"] != sha256(directory / "record_receipt.json"))
    ):
        raise ValueError("Incomplete or changed formal inference")
    ledger = Ledger(directory / "ledger.json", binding["budgets"]["R1"])
    for attempt in ledger.state["attempts"]:
        if attempt["status"] in {"complete", "invalid", "runtime_error"}:
            raw = read_json(directory / "attempts" / f"{attempt['number']:05d}.json")
            if digest(raw) != attempt["record_sha256"] or raw["request_sha256"] != digest(raw["specification"]):
                raise ValueError("Request audit drift")
    from backend import MODEL
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    frozen_frames = {r["qid"]: r["candidate_frames"] for r in legacy.read_jsonl(run / "candidate_inputs.jsonl")}
    receipt = read_json(directory / "record_receipt.json")
    predictions = {a: [] for a in ["B0", "E0", *ARMS]}
    failures = {a: 0 for a in predictions}
    comparison_states = {a: {k: 0 for k in ("winner", "tie", "unresolved")} for a in ARMS}
    memory_lengths = {a: [] for a in ("M_plain", "M_decision")}
    e0_tokens, blind_key, blind_payload = 0, [], []
    for row in rows:
        path = directory / "records" / f"{row['qid']}.json"
        if sha256(path) != receipt[str(row["qid"])]:
            raise ValueError("Prediction receipt changed")
        record = read_json(path)
        if record["binding_sha256"] != binding_hash or any(record[k] != row[k] for k in ("qid", "vid", "query")):
            raise ValueError("Prediction identity drift")
        baseline, pair = parents(binding, row)
        original, selected = baseline["prediction"]["pred_saliency_scores"], pair["candidate_ids"]
        if record["candidate_ids"] != selected or record["original"] != original:
            raise ValueError("Candidate/B0 drift")
        e0 = pair["arms"]["P_pair"]
        e0_tokens += e0["cost"]["input_tokens"] + e0["cost"]["output_tokens"]
        for arm, scores in (("B0", original), ("E0", e0["scores"])):
            predictions[arm].append(benchmark_prediction(row, scores, original, selected))
        for arm in ARMS:
            output = record["arms"][arm]
            if output["status"] == "complete":
                if record["candidate_frames"] != frozen_frames[row["qid"]]:
                    raise ValueError("Actual candidate frames drifted")
                view = {} if arm == "C_direct4" else retrieve(record["memories"][arm], tokenizer)
                if arm != "C_direct4":
                    memory_lengths[arm].append(record["memories"][arm]["tokens"])
                ids = {c: [f["id"] for f in frame_map(record["candidate_frames"][str(c)])] for c in selected}
                for item in output["comparisons"]:
                    validate_comparison(item["result"], item["a"], item["b"], ids, view)
                    raw = read_json(directory / "attempts" / f"{item['attempt']:05d}.json")
                    if raw["status"] != "complete" or raw["parsed"] != item["result"]:
                        raise ValueError("Comparison no longer matches raw output")
                replay, ranking = rerank(original, selected, output["comparisons"])
                if replay != output["scores"] or digest(ranking) != digest(output["ranking"]):
                    raise ValueError("Four-state ranking did not replay")
                for edge in ranking["edges"]:
                    comparison_states[arm][edge["state"]] += 1
            elif output["status"] == "fallback" and output.get("error") and output["scores"] == e0["scores"]:
                failures[arm] += 1
                view = None
            else:
                raise ValueError("Invalid fallback")
            predictions[arm].append(benchmark_prediction(row, output["scores"], original, selected))
            blind_id = digest(["revalue-human-v1", row["qid"], arm])[:16]
            blind_key.append({"blind_id": blind_id, "qid": row["qid"], "arm": arm})
            blind_payload.append(
                {
                    "blind_id": blind_id,
                    "query": row["query"],
                    "video_path": row["video_path"],
                    "frame_sources": frozen_frames[row["qid"]],
                    "active_memory": view,
                    "comparisons": [
                        {k: v for k, v in x.items() if k in {"a", "b", "result"}} for x in output.get("comparisons", [])
                    ],
                }
            )
    # Labels/evaluator are loaded only here, after all inference outputs are immutable.
    source = Path(os.environ.get("QVH_EVALUATOR_ROOT", "third_party/moment_detr"))
    audit = read_json(run / "evaluation_binding.json")
    labels_path = source / "data/highlight_train_release.jsonl"
    if sha256(labels_path) != audit["train_labels_sha256"]:
        raise ValueError("Ground truth drift")
    for name, expected in audit["evaluator_sha256"].items():
        if sha256(source / "standalone_eval" / name) != expected:
            raise ValueError("Official evaluator drift")
    sys.path.insert(0, str(source))
    from standalone_eval.eval import compute_ap_from_tuple, eval_submission, mk_gt_scores
    from summarize import paired_interval

    labels_by_id = {r["qid"]: r for r in legacy.read_jsonl(labels_path)}
    labels = [labels_by_id[r["qid"]] for r in rows]
    for row, gt in zip(rows, labels, strict=True):
        if any(row[k] != gt[k] for k in ("qid", "vid", "query", "duration")):
            raise ValueError("GT/inference identity mismatch")
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
        incremental = 0 if arm == "B0" else e0_tokens if arm == "E0" else ledger.cost(arm)
        ratio = 1 + incremental / binding["baseline_tokens"]
        if arm in ARMS and ratio > 6:
            raise ValueError("End-to-end token budget exceeded")
        results[arm] = {
            "metrics": metrics,
            "unrounded_VeryGood": dict(zip(("mAP", "Hit1"), means.tolist())),
            "fallback_queries": failures[arm],
            "end_to_end_token_ratio": ratio,
            "independent_incremental_tokens_with_failed_reservations": incremental,
        }
        legacy.write_jsonl(directory / "results" / f"{arm}_predictions.jsonl", preds)
    groups = np.array([r["vid"].rsplit("_", 2)[0] for r in rows])
    paired = {
        f"{a}_minus_{b}": paired_interval(values[a] - values[b], groups)
        for a, b in (
            ("C_direct4", "E0"),
            ("M_plain", "C_direct4"),
            ("M_decision", "C_direct4"),
            ("M_decision", "M_plain"),
        )
    }
    report = {
        "status": "complete" if complete["status"] == "inference_complete" else "partial_budget_stop",
        "evaluated_at": utc_now(),
        "binding_sha256": binding_hash,
        "queries": 20,
        "source_videos": len(set(groups)),
        "split": "fixed previously used train explore100 subset; development only",
        "results": results,
        "paired_comparisons": paired,
        "comparison_states": comparison_states,
        "memory_token_lengths": memory_lengths,
        "actual_new_shared_compute_tokens": ledger.cost(),
        "actual_attempts": len(ledger.state["attempts"]),
        "human_review": "not_reviewed",
        "memory_contribution_control": "M_decision minus M_plain, same comparator and capacity",
        "R2_or_R3_started": False,
    }
    write_json(directory / "results/evaluation.json", report)
    legacy.write_jsonl(
        directory / "results/per_query.jsonl",
        [
            {
                "qid": row["qid"],
                "vid": row["vid"],
                **{a: dict(zip(("mAP", "Hit1"), values[a][i].tolist())) for a in values},
            }
            for i, row in enumerate(rows)
        ],
    )
    legacy.write_jsonl(directory / "results/human_review_key.private.jsonl", blind_key)
    blind_payload.sort(key=lambda r: r["blind_id"])
    legacy.write_jsonl(directory / "results/human_review_evidence.jsonl", blind_payload)
    with (directory / "results/human_review_blank.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["blind_id", "event_confusion", "detail_retention", "unsupported_claims", "comparison_quality", "notes"]
        )
        for item in blind_payload:
            writer.writerow([item["blind_id"], "", "", "", "", ""])
    lines = [
        "# ReVALUE R1：固定 20 条记忆与四态比较诊断",
        "",
        "开发诊断，完整分母 20；技术失败回退 E0。人工审阅未完成；未启动 R2/R3。",
        "",
        "| 条件 | VeryGood mAP | Hit@1 | 回退 | 端到端 token / B0 |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm, result in results.items():
        metric = result["unrounded_VeryGood"]
        lines.append(
            f"| {arm} | {metric['mAP']:.2f} | {metric['Hit1']:.2f} | {failures[arm]} | {result['end_to_end_token_ratio']:.3f} |"
        )
    lines += [
        "",
        "记忆贡献的直接对照为 M_decision − M_plain；C_direct4 − E0 包含比较输出与排序规则变化。",
        "两记忆的容量上限相同，实际长度单独报告。失败、未决和并列分开计数，bootstrap 按源视频聚类。",
        "20 条已用于开发，不构成独立性能确认；预算终止时不得解释为完整方法达到精度结论。",
        "",
    ]
    (directory / "results/REPORT.md").write_text("\n".join(lines))
    write_json(
        run / "decision.json",
        {
            "stage": "R1",
            "status": report["status"],
            "next": "Review fixed diagnostic and failures, then implement/freeze R2; no automatic full validation",
            "human_review": "pending",
        },
    )
    write_json(
        run / "complete.json",
        {
            "status": report["status"],
            "finished_at": utc_now(),
            "evaluation_sha256": sha256(directory / "results/evaluation.json"),
        },
    )
    write_json(
        run / "status.json",
        {"status": report["status"], "stage": "R1", "updated_at": utc_now(), "completed_queries": 20},
    )
    print(compact(report["results"]))


if __name__ == "__main__":
    main()
