"""Audit frozen R2 outputs before loading labels; evaluate all 100 queries."""

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from comparison import rerank, validate_comparison
from exporters import benchmark_prediction
from io_utils import compact, digest, legacy, parents, read_json, sha256, utc_now, write_json
from memory import retrieve
from prompts import OBSERVE
from r2_dispatch import active_seconds
from r2_observer import pair_actions, source_timeline, timeline_hash
from r2_planner import actionable_gaps, candidate_ranges, resolve_plan, update_memory, validate_plan
from r2_protocol import ARMS, OBSERVERS, SharedLedger, load_run
from schema import ObservationFacts
from task_grounding import validate_grounded_task
from worker import frame_map


def audit_observation_request(raw, action, expected_ids, timestamps, baseline_seen):
    trace = raw["specification"]["evidence"][0]
    frames = trace["frames"]
    ids = [f["frame_index"] for f in frames]
    if trace["action"] != action or ids != expected_ids or len(set(ids)) != len(ids) or not 2 <= len(ids) <= 24:
        raise ValueError("Actual observation changed its shared question or paired source-frame allocation")
    if set(ids) & set(baseline_seen):
        raise ValueError("Observation repeats parent B0 frames")
    for frame in frames:
        if frame["pts"] != timestamps[frame["frame_index"]].tolist():
            raise ValueError("Observation PTS differs from source")
    text = OBSERVE.format(
        question=action["question"], frames=compact([{"id": f["id"], "pts": f["pts"]} for f in frames])
    )
    if raw["specification"]["text"] != text:
        raise ValueError("Paired observation question or frame map changed")
    return len(ids)


def stage_gate(results, paired, within_budget, complete, observed_queries=0):
    controls = ["E0", "C_direct4", "E_memory", "E_uniform"]
    reference = max(controls, key=lambda a: results[a]["unrounded_VeryGood"]["mAP"])
    target, control = results["E_targeted"], results[reference]
    delta_map = target["unrounded_VeryGood"]["mAP"] - control["unrounded_VeryGood"]["mAP"]
    delta_hit = target["unrounded_VeryGood"]["Hit1"] - control["unrounded_VeryGood"]["Hit1"]
    checks = {
        "map_gain_at_least_1pp": delta_map >= 1.0,
        "hit1_loss_at_most_1pp": delta_hit >= -1.0,
        "technical_fallback_within_parent_and_1_percent": all(
            results[a]["fallback_queries"] <= min(1, results["E_memory"]["fallback_queries"]) for a in OBSERVERS
        ),
        "paired_questions_frames_and_calls": paired,
        "actual_paired_observation_executed": observed_queries > 0,
        "within_resource_caps": within_budget,
        "all_100_inference_completed": complete,
    }
    return {
        "status": "passed" if all(checks.values()) else "not_passed",
        "checks": checks,
        "reference": reference,
        "delta_mAP_pp": delta_map,
        "delta_Hit1_pp": delta_hit,
        "automatic_R3_R4_or_fullval": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    binding, rows, binding_hash = load_run(run)
    directory = run / "R2"
    done = read_json(directory / "inference_complete.json")
    if (
        done["binding_sha256"] != binding_hash
        or done["queries"] != 100
        or done["record_receipt_sha256"] != sha256(directory / "record_receipt.json")
        or done["ledger_sha256"] != sha256(directory / "ledger.json")
    ):
        raise ValueError("R2 completion receipt drift")
    ledger = SharedLedger(directory, binding["budgets"])
    state = ledger.state
    raw_records = {}
    for item in state["attempts"]:
        if item["status"] in {"complete", "invalid", "runtime_error"}:
            path = directory / "attempts" / f"{item['number']:05d}.json"
            raw = read_json(path)
            if digest(raw) != item["record_sha256"] or digest(raw["specification"]) != raw["request_sha256"]:
                raise ValueError("R2 raw attempt hash drift")
            raw_records[str(path)] = raw
    for entry in state.get("reused", {}).values():
        path = Path(entry["path"])
        raw = read_json(path)
        if sha256(path) != entry["file_sha256"] or digest(raw) != entry["record_sha256"] or raw["status"] != "complete":
            raise ValueError("R2 exact R1 cache changed")
        raw_records[str(path)] = raw
    from backend import MODEL
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    receipt = read_json(directory / "record_receipt.json")
    frozen = {r["qid"]: r for r in legacy.read_jsonl(run / "candidate_inputs.jsonl")}
    predictions = {a: [] for a in ["B0", "E0", *ARMS]}
    failures = Counter()
    comparison_states = {a: Counter() for a in ARMS}
    pair_audit, e0_tokens = [], 0
    for row in rows:
        path = directory / "records" / f"{row['qid']}.json"
        if sha256(path) != receipt[str(row["qid"])]:
            raise ValueError("R2 result changed after inference")
        record = read_json(path)
        if record["binding_sha256"] != binding_hash or any(record[k] != row[k] for k in ("qid", "vid", "query")):
            raise ValueError("R2 result identity mismatch")
        baseline, pair = parents(binding, row)
        selected, original = pair["candidate_ids"], baseline["prediction"]["pred_saliency_scores"]
        e0 = pair["arms"]["P_pair"]
        if record["candidate_ids"] != selected or record["original"] != original or record["E0"] != e0["scores"]:
            raise ValueError("R2 parent prediction drift")
        e0_tokens += sum(e0["cost"][k] for k in ("input_tokens", "output_tokens"))
        if record.get("task"):
            task_raw = raw_records[record["task_response_path"]]
            if validate_grounded_task(task_raw["parsed"], row["query"]) != record["task"]:
                raise ValueError("Grounded task differs from actual model output")
        for arm, scores in (("B0", original), ("E0", e0["scores"])):
            predictions[arm].append(benchmark_prediction(row, scores, original, selected))
        for arm in ARMS:
            result = record["arms"][arm]
            if result["status"] == "complete":
                if record["candidate_frames"] != frozen[row["qid"]]["candidate_frames"]:
                    raise ValueError("R2 candidate evidence drift")
                view = {} if arm == "C_direct4" else retrieve(record["memories"][arm], tokenizer)
                ids = {c: [f["id"] for f in frame_map(record["candidate_frames"][str(c)])] for c in selected}
                for comparison in result["comparisons"]:
                    validate_comparison(comparison["result"], comparison["a"], comparison["b"], ids, view)
                    raw = raw_records[comparison["response_path"]]
                    if raw["status"] != "complete" or raw["parsed"] != comparison["result"]:
                        raise ValueError("Comparison differs from recorded model output")
                scores, rank = rerank(original, selected, result["comparisons"])
                if scores != result["scores"] or rank != result["ranking"]:
                    raise ValueError("R2 score permutation does not replay")
                comparison_states[arm].update(edge["state"] for edge in rank["edges"])
            elif result["status"] == "fallback" and result.get("error"):
                parent_scores = e0["scores"] if result["parent_arm"] == "E0" else record["arms"]["E_memory"]["scores"]
                if result["scores"] != parent_scores:
                    raise ValueError("R2 fallback differs from declared parent")
                failures[arm] += 1
            else:
                raise ValueError("Unknown R2 result status")
            predictions[arm].append(benchmark_prediction(row, result["scores"], original, selected))
        planning = record.get("planning", {})
        if planning:
            if planning["gaps"] != actionable_gaps(record["arms"]["E_memory"]):
                raise ValueError("Planner bypassed its frozen uncertainty trigger")
            if planning.get("response_path"):
                raw = raw_records[planning["response_path"]]
                plan = validate_plan(
                    raw["parsed"], planning["gaps"], candidate_ranges(selected, float(row["duration"]))
                )
                resolved = resolve_plan(
                    plan, planning["gaps"], candidate_ranges(selected, float(row["duration"])), record["task"]
                )
                if planning["model_actions"] != plan["actions"] or any(planning[k] != v for k, v in resolved.items()):
                    raise ValueError("Observation plan/controller resolution does not replay")
        expected = []
        if record.get("paired_actions"):
            reader, timestamps, seen = source_timeline(row, baseline)
            if timeline_hash(timestamps) != frozen[row["qid"]]["source_timeline_sha256"]:
                raise ValueError("Source timeline drift during paired audit")
            expected = pair_actions(timestamps, planning["actions"], seen)
            for actual, wanted in zip(record["paired_actions"], expected, strict=True):
                if any(actual[k] != wanted[k] for k in ("action", "uniform", "targeted", "number")):
                    raise ValueError("Paired frame plan does not replay")
            del reader
        obs_counts, comparison_counts, source_counts, per_action_counts = {}, {}, {}, {}
        for arm in OBSERVERS:
            prefix = f"{row['qid']}/{arm}/"
            attempts = [a for a in state["attempts"] if a["key"].startswith(prefix)]
            obs = [a for a in attempts if "/observe/" in a["key"]]
            obs_counts[arm] = len({a["key"] for a in obs})
            comparison_counts[arm] = len({a["key"] for a in attempts if "/observe/" not in a["key"]})
            source_ids = set()
            per_action_counts[arm] = {}
            for attempt in obs:
                raw = raw_records.get(str(directory / "attempts" / f"{attempt['number']:05d}.json"))
                if raw:
                    frames = raw["specification"]["evidence"][0]["frames"]
                    if {f["frame_index"] for f in frames} & set(frozen[row["qid"]]["B0_seen_indices"]):
                        raise ValueError("A supposedly new observation repeats B0 source frames")
                    source_ids.update(f["frame_index"] for f in frames)
                    number = int(attempt["key"].rsplit("/", 1)[1])
                    per_action_counts[arm][number] = len(frames)
                    if expected:
                        planned = expected[number]
                        key = "uniform" if arm == "E_uniform" else "targeted"
                        audit_observation_request(
                            raw, planned["action"], planned[key], timestamps, frozen[row["qid"]]["B0_seen_indices"]
                        )
            source_counts[arm] = len(source_ids)
            observations = record.get("observations", {}).get(arm, [])
            for observation in observations:
                raw = raw_records[observation["response_path"]]
                value = ObservationFacts.model_validate(raw["parsed"]).model_dump()
                if value != observation["result"] or raw["specification"]["evidence"] != [observation["trace"]]:
                    raise ValueError("Observation differs from actual source evidence")
            if observations and record["arms"][arm]["status"] == "complete":
                replay = update_memory(record["memories"]["E_memory"], observations, tokenizer)
                if replay != record["memories"][arm]:
                    raise ValueError("Memory update bypassed deterministic capacity rule")
        balanced = (
            len(set(obs_counts.values()))
            == len(set(comparison_counts.values()))
            == len(set(source_counts.values()))
            == 1
            and per_action_counts["E_uniform"] == per_action_counts["E_targeted"]
            and (bool(expected) or not any(obs_counts.values()))
        )
        pair_audit.append(
            {
                "qid": row["qid"],
                "planner_called": planning.get("called", False),
                "plan_source": planning.get("source"),
                "planned_actions": len(planning.get("actions", [])),
                "model_actions": len(planning.get("model_actions", [])),
                "observation_requests": obs_counts,
                "comparison_requests": comparison_counts,
                "new_unique_source_frames": source_counts,
                "frames_per_action": per_action_counts,
                "balanced": balanced,
            }
        )
    # All immutable predictions and source evidence were audited before labels enter.
    source = Path(os.environ.get("QVH_EVALUATOR_ROOT", "third_party/moment_detr"))
    labels_path = source / "data/highlight_train_release.jsonl"
    audit = read_json(run / "evaluation_binding.json")
    if sha256(labels_path) != audit["train_labels_sha256"]:
        raise ValueError("Ground truth changed")
    for name, expected in audit["evaluator_sha256"].items():
        if sha256(source / "standalone_eval" / name) != expected:
            raise ValueError("Official evaluator changed")
    sys.path.insert(0, str(source))
    from standalone_eval.eval import compute_ap_from_tuple, eval_submission, mk_gt_scores
    from summarize import paired_interval

    by_id = {r["qid"]: r for r in legacy.read_jsonl(labels_path)}
    labels = [by_id[r["qid"]] for r in rows]
    for row, gt in zip(rows, labels, strict=True):
        if any(row[k] != gt[k] for k in ("qid", "vid", "query", "duration")):
            raise ValueError("Ground-truth identity mismatch")
    results, values = {}, {}
    for arm, preds in predictions.items():
        metrics = eval_submission(preds, labels, verbose=False, match_number=True)
        per_query = []
        for prediction, gt in zip(preds, labels, strict=True):
            scores = np.asarray(prediction["pred_saliency_scores"])
            binary = (mk_gt_scores(gt) >= 4).astype(float)
            ap = np.mean([compute_ap_from_tuple((0, w, binary[:, w], scores))[2] for w in range(3)])
            per_query.append([100 * ap, 100 * binary[int(np.argmax(scores))].max()])
        values[arm] = np.asarray(per_query)
        means = values[arm].mean(axis=0)
        np.testing.assert_allclose(
            np.round(means, 2), [metrics["HL-min-VeryGood"]["HL-mAP"], metrics["HL-min-VeryGood"]["HL-Hit1"]]
        )
        incremental = 0 if arm == "B0" else e0_tokens if arm == "E0" else ledger.cost(arm)
        results[arm] = {
            "metrics": metrics,
            "unrounded_VeryGood": dict(zip(("mAP", "Hit1"), means.tolist())),
            "fallback_queries": failures[arm],
            "independent_incremental_tokens": incremental,
            "end_to_end_token_ratio": 1 + incremental / binding["baseline_tokens"],
        }
        legacy.write_jsonl(directory / "results" / f"{arm}_predictions.jsonl", preds)
    attempts = read_json(directory / "worker_attempts.json")
    within_budget = (
        active_seconds(attempts) <= binding["maximum_active_worker_seconds"]
        and all(results[a]["end_to_end_token_ratio"] <= 6 for a in ARMS)
        and len(state["attempts"]) <= binding["budgets"]["requests"] + 460
    )
    groups = np.array([r["vid"].rsplit("_", 2)[0] for r in rows])
    paired = {
        f"{a}_minus_{b}": paired_interval(values[a] - values[b], groups)
        for a, b in [
            ("C_direct4", "E0"),
            ("E_memory", "C_direct4"),
            ("E_uniform", "E_memory"),
            ("E_targeted", "E_memory"),
            ("E_targeted", "E_uniform"),
            ("E_targeted", "E0"),
        ]
    }
    coverage = {
        "planner_queries": sum(a["planner_called"] for a in pair_audit),
        "model_action_queries": sum(a["model_actions"] > 0 for a in pair_audit),
        "controller_probe_queries": sum(a["plan_source"] == "controller_conflict_probe" for a in pair_audit),
        "paired_observed_queries": sum(all(a["observation_requests"].values()) for a in pair_audit),
        "new_frames": {arm: sum(a["new_unique_source_frames"][arm] for a in pair_audit) for arm in OBSERVERS},
    }
    gate = stage_gate(
        results,
        all(a["balanced"] for a in pair_audit),
        within_budget,
        done["status"] == "inference_complete",
        coverage["paired_observed_queries"],
    )
    report = {
        "status": "complete" if done["status"] == "inference_complete" else "partial_stopped",
        "stopped_reason": done["stopped_reason"],
        "binding_sha256": binding_hash,
        "queries": 100,
        "source_videos": len(set(groups)),
        "evaluated_at": utc_now(),
        "split": "previously used train explore100; development only",
        "results": results,
        "paired_comparisons": paired,
        "comparison_states": comparison_states,
        "pair_audit": pair_audit,
        "observation_coverage": coverage,
        "gate": gate,
        "actual_new_attempts": len(state["attempts"]),
        "reused_requests": len(state.get("reused", {})),
        "actual_new_shared_compute_tokens": ledger.cost(),
        "summed_gpu_seconds": active_seconds(attempts),
        "human_review": "not_reviewed",
        "R3_R4_started": False,
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
    lines = [
        "# ReVALUE R2：100 条候选附近配对观察实验",
        "",
        "完整分母 100；原 explore100 开发诊断。R3/R4 未启动。",
        "",
        "| 条件 | VeryGood mAP | Hit@1 | 回退 | 端到端 token / B0 |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm, item in results.items():
        m = item["unrounded_VeryGood"]
        lines.append(
            f"| {arm} | {m['mAP']:.2f} | {m['Hit1']:.2f} | {item['fallback_queries']} | {item['end_to_end_token_ratio']:.3f} |"
        )
    lines += [
        "",
        f"精度确认门槛：{gate['status']}。参考条件：{gate['reference']}。",
        f"规划触发 {coverage['planner_queries']} 条；模型动作 {coverage['model_action_queries']} 条；"
        f"控制器冲突查证 {coverage['controller_probe_queries']} 条；实际配对补看 {coverage['paired_observed_queries']} 条。",
        f"两组新增源帧：{coverage['new_frames']}。零次回看不构成回看有效性的检验。",
        "",
        "E_targeted − E_uniform 检验观察位置；E_uniform − E_memory 检验补看；不据局部实验声称完整 Agent 或候选外召回。",
        "",
    ]
    (directory / "results/REPORT.md").write_text("\n".join(lines))
    write_json(run / "decision.json", gate)
    complete = {
        "status": report["status"],
        "finished_at": utc_now(),
        "evaluation_sha256": sha256(directory / "results/evaluation.json"),
        "gate": gate["status"],
    }
    write_json(run / "complete.json", complete)
    write_json(run / "status.json", {**complete, "stage": "R2", "completed_queries": 100})
    print(compact({"status": report["status"], "gate": gate, "results": results}))


if __name__ == "__main__":
    main()
