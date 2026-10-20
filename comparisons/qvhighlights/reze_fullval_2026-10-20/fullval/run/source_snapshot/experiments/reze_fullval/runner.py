"""Full-validation extension of frozen Stage A; reuse B0 and never tune on labels."""

import argparse
import fcntl
import importlib.metadata
import json
import math
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "experiments/reze_evidence"))
import protocol  # noqa: E402
from candidate_frames import eligible_observations  # noqa: E402
from launch import inventory, run_phase  # noqa: E402
from ranking import candidates  # noqa: E402

ARMS = ("B0", "P_abs", "P_pair")
COST_KEYS = ("logical_requests", "input_tokens", "output_tokens", "model_frames", "inference_seconds")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def exact_coverage(rows, receipt, expected):
    protocol.validate_rows(rows)
    require(len(rows) == expected, "Incomplete full-validation manifest")
    require(set(receipt) == {str(row["qid"]) for row in rows}, "Receipt coverage mismatch")


def check_complete(run, rows):
    complete = json.loads((run / "completion.json").read_text())
    require(complete["status"] == "complete", "Incomplete run")
    require(complete.get("queries", complete.get("completed_queries")) == len(rows), "Incomplete query count")
    require(complete["binding_sha256"] == protocol.sha256(run / "binding.json"), "Binding drift")
    require(complete["record_receipt_sha256"] == protocol.sha256(run / "record_receipt.json"), "Receipt drift")
    receipt = json.loads((run / "record_receipt.json").read_text())
    exact_coverage(rows, receipt, len(rows))
    for qid, digest in receipt.items():
        require(protocol.sha256(run / "records" / f"{qid}.json") == digest, "Record receipt mismatch")
    return complete


def baseline_cost(record):
    """Original full B0 has no Stage-A cost field: reconstruct from raw outputs."""
    return {
        "logical_requests": len(record["clips"]),
        "input_tokens": sum(r["prompt_tokens"] for r in record["clips"]),
        "output_tokens": len(record["clips"]),
        "model_frames": record["media"]["model_input_frames"],
        "inference_seconds": record["generate_seconds"],
    }


def preflight(plan):
    from reze_run_io import load_bound_run

    parent = Path(plan["baseline_run"])
    require(protocol.sha256(parent / "completion.json") == plan["baseline_completion_sha256"], "B0 changed")
    binding, rows = load_bound_run(parent)
    require(protocol.sha256(parent / "inference.jsonl") == plan["manifest_sha256"], "Manifest changed")
    require(len(rows) == plan["queries"] == 1550, "Full validation requires 1550 queries")
    check_complete(parent, rows)
    require(
        protocol.sha256(parent / "predictions.jsonl") == plan["baseline_prediction_sha256"], "B0 predictions changed"
    )
    for name, digest in plan["frozen_method_sources"].items():
        require(protocol.sha256(PROJECT / name) == digest, f"Stage-A method changed: {name}")
    require(len({r["vid"] for r in rows}) == plan["video_segments"], "Segment count mismatch")
    require(len({r["vid"].rsplit("_", 2)[0] for r in rows}) == plan["source_videos"], "Source count mismatch")
    require(protocol.sha256(plan["labels_path"]) == plan["labels_sha256"], "Labels changed")
    labels = {r["qid"]: r for r in protocol.read_jsonl(plan["labels_path"])}
    require(set(labels) == {r["qid"] for r in rows}, "Validation label coverage mismatch")
    for row in rows:
        require(
            all(row[k] == labels[row["qid"]][k] for k in ("qid", "vid", "query", "duration")), "Label identity mismatch"
        )
    for name, digest in plan["evaluator_sha256"].items():
        require(protocol.sha256(Path(plan["evaluator_root"]) / "standalone_eval" / name) == digest, "Evaluator changed")
    model_path = PROJECT / "manifests/reze_model_audit.json"
    require(protocol.sha256(model_path) == plan["model_audit_sha256"], "Model audit changed")
    model = json.loads(model_path.read_text())
    require(model["revision"] == plan["model_revision"], "Model revision changed")
    for name, spec in model["files"].items():
        path = Path(model["model_path"]) / name
        require(
            path.stat().st_size == spec["size"] and protocol.sha256(path) == spec["sha256"], f"Model changed: {name}"
        )
    predictions = {r["qid"]: r for r in protocol.read_jsonl(parent / "predictions.jsonl")}
    require(set(predictions) == set(labels), "B0 prediction coverage mismatch")
    cost = dict.fromkeys(COST_KEYS, 0)
    unavailable = []
    for i, row in enumerate(rows):
        record = protocol.read_baseline(parent, row)
        require(record["prediction"] == predictions[row["qid"]], "B0 prediction differs from raw record")
        require(Path(row["video_path"]).is_file(), "Source video missing")
        if any(
            not eligible_observations(record["media"], c)
            for c in candidates(record["prediction"]["pred_saliency_scores"])
        ):
            unavailable.append(row["qid"])
        for key, value in baseline_cost(record).items():
            cost[key] += value
        if (i + 1) % 250 == 0:
            print(json.dumps({"preflight_baselines_replayed": i + 1}), flush=True)
    return rows, {
        "created_at": protocol.utc_now(),
        "passed": True,
        "queries": len(rows),
        "model_files_verified": len(model["files"]),
        "model_audit_sha256": plan["model_audit_sha256"],
        "model_revision": plan["model_revision"],
        "manifest_sha256": plan["manifest_sha256"],
        "video_sha256": binding["video_sha256"],
        "baseline_raw_cost": cost,
        "candidate_evidence_unavailable_qids": unavailable,
        "video_check": "Existence checked; every used video SHA and decoded frame bytes rechecked against B0 inside worker.",
        "baseline_completion_sha256": plan["baseline_completion_sha256"],
    }


def prepare(plan_path):
    plan = json.loads(plan_path.read_text())
    store = Path(plan["store"])
    run = store / "runs" / plan["run_name"]
    store.mkdir(parents=True, exist_ok=True)
    with (store / "prepare.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (run / "binding.json").exists():
            binding, _ = protocol.load_run(run)
            require(protocol.sha256(plan_path) == binding["plan_sha256"], "Resume plan mismatch")
            return run
        rows, audit = preflight(plan)
        protocol.atomic_json(store / "audit/inputs.json", audit)
        for name in ("records", "logs"):
            (run / name).mkdir(parents=True, exist_ok=True)
        shutil.copy2(plan_path, run / "plan.json")
        parent = Path(plan["baseline_run"])
        shutil.copy2(parent / "inference.jsonl", run / "inference.jsonl")
        sources = json.loads((parent / "binding.json").read_text())["source_sha256"] | protocol.source_hashes()
        extension = "experiments/reze_fullval/runner.py"
        sources[extension] = protocol.sha256(PROJECT / extension)
        for name, digest in sources.items():
            require(protocol.sha256(PROJECT / name) == digest, f"Source drift: {name}")
            target = run / "source_snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(PROJECT / name, target)
        devices = inventory()
        workers = []
        for shard, gpu in enumerate(plan["physical_gpus"]):
            device = devices[gpu]
            require(device["memory_mib"] <= 1000 and device["utilization"] <= 5, f"GPU {gpu} occupied")
            calls = 40 * len(rows[shard :: len(plan["physical_gpus"])])
            workers.append(
                {
                    "physical_index": gpu,
                    "uuid": device["uuid"],
                    "logical_request_cap": calls,
                    "attempt_request_cap": math.ceil(calls * 1.1),
                }
            )
        require(sum(w["logical_request_cap"] for w in workers) == plan["logical_requests_max"], "Call cap mismatch")
        binding = {
            "created_at": protocol.utc_now(),
            "mode": "pair",
            "split": "val1550_frozen_stage_A_extension",
            "manifest_sha256": protocol.sha256(run / "inference.jsonl"),
            "plan_sha256": protocol.sha256(run / "plan.json"),
            "source_sha256": sources,
            "model_audit_sha256": plan["model_audit_sha256"],
            "model_revision": plan["model_revision"],
            "input_audit_sha256": protocol.sha256(store / "audit/inputs.json"),
            "video_sha256": audit["video_sha256"],
            "workers": workers,
            "max_gpu_hours": plan["max_gpu_hours"],
            "stalled_seconds": plan["stalled_seconds"],
            "max_worker_restarts": plan["max_worker_restarts"],
            "parent_run": str(parent),
            "parent_completion_sha256": plan["baseline_completion_sha256"],
            "logical_request_cap": plan["logical_requests_max"],
            "attempt_request_cap": plan["attempt_requests_max"],
            "environment": {
                n: importlib.metadata.version(n)
                for n in ("torch", "torchvision", "transformers", "vllm", "qwen-vl-utils", "decord", "numpy", "scipy")
            },
        }
        protocol.atomic_json(run / "binding.json", binding)
        return run


def audit_evidence(record, baseline):
    for candidate, trace in record["candidate_frames"].items():
        observed = [(f["requested_time"], f["baseline_window"], f["baseline_position"]) for f in trace["frames"]]
        require(observed == eligible_observations(baseline["media"], int(candidate)), "Evidence selection drift")
        require(1 <= trace["frames_observed"] <= 4, "Invalid frame count")
        for frame, index in zip(trace["frames"], trace["metadata"]["frames_indices"], strict=True):
            w, p = frame["baseline_window"], frame["baseline_position"]
            require(frame["frame_index"] == baseline["media"]["frame_indices"][w][p], "Source frame index drift")
            require(frame["frame_timestamps"] == baseline["media"]["frame_timestamps"][w][p], "Source PTS drift")
            require(
                abs(index / trace["metadata"]["fps"] - (frame["requested_time"] - trace["start"])) <= 0.00000051,
                "Timestamp encoding drift",
            )
    first = record["arms"]["P_abs"]["requests"]
    second = record["arms"]["P_pair"]["requests"]
    for a, b in zip(first, second, strict=True):
        require(all(a[k] == b[k] for k in ("a", "b", "evidence_sha256", "model_frames")), "Unmatched arm evidence")
    return len(first)


def metric_arrays(predictions, labels, evaluator_root):
    import numpy as np

    sys.path.insert(0, str(evaluator_root))
    from standalone_eval.eval import compute_ap_from_tuple, eval_submission, mk_gt_scores

    metrics = eval_submission(predictions, labels, verbose=False, match_number=True)
    values = []
    for pred, gt in zip(predictions, labels, strict=True):
        require(pred["qid"] == gt["qid"], "Metric order mismatch")
        scores = np.asarray(pred["pred_saliency_scores"])
        binary = (mk_gt_scores(gt) >= 4).astype(float)
        ap = np.mean([compute_ap_from_tuple((0, w, binary[:, w], scores))[2] for w in range(3)])
        values.append([100 * ap, 100 * binary[int(np.argmax(scores))].max()])
    values = np.asarray(values)
    np.testing.assert_allclose(
        np.round(values.mean(axis=0), 2), [metrics["HL-min-VeryGood"]["HL-mAP"], metrics["HL-min-VeryGood"]["HL-Hit1"]]
    )
    return metrics, values


def evaluate(run):
    import numpy as np
    from summarize import paired_interval

    binding, rows = protocol.load_run(run)
    plan = json.loads((run / "plan.json").read_text())
    require(len(rows) == plan["queries"] == 1550, "Partial validation evaluation forbidden")
    complete = check_complete(run, rows)
    parent = Path(binding["parent_run"])
    parent_complete = check_complete(parent, rows)
    require(
        protocol.sha256(parent / "predictions.jsonl") == plan["baseline_prediction_sha256"], "B0 predictions changed"
    )
    require(protocol.sha256(plan["labels_path"]) == plan["labels_sha256"], "Labels changed")
    for name, digest in plan["evaluator_sha256"].items():
        require(protocol.sha256(Path(plan["evaluator_root"]) / "standalone_eval" / name) == digest, "Evaluator changed")
    by_qid = {r["qid"]: r for r in protocol.read_jsonl(plan["labels_path"])}
    labels = [by_qid[r["qid"]] for r in rows]
    predictions, costs, fallbacks = (
        {a: [] for a in ARMS},
        {a: dict.fromkeys(COST_KEYS, 0) for a in ARMS},
        {a: [] for a in ARMS},
    )
    matched, changed = 0, {a: 0 for a in ARMS[1:]}
    for row in rows:
        baseline = protocol.read_baseline(parent, row)
        record = json.loads((run / "records" / f"{row['qid']}.json").read_text())
        protocol.validate_pair_record(record, row, binding, protocol.sha256(run / "binding.json"), baseline)
        matched += audit_evidence(record, baseline)
        predictions["B0"].append(baseline["prediction"])
        for key, value in baseline_cost(baseline).items():
            costs["B0"][key] += value
        for arm in ARMS[1:]:
            result = record["arms"][arm]
            predictions[arm].append(
                {k: row[k] for k in ("qid", "vid", "query")} | {"pred_saliency_scores": result["scores"]}
            )
            for key in COST_KEYS:
                costs[arm][key] += result["cost"][key]
            for cost_key, raw_key in (
                ("input_tokens", "prompt_tokens"),
                ("output_tokens", "output_tokens"),
                ("model_frames", "model_frames"),
            ):
                require(result["cost"][cost_key] == sum(r[raw_key] for r in result["requests"]), "Raw cost mismatch")
            if result["status"] == "fallback":
                fallbacks[arm].append({"qid": row["qid"], "reason": result["error"]})
            changed[arm] += result["scores"] != baseline["prediction"]["pred_saliency_scores"]
    ledger = [r for path in run.glob("attempts_*.jsonl") for r in protocol.read_jsonl(path)]
    attempted = sum(r["requests"] for r in ledger if r["event"] == "reserved")
    returned = {
        k: sum(r[k] for r in ledger if r["event"] == "returned") for k in ("requests", "input_tokens", "output_tokens")
    }
    require(attempted == complete["attempted_requests"] <= binding["attempt_request_cap"], "Attempt ledger mismatch")
    for key, cost_key in (
        ("requests", "logical_requests"),
        ("input_tokens", "input_tokens"),
        ("output_tokens", "output_tokens"),
    ):
        require(returned[key] >= sum(costs[a][cost_key] for a in ARMS[1:]), "Returned cost ledger lost requests")
    destination = run.parent.parent / "results"
    destination.mkdir(exist_ok=True)
    results, values = {}, {}
    baseline_tokens = costs["B0"]["input_tokens"] + costs["B0"]["output_tokens"]
    for arm in ARMS:
        metrics, values[arm] = metric_arrays(predictions[arm], labels, plan["evaluator_root"])
        if arm == "B0":
            require(metrics == json.loads((parent / "metrics.json").read_text()), "B0 metrics failed to reproduce")
        tokens = costs[arm]["input_tokens"] + costs[arm]["output_tokens"]
        results[arm] = {
            "metrics": metrics,
            "unrounded_VeryGood": dict(zip(("mAP", "Hit1"), values[arm].mean(axis=0).tolist())),
            "cost": costs[arm],
            "fallback_queries": len(fallbacks[arm]),
            "cold_total_tokens": tokens if arm == "B0" else baseline_tokens + tokens,
            "cold_total_token_ratio": 1.0 if arm == "B0" else 1 + tokens / baseline_tokens,
            "tied_max_queries": sum(
                p["pred_saliency_scores"].count(max(p["pred_saliency_scores"])) > 1 for p in predictions[arm]
            ),
        }
        protocol.write_jsonl(destination / f"{arm}_predictions.jsonl", predictions[arm])
    groups = np.asarray([r["vid"].rsplit("_", 2)[0] for r in rows])
    comparisons = {
        f"{a}_minus_{b}": paired_interval(values[a] - values[b], groups, repeats=10000, seed=42)
        for a, b in (("P_abs", "B0"), ("P_pair", "B0"), ("P_pair", "P_abs"))
    }
    report = {
        "evaluated_at": protocol.utc_now(),
        "split": "QVHighlights val1550, previously examined",
        "queries": len(rows),
        "source_videos": len(set(groups)),
        "results": results,
        "paired_comparisons": comparisons,
        "bootstrap": plan["bootstrap"],
        "fallbacks": fallbacks,
        "actual_scores_changed": changed,
        "paired_requests_with_identical_evidence": matched,
        "baseline_reused": True,
        "new_attempted_requests": attempted,
        "new_returned_attempt_cost": returned,
        "new_allocated_gpu_hours": complete["allocated_gpu_hours"],
        "worker_restarts": complete["worker_restarts"],
        "new_elapsed_wall_seconds": (
            datetime.fromisoformat(complete["completed_at"]) - datetime.fromisoformat(binding["created_at"])
        ).total_seconds(),
        "baseline_historical_wall_seconds": parent_complete["wall_seconds_this_launch"],
        "provenance": {
            "binding_sha256": protocol.sha256(run / "binding.json"),
            "plan_sha256": binding["plan_sha256"],
            "baseline_completion_sha256": binding["parent_completion_sha256"],
            "labels_sha256": plan["labels_sha256"],
            "evaluator_sha256": plan["evaluator_sha256"],
        },
        "prior_stage_A_decision": plan["prior_stage_A_decision"],
        "automatic_next_stage": None,
        "limits": [
            "Full-validation extension requested after the failed Stage-A gate; not an untouched test set.",
            "Only reorder B0 top-five original scores; no new candidates, memory, or revisit.",
            "Both arms see A/B videos; P_abs is not single-video absolute scoring.",
            "Joint batch half-time attribution does not measure independent arm latency.",
        ],
    }
    protocol.atomic_json(destination / "evaluation.json", report)
    protocol.write_jsonl(
        destination / "per_query.jsonl",
        [
            {
                "qid": row["qid"],
                "vid": row["vid"],
                **{a: dict(zip(("mAP", "Hit1"), values[a][i].tolist())) for a in ARMS},
            }
            for i, row in enumerate(rows)
        ],
    )
    lines = [
        "# REZE 固定评分方案：全量验证结果",
        "",
        f"完成时间：{report['evaluated_at']}",
        "",
        "QVHighlights 验证集全部 1,550 条查询、474 个源视频。复用冻结 B0，并列报告 P_abs / P_pair。",
        "此前 explore100 未通过 A 阶段门槛；本轮是用户要求的扩大验证，不改变该历史结论。验证集此前已查看。",
        "",
        "| 方法 | VeryGood mAP | Hit@1 | 含 B0 总 token 比 | 回退查询 |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        r = results[arm]
        m = r["metrics"]["HL-min-VeryGood"]
        lines.append(
            f"| {arm} | {m['HL-mAP']:.2f} | {m['HL-Hit1']:.2f} | {r['cold_total_token_ratio']:.3f}× | {r['fallback_queries']} |"
        )
    lines += ["", "## 配对差异", ""]
    for name, result in comparisons.items():
        delta, ci = result["delta_percentage_points"], result["ci95_percentage_points"]
        lines.append(
            f"- {name}：mAP {delta['mAP']:+.3f} pp，95% CI [{ci['mAP'][0]:.3f}, {ci['mAP'][1]:.3f}]；"
            f"Hit@1 {delta['Hit1']:+.3f} pp，95% CI [{ci['Hit1'][0]:.3f}, {ci['Hit1'][1]:.3f}]。"
        )
    lines += [
        "",
        "源视频聚类配对 bootstrap：10,000 次，seed=42。所有查询保留在分母中。",
        "",
        "## 开销与边界",
        "",
        f"新增调用（含重试预留）：{attempted:,}；实际分配 GPU 小时：{complete['allocated_gpu_hours']:.3f}。",
        f"完整运行墙钟时间：{report['new_elapsed_wall_seconds'] / 3600:.3f} 小时。复用 B0 的历史运行时间单独记录。",
        "两组共享批次，分摊时间不代表独立延迟。只置换原前五候选分数；并列分数与候选覆盖仍是限制。",
        "未训练、未引入 memory 或回看；本轮结束后不自动扩展实验。",
        "",
        "## 结果文件",
        "",
        f"- 完整指标、开销与哈希：`{destination / 'evaluation.json'}`",
        f"- 逐查询比较：`{destination / 'per_query.jsonl'}`",
        f"- 原始输出、帧证据与日志：`{run}`",
        "",
    ]
    (destination / "REPORT.md").write_text("\n".join(lines))
    return report


def archive(run):
    plan = json.loads((run / "plan.json").read_text())
    store, target = run.parent.parent, Path(plan["archive"])
    for name in ("audit", "results"):
        shutil.copytree(store / name, target / name, dirs_exist_ok=True)
    archived_run = target / "run"
    archived_run.mkdir(parents=True, exist_ok=True)
    for name in (
        "binding.json",
        "plan.json",
        "inference.jsonl",
        "completion.json",
        "record_receipt.json",
        "supervisor.json",
    ):
        shutil.copy2(run / name, archived_run / name)
    for path in run.glob("attempts_*.jsonl"):
        shutil.copy2(path, archived_run / path.name)
    shutil.copytree(
        run / "source_snapshot",
        archived_run / "source_snapshot",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copy2(store / "results/REPORT.md", plan["result_document"])
    protocol.atomic_json(
        target / "archive_receipt.json",
        {
            "created_at": protocol.utc_now(),
            "raw_run": str(run),
            "sha256": {
                str(p.relative_to(target)): protocol.sha256(p)
                for p in sorted(target.rglob("*"))
                if p.is_file() and p.name != "archive_receipt.json"
            },
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", type=Path)
    parser.add_argument("--run", type=Path)
    args = parser.parse_args()
    if args.prepare:
        require(args.run is None, "Choose prepare or run")
        print(prepare(args.prepare), flush=True)
        return
    require(args.run is not None, "Run path required")
    run = args.run
    binding, _ = protocol.load_run(run)
    store = run.parent.parent
    require(protocol.sha256(store / "audit/inputs.json") == binding["input_audit_sha256"], "Input audit changed")
    for name, version in binding["environment"].items():
        require(importlib.metadata.version(name) == version, f"Environment drift: {name}")
    with (store / "campaign.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def state(status, **extra):
            value = {"time": protocol.utc_now(), "status": status, "run": str(run), **extra}
            protocol.atomic_json(store / "campaign.json", value)
            print(json.dumps(value), flush=True)

        try:
            if (run / "supervisor.json").exists():
                budget = json.loads((run / "supervisor.json").read_text())
                require(
                    all(n <= binding["max_worker_restarts"] for n in budget["worker_restarts"].values()),
                    "Bounded worker retries exhausted",
                )
                require(budget["allocated_gpu_seconds"] <= binding["max_gpu_hours"] * 3600, "GPU budget exhausted")
            state("running", queries=1550)
            run_phase(run)
            state("evaluating", queries=1550)
            evaluate(run)
            archive(run)
            state("complete", queries=1550, automatic_next_stage=None)
        except Exception as error:
            state("failed", error=str(error), traceback=traceback.format_exc())
            raise


if __name__ == "__main__":
    main()
