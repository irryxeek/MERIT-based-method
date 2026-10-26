"""Freeze twenty unlabelled queries, real pixels, model bytes, code and budgets."""

import argparse
import importlib.metadata
import os
import shutil
import subprocess
import sys
from pathlib import Path

from io_utils import HERE, ROOT, fixed_twenty, legacy, parents, read_json, sha256, utc_now, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--execution-plan", type=Path, required=True)
    artifact_root = Path(os.environ.get("ZEROSHOT_ARTIFACT_ROOT", "artifacts/zeroshot"))
    parser.add_argument("--pair-run", type=Path, default=artifact_root / "reze_evidence_v1/runs/stage_a_pairs")
    parser.add_argument("--candidate-decision", type=Path, default=artifact_root / "reze_agent_v2_v1_explore100/decision.json")
    parser.add_argument("--gpu-uuids", nargs="+", help="Optional GPU UUID allowlist; defaults to all visible GPUs")
    parser.add_argument("--repair-from", type=Path)
    parser.add_argument("--repair-plan", type=Path)
    parser.add_argument("--recover-monitor-startup", action="store_true")
    parser.add_argument("--repair-observation-contract", action="store_true")
    parser.add_argument("--repair-observation-content", action="store_true")
    args = parser.parse_args()
    if bool(args.repair_from) != bool(args.repair_plan):
        parser.error("Repair requires both --repair-from and --repair-plan")
    if args.recover_monitor_startup and not args.repair_from:
        parser.error("Monitoring recovery requires the failed repair run")
    if args.repair_observation_contract and (not args.repair_from or args.recover_monitor_startup):
        parser.error("Observation repair requires its failed parent and is a separate declared exception")
    if args.repair_observation_content and (
        not args.repair_from or args.recover_monitor_startup or args.repair_observation_contract
    ):
        parser.error("Observation content diagnostic requires its v4 parent and is an isolated R0-only run")
    run = args.run.resolve()
    if run.exists():
        raise ValueError("Use a new run directory; never overwrite a frozen/failed attempt")
    pair_run = args.pair_run.resolve()
    baseline_run = pair_run.parent / "stage_a_b0"
    decision = args.candidate_decision.resolve()
    if read_json(decision)["selected_candidates"] != "C0":
        raise ValueError("Plan presumes audited C0 selection")
    old_binding, all_rows = legacy.load_run(pair_run)
    if old_binding["split"] != "explore100":
        raise ValueError("Expected explore100 parent")
    rows = fixed_twenty(all_rows)
    binding = {
        "protocol": "revalue-v3-r01-v5",
        "created_at": utc_now(),
        "baseline_run": str(baseline_run),
        "pair_run": str(pair_run),
        "parent_receipts": {str(decision): sha256(decision)},
        "input_sha256": {},
        "source_sha256": {},
        "stage_order": ["R0", "R1"],
        "gpu_candidates": args.gpu_uuids or "all",
        "maximum_active_worker_seconds_per_stage": 7200,
        "maximum_worker_attempts_per_stage": 3,
        "maximum_worker_stall_seconds": 900,
        "queue_expiry_hours": 72,
        "model_revision": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
    }
    for parent in (baseline_run, pair_run):
        complete = read_json(parent / "completion.json")
        if (
            complete["status"] != "complete"
            or complete["queries"] != 100
            or (complete["binding_sha256"] != sha256(parent / "binding.json"))
            or complete["record_receipt_sha256"] != sha256(parent / "record_receipt.json")
        ):
            raise ValueError("Parent is not complete and receipt-verified")
        for name in ("binding.json", "completion.json", "record_receipt.json", "inference.jsonl"):
            binding["parent_receipts"][str(parent / name)] = sha256(parent / name)
    audit, total_tokens = [], 0
    for row in all_rows:
        baseline, pair = parents(binding, row)
        if row in rows:
            tokens = baseline["cost"]["input_tokens"] + baseline["cost"]["output_tokens"]
            total_tokens += tokens
            audit.append(
                {
                    "qid": row["qid"],
                    "candidate_ids": pair["candidate_ids"],
                    "baseline_tokens": tokens,
                    "baseline_record_sha256": sha256(baseline_run / "records" / f"{row['qid']}.json"),
                    "pair_record_sha256": sha256(pair_run / "records" / f"{row['qid']}.json"),
                }
            )
    binding["baseline_tokens"] = total_tokens
    binding["budgets"] = {
        "R0": {"requests": 300, "extra_attempts": 200, "arm_incremental_tokens": {}},
        "R1": {
            "requests": 1360,
            "extra_attempts": 68,
            "arm_incremental_tokens": {a: 5 * total_tokens for a in ("C_direct4", "M_plain", "M_decision")},
        },
    }
    reuse_index = None
    if args.repair_from:
        from repair import inspect_failed_r0, worker_remaining

        reuse_index, history, receipts, limits = inspect_failed_r0(
            args.repair_from,
            rows,
            binding,
            reuse_comparisons=args.repair_observation_contract or args.repair_observation_content,
            content_repair=args.repair_observation_content,
        )
        binding["repair_from"] = str(args.repair_from.resolve())
        binding["historical_usage"] = {"R0": history}
        binding["parent_receipts"].update(receipts)
        binding["budgets"]["R0"] = limits
        if args.recover_monitor_startup:
            from repair import monitor_startup_recovery

            binding["startup_recovery_exception"] = monitor_startup_recovery(args.repair_from)
            binding["maximum_worker_attempts_by_stage"] = {"R0": 4, "R1": 3}
        if args.repair_observation_contract:
            from repair import observation_contract_recovery

            binding["observation_repair_exception"] = observation_contract_recovery(args.repair_from)
            binding["maximum_worker_attempts_by_stage"] = {"R0": 5, "R1": 3}
        if args.repair_observation_content:
            from repair import observation_content_recovery

            binding["observation_content_exception"] = observation_content_recovery(args.repair_from)
            binding["maximum_worker_attempts_by_stage"] = {"R0": 6, "R1": 3}
            binding["stage_order"] = ["R0"]
        seconds, launches = worker_remaining(binding, "R0", [])
        if seconds <= 35 or launches <= 0:
            raise ValueError("Cumulative engineering GPU time or worker-launch budget exhausted")
    binding["environment"] = {
        p: importlib.metadata.version(p)
        for p in ("torch", "torchvision", "transformers", "vllm", "qwen-vl-utils", "decord", "numpy", "pydantic")
    }
    run.mkdir(parents=True)
    if reuse_index is not None:
        write_json(run / "reuse_index.json", reuse_index)
        shutil.copy2(args.repair_plan, run / "repair_plan.md")
    result = subprocess.run([sys.executable, "-m", "pytest", str(HERE / "tests"), "-q"], text=True, capture_output=True)
    (run / "cpu_tests.log").write_text(result.stdout + result.stderr)
    write_json(
        run / "cpu_tests.json",
        {
            "returncode": result.returncode,
            "log_sha256": sha256(run / "cpu_tests.log"),
            "executed_at": utc_now(),
            "command": "python -m pytest experiments/revalue/tests -q",
        },
    )
    if result.returncode:
        raise ValueError("CPU engineering tests failed; preserve this run as failed preparation")
    legacy.write_jsonl(run / "inference.jsonl", rows)
    write_json(run / "parent_audit.json", {"selected_records": audit, "all_100_parents_verified": True})
    import torch
    from candidate_frames import prepare_candidates
    from observer import read_observation

    torch.set_num_threads(2)
    frames = []
    # V4 changes only the observation response contract. Inherit the verified
    # pixel manifest; worker.one_query still redecodes and compares exact pixels
    # before every query, including queries whose unchanged calls are reused.
    preflight_rows = [] if args.repair_observation_contract or args.repair_observation_content else rows
    for index, row in enumerate(preflight_rows):
        baseline, pair = parents(binding, row)
        targets, traces, seconds = prepare_candidates(row, baseline, pair["candidate_ids"])
        if traces != pair["candidate_frames"]:
            raise ValueError("C0 pixels changed")
        start = pair["candidate_ids"][0] * 2.0
        action = {
            "gap": "value",
            "tool": "Segment",
            "question": "What action and visible outcome occur around the target, and which parts remain unclear?",
            "intervals": [[max(0.0, start - 4), min(float(row["duration"]), start + 6)]],
            "frame_limit": 24,
        }
        seen = {i for group in baseline["media"]["frame_indices"] for i in group}
        observation, trace = read_observation(row, action, seen, baseline["media"]["resized_height_width"])
        if trace["displayed_frames"] != 24 or {f["frame_index"] for f in trace["frames"]} & seen:
            raise ValueError("R0 requires exactly 24 real previously unseen frames per query")
        frames.append(
            {"qid": row["qid"], "candidate_frames": traces, "observation": trace, "decode_preprocess_seconds": seconds}
        )
        del targets, observation
        print(f"Frozen exact old/new frame provenance {index + 1}/20", flush=True)
    if args.repair_from:
        previous = legacy.read_jsonl(args.repair_from / "candidate_inputs.jsonl")
        for actual, expected in zip(frames, previous if preflight_rows else [], strict=True):
            for key in ("qid", "candidate_frames", "observation"):
                if actual[key] != expected[key]:
                    raise ValueError(f"Repair changed previously frozen {key}")
        if args.repair_observation_contract or args.repair_observation_content:
            binding["frame_verification"] = {
                "preflight": "inherited byte-identical hash-verified manifest from v3",
                "runtime": "redecode exact candidate and observation pixels/PTS before each query",
            }
        # Preserve byte-identical inputs; current decoding time is not evidence.
        shutil.copy2(args.repair_from / "candidate_inputs.jsonl", run / "candidate_inputs.jsonl")
    else:
        legacy.write_jsonl(run / "candidate_inputs.jsonl", frames)
    model = read_json(ROOT / "manifests/reze_model_audit.json")
    stats = {}
    for name, expected in model["files"].items():
        path = Path(model["model_path"]) / name
        if path.stat().st_size != expected["size"] or sha256(path) != expected["sha256"]:
            raise ValueError(f"Model drift: {name}")
        stats[name] = {"size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
    write_json(run / "model_verification.json", {"audit": model, "stats": stats})
    shutil.copy2(args.execution_plan, run / "execution_plan.md")
    shutil.copy2(pair_run.parents[1] / "audit/inputs_v1.json", run / "evaluation_binding.json")
    write_json(
        run / "plan.json",
        {
            "R0": (
                f"20 queries: verify/reuse {len(reuse_index)} unchanged requests; execute {binding['budgets']['R0']['requests']} remaining comparisons/observations; carry all historical costs"
                if args.repair_from
                else "20 engineering queries; task + 5 facts + 2 memories + 6 comparisons + 1 real 24-frame observation = 300 planned calls"
            ),
            "R0_gate": "all 20 valid, all three arms plus observation; no engineering HD metrics; CPU requirements pass; within budget",
            "R1": "20 queries; E0 cached; C_direct4/M_plain/M_decision; 68 formal new calls per query",
            "sample_selection": "sort SHA256('revalue-v3:' + str(qid)) from prior explore100 and take 20",
            "candidate_selection": "C0 original Top-5; exact parent frame tensors, up to four per candidate",
            "fallback_parent": "E0 audited P_pair scores; full 20-query denominator",
            "formal_prompts": "predeclared once here; frozen after successful R0; format fixes need a new run/version",
            "R0_observation": "deterministic engineering question, not an R2 adaptive policy or accuracy result",
            "next_stage": (
                "stop after R0 observation-only validation; existing v4 R1 continues unchanged"
                if args.repair_observation_content
                else "stop after R1 evaluation; R2/R3 model policy not enabled in this campaign"
            ),
            "no_labels_in_inference": True,
            "human_review": "blank until a human independently reviews",
        },
    )
    names = set(legacy.source_hashes())
    names.update(
        str(p.relative_to(ROOT))
        for p in HERE.rglob("*")
        if p.is_file()
        and p.suffix in {".py", ".md"}
        and "__pycache__" not in p.parts
        and ".pytest_cache" not in p.parts
    )
    for name in sorted(names):
        target = run / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
        binding["source_sha256"][name] = sha256(target)
    for name in (
        "inference.jsonl",
        "parent_audit.json",
        "candidate_inputs.jsonl",
        "model_verification.json",
        "execution_plan.md",
        "evaluation_binding.json",
        "plan.json",
        "cpu_tests.json",
        "cpu_tests.log",
    ):
        binding["input_sha256"][name] = sha256(run / name)
    if args.repair_from:
        for name in ("reuse_index.json", "repair_plan.md"):
            binding["input_sha256"][name] = sha256(run / name)
    write_json(run / "binding.json", binding)
    write_json(run / "status.json", {"status": "prepared_waiting_for_idle_gpu", "updated_at": utc_now(), "stage": "R0"})
    print(f"Prepared {run}; B0 tokens={total_tokens}; per-method incremental cap={5 * total_tokens}")


if __name__ == "__main__":
    main()
