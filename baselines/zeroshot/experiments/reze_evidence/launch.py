"""Bounded, resumable stage-A campaign supervisor, suitable for user systemd."""

import argparse
import fcntl
import importlib.metadata
import json
import math
import os
import shutil
import signal
import subprocess
import time
import traceback

import protocol
from reze_core import segments_for


def inventory():
    raw = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    result = {}
    for line in raw.splitlines():
        index, uuid, memory, utilization = [p.strip() for p in line.split(",")]
        result[int(index)] = {
            "physical_index": int(index),
            "uuid": uuid,
            "memory_mib": int(memory),
            "utilization": int(utilization),
        }
    return result


def create_run(name, mode, split, rows, gpus, max_gpu_hours, audit, parent=None):
    run = protocol.STORE / "runs" / name
    if (run / "binding.json").exists():
        binding, old_rows = protocol.load_run(run, check_current=False)
        if rows != old_rows or binding["mode"] != mode or binding["split"] != split:
            raise ValueError("Resume specification mismatch")
        for source in protocol.BASE_SOURCES:
            if protocol.sha256(protocol.PROJECT / source) != binding["source_sha256"][source]:
                raise ValueError("Frozen baseline dependency changed")
        return run
    protocol.validate_rows(rows)
    run.mkdir(parents=True, exist_ok=True)
    (run / "records").mkdir(exist_ok=True)
    (run / "logs").mkdir(exist_ok=True)
    protocol.write_jsonl(run / "inference.jsonl", rows)
    shutil.copy2(protocol.PLAN, run / "plan.json")
    sources = protocol.source_hashes()
    for name in sources:
        destination = run / "source_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(protocol.PROJECT / name, destination)
    devices = inventory()
    workers = []
    for shard, gpu in enumerate(gpus):
        subset = rows[shard :: len(gpus)]
        nominal = sum(
            len(segments_for(row["duration"])) if mode == "baseline" else 10 if mode == "stress" else 40
            for row in subset
        )
        workers.append(
            {
                "physical_index": gpu,
                "uuid": devices[gpu]["uuid"],
                "logical_request_cap": nominal,
                "attempt_request_cap": math.ceil(nominal * 1.1),
            }
        )
    binding = {
        "created_at": protocol.utc_now(),
        "mode": mode,
        "split": split,
        "manifest_sha256": protocol.sha256(run / "inference.jsonl"),
        "plan_sha256": protocol.sha256(run / "plan.json"),
        "source_sha256": sources,
        "model_audit_sha256": audit["model_audit_sha256"],
        "model_revision": audit["model_revision"],
        "input_audit_sha256": protocol.sha256(protocol.STORE / "audit/inputs_v1.json"),
        "video_sha256": {row["vid"]: audit["video_sha256"][row["vid"]] for row in rows},
        "workers": workers,
        "max_gpu_hours": max_gpu_hours,
        "stalled_seconds": 900,
        "max_worker_restarts": 2,
        "environment": {
            name: importlib.metadata.version(name)
            for name in ("torch", "torchvision", "transformers", "vllm", "qwen-vl-utils", "decord", "numpy", "scipy")
        },
        "parent_run": str(parent) if parent else None,
        "parent_completion_sha256": protocol.sha256(parent / "completion.json") if parent else None,
        "logical_request_cap": sum(w["logical_request_cap"] for w in workers),
        "attempt_request_cap": math.floor(sum(w["logical_request_cap"] for w in workers) * 1.1),
    }
    protocol.atomic_json(run / "binding.json", binding)
    return run


def stop_group(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def complete_run(run, binding, rows, budget):
    receipt = {}
    for row in rows:
        path = run / "records" / f"{row['qid']}.json"
        record = json.loads(path.read_text())
        if binding["mode"] == "baseline":
            protocol.validate_baseline(record, row, binding, protocol.sha256(run / "binding.json"))
        elif binding["mode"] == "stress":
            protocol.validate_stress_record(record, row, binding, protocol.sha256(run / "binding.json"))
        else:
            baseline = protocol.read_baseline(binding["parent_run"], row)
            protocol.validate_pair_record(record, row, binding, protocol.sha256(run / "binding.json"), baseline)
        receipt[str(row["qid"])] = protocol.sha256(path)
    protocol.atomic_json(run / "record_receipt.json", receipt)
    attempts = sum(
        r["requests"] for p in run.glob("attempts_*.jsonl") for r in protocol.read_jsonl(p) if r["event"] == "reserved"
    )
    if attempts > binding["attempt_request_cap"]:
        raise ValueError("Global attempt cap exceeded")
    completion = {
        "status": "complete",
        "completed_at": protocol.utc_now(),
        "queries": len(rows),
        "binding_sha256": protocol.sha256(run / "binding.json"),
        "record_receipt_sha256": protocol.sha256(run / "record_receipt.json"),
        "attempted_requests": attempts,
        "allocated_gpu_hours": budget["allocated_gpu_seconds"] / 3600,
        "worker_restarts": budget["worker_restarts"],
    }
    protocol.atomic_json(run / "completion.json", completion)
    return completion


def run_phase(run):
    # Completed engineering B0 remains reusable after an isolated pair-input fix.
    # Workers execute their own verified snapshot, never mutable live code.
    binding, rows = protocol.load_run(run, check_current=False)
    if (run / "completion.json").exists():
        receipt = json.loads((run / "record_receipt.json").read_text())
        completion = json.loads((run / "completion.json").read_text())
        if completion["binding_sha256"] != protocol.sha256(run / "binding.json"):
            raise ValueError("Completed binding changed")
        if completion["record_receipt_sha256"] != protocol.sha256(run / "record_receipt.json"):
            raise ValueError("Completed receipt changed")
        if any(protocol.sha256(run / "records" / f"{qid}.json") != value for qid, value in receipt.items()):
            raise ValueError("Completed query changed")
        return completion
    budget_path = run / "supervisor.json"
    budget = (
        json.loads(budget_path.read_text())
        if budget_path.exists()
        else {
            "allocated_gpu_seconds": 0.0,
            "worker_restarts": {str(i): 0 for i in range(len(binding["workers"]))},
        }
    )
    processes, logs = {}, []

    def start(shard):
        spec = binding["workers"][shard]
        device = inventory()[spec["physical_index"]]
        if device["uuid"] != spec["uuid"] or device["memory_mib"] > 1000 or device["utilization"] > 5:
            raise RuntimeError(f"Reserved GPU is unavailable; leaving other jobs alone: {device}")
        log = (run / "logs" / f"worker_{shard}_attempt_{budget['worker_restarts'][str(shard)]}.log").open("a")
        logs.append(log)
        environment = os.environ.copy()
        environment.update(
            CUDA_DEVICE_ORDER="PCI_BUS_ID",
            CUDA_VISIBLE_DEVICES=str(spec["physical_index"]),
            OMP_NUM_THREADS="2",
            MKL_NUM_THREADS="2",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            TOKENIZERS_PARALLELISM="false",
            PYTHONUNBUFFERED="1",
            VLLM_WORKER_MULTIPROC_METHOD="spawn",
        )
        script = run / "source_snapshot/experiments/reze_evidence/worker.py"
        process = subprocess.Popen(
            [protocol.PYTHON, str(script), "--run", str(run), "--shard", str(shard)],
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        processes[shard] = {"process": process, "started": time.time(), "done": False}

    started = last = time.time()
    try:
        for shard in range(len(binding["workers"])):
            start(shard)
        while True:
            now = time.time()
            active_count = sum(not p["done"] for p in processes.values())
            budget["allocated_gpu_seconds"] += (now - last) * active_count
            last = now
            if budget["allocated_gpu_seconds"] > binding["max_gpu_hours"] * 3600:
                raise RuntimeError("Allocated GPU-hour cap reached; phase remains incomplete")
            attempts = sum(
                r["requests"]
                for path in run.glob("attempts_*.jsonl")
                for r in protocol.read_jsonl(path)
                if r["event"] == "reserved"
            )
            if attempts > binding["attempt_request_cap"]:
                raise RuntimeError("Attempt cap reached")
            all_done = True
            for shard, state in list(processes.items()):
                if state["done"]:
                    continue
                process = state["process"]
                status_path = run / f"worker_{shard}.json"
                status = json.loads(status_path.read_text()) if status_path.exists() else {}
                code = process.poll()
                stamp = max(state["started"], status_path.stat().st_mtime if status_path.exists() else 0)
                stalled = now - stamp > binding["stalled_seconds"]
                if code == 0 and status.get("status") == "complete":
                    state["done"] = True
                    stop_group(process)  # clean up only descendants of our worker process group
                    continue
                all_done = False
                if code is not None or stalled:
                    stop_group(process)
                    budget["worker_restarts"][str(shard)] += 1
                    protocol.atomic_json(budget_path, budget | {"status": "recovering", "time": protocol.utc_now()})
                    if budget["worker_restarts"][str(shard)] > binding["max_worker_restarts"]:
                        raise RuntimeError(f"Shard {shard} exhausted bounded restarts (exit={code}, stalled={stalled})")
                    time.sleep(3)
                    start(shard)
            protocol.atomic_json(
                budget_path,
                budget
                | {
                    "status": "running",
                    "time": protocol.utc_now(),
                    "wall_seconds_current_session": now - started,
                    "attempted_requests": attempts,
                },
            )
            if all_done:
                break
            time.sleep(5)
    finally:
        for state in processes.values():
            stop_group(state["process"])
        for log in logs:
            log.close()
        protocol.atomic_json(budget_path, budget | {"time": protocol.utc_now(), "status": "workers_stopped"})
    return complete_run(run, binding, rows, budget)


def engineering_gate(baseline, stress):
    from candidate_frames import eligible_observations

    _, rows = protocol.load_run(stress)
    if len(rows) != 21:
        raise ValueError("Engineering sustained check must cover 21 queries")
    records = [json.loads((stress / "records" / f"{r['qid']}.json").read_text()) for r in rows]
    memory = [r["gpu_memory_mib"] for r in records]
    pairs = protocol.STORE / "runs/engineering_pairs_v2"
    corrected = protocol.STORE / "runs/engineering_tail_fix_v3"
    for row in rows:
        source = corrected if row["qid"] == 688 else pairs
        full = json.loads((source / "records" / f"{row['qid']}.json").read_text())
        source_binding, _ = protocol.load_run(source, check_current=False)
        parent = protocol.read_baseline(baseline, row)
        protocol.validate_pair_record(full, row, source_binding, protocol.sha256(source / "binding.json"), parent)
        if any(full["arms"][arm]["status"] != "complete" for arm in ("P_abs", "P_pair")):
            raise ValueError("Engineering full-comparison output remains invalid")
        for candidate in full["candidate_ids"]:
            expected = eligible_observations(parent["media"], candidate)
            actual = [
                (f["requested_time"], f["baseline_window"], f["baseline_position"])
                for f in full["candidate_frames"][str(candidate)]["frames"]
            ]
            if actual != expected:
                raise ValueError("Cached engineering output used different observations")
    engineering_runs = sorted((protocol.STORE / "runs").glob("engineering*"))
    attempted = sum(
        r["requests"]
        for run in engineering_runs
        for path in run.glob("attempts_*.jsonl")
        for r in protocol.read_jsonl(path)
        if r["event"] == "reserved"
    )
    gpu_seconds = sum(
        json.loads((run / "supervisor.json").read_text())["allocated_gpu_seconds"]
        for run in engineering_runs
        if (run / "supervisor.json").exists()
    )
    checks = {
        "all_21_full_comparisons_valid_after_tail_fix": True,
        "all_21_sustained_native_stress_checks_valid": all(r["status"] == "complete" for r in records),
        "validated_profile_no_worker_restart": all(
            v == 0
            for run in (baseline, pairs, corrected, stress)
            for v in json.loads((run / "completion.json").read_text())["worker_restarts"].values()
        ),
        "late_memory_growth_below_512MiB": max(memory[-5:]) - max(memory[5:10]) <= 512,
        "reserved_GPU_memory_below_23500MiB": max(memory) < 23500,
        "all_engineering_attempts_including_failed_version_below_2500": attempted <= 2500,
        "all_engineering_GPU_hours_below_4": gpu_seconds <= 4 * 3600,
    }
    result = {
        "time": protocol.utc_now(),
        "checks": checks,
        "passed": all(checks.values()),
        "query_gpu_memory_mib": memory,
        "accuracy_not_evaluated": True,
        "all_engineering_attempted_requests": attempted,
        "all_engineering_allocated_gpu_hours": gpu_seconds / 3600,
        "stress_requests_per_query": 10,
        "stress_design": "five-candidate cycle in both arms; all candidates appear in both video positions",
        "full_comparison_design": "Reuse 20 byte-identical full engineering outputs; rerun all 40 calls for qid 688",
    }
    protocol.atomic_json(protocol.STORE / "engineering_gate.json", result)
    if not result["passed"]:
        raise RuntimeError("Engineering gate failed; preserve records and diagnose before stage A")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", required=True, help="Comma-separated allocated numeric GPU indices")
    args = parser.parse_args()
    gpus = [int(p) for p in args.gpus.split(",")]
    if not gpus or len(set(gpus)) != len(gpus) or any(g < 0 for g in gpus):
        raise ValueError("Unexpected GPU assignment")
    protocol.STORE.mkdir(parents=True, exist_ok=True)
    lock = (protocol.STORE / "campaign.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = protocol.STORE / "campaign.json"

    def state(phase, **extra):
        value = {"time": protocol.utc_now(), "phase": phase, **extra}
        protocol.atomic_json(status_path, value)
        print(json.dumps(value, ensure_ascii=False), flush=True)

    plan = json.loads(protocol.PLAN.read_text())
    audit = json.loads((protocol.STORE / "audit/inputs_v1.json").read_text())
    dataset = plan["datasets"]["explore100"]
    if protocol.sha256(dataset["path"]) != dataset["sha256"] or audit["manifest_sha256"] != dataset["sha256"]:
        raise ValueError("Frozen exploration split changed")
    rows = protocol.read_jsonl(dataset["path"])
    engineering_spec = plan["datasets"]["engineering20"]
    if protocol.sha256(engineering_spec["path"]) != engineering_spec["sha256"]:
        raise ValueError("Frozen engineering split changed")
    engineering20 = protocol.read_jsonl(engineering_spec["path"])
    by_qid = {row["qid"]: row for row in rows}
    engineering21 = [by_qid[r["qid"]] for r in engineering20] + [
        by_qid[engineering_spec["extra_query_for_sustained21"]]
    ]
    try:
        for name, mode, split, data, devices, gpu_hours, parent_name in (
            ("engineering_b0", "baseline", "engineering21", engineering21, gpus[:1], 2.0, None),
            ("engineering_pairs_v2", "pair", "engineering21", engineering21, gpus[:1], 2.0, "engineering_b0"),
            ("engineering_tail_fix_v3", "pair", "engineering_tail_fix", [by_qid[688]], gpus[:1], 0.1, "engineering_b0"),
            ("engineering_stress21_v3", "stress", "engineering21", engineering21, gpus[:1], 0.5, "engineering_b0"),
            ("stage_a_b0", "baseline", "explore100", rows, gpus, 1.0, None),
            ("stage_a_pairs", "pair", "explore100", rows, gpus, 1.0, "stage_a_b0"),
        ):
            parent = protocol.STORE / "runs" / parent_name if parent_name else None
            state(name, status="running")
            run = create_run(name, mode, split, data, devices, gpu_hours, audit, parent)
            run_phase(run)
            if name == "engineering_stress21_v3":
                engineering_gate(parent, run)
        state("stage_a_evaluation", status="running")
        from stage_a_eval import evaluate

        destination = protocol.STORE / "stage_a_results"
        report = evaluate(protocol.STORE / "runs/stage_a_b0", protocol.STORE / "runs/stage_a_pairs", destination)
        archive = protocol.PROJECT / "results/reze_evidence_v1"
        archive.mkdir(parents=True, exist_ok=True)
        shutil.copytree(destination, archive / "stage_a", dirs_exist_ok=True)
        shutil.copy2(protocol.STORE / "engineering_gate.json", archive / "engineering_gate.json")
        for name in (
            "engineering_b0",
            "engineering_pairs_v2",
            "engineering_tail_fix_v3",
            "engineering_stress21_v3",
            "stage_a_b0",
            "stage_a_pairs",
        ):
            run = protocol.STORE / "runs" / name
            out = archive / "runs" / name
            out.mkdir(parents=True, exist_ok=True)
            for filename in ("binding.json", "inference.jsonl", "completion.json", "record_receipt.json", "plan.json"):
                shutil.copy2(run / filename, out / filename)
        shutil.copy2(
            destination / "REPORT.md", (os.environ.get("MERIT_ROOT", ".") + "/docs/reze_stage_a_results_2026-10-19.md")
        )
        state(
            "stage_a_complete",
            status="stopped_by_prespecified_gate" if not report["gate"]["selected_arm"] else "stage_B_required",
            gate=report["gate"],
            result_path=str(destination),
        )
    except Exception as error:
        state("interrupted", status="requires_diagnosis", error=str(error), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
