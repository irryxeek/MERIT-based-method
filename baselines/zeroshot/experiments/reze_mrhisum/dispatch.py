"""Persistent bounded queue using numeric GPU indices with UUID verification."""

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from shared import HERE, atomic_json, load_run, sha256, utc_now


def free_gpu(candidates):
    raw = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True,
    )
    inventory = {}
    for line in raw.splitlines():
        index, uuid, memory, utilization = [v.strip() for v in line.split(",")]
        inventory[uuid] = {
            "index": int(index),
            "uuid": uuid,
            "memory_mib": int(memory),
            "utilization": int(utilization),
        }
    occupied = {
        line.strip()
        for line in subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"], text=True
        ).splitlines()
    }
    for uuid in candidates:
        gpu = inventory.get(uuid)
        if gpu and uuid not in occupied and gpu["memory_mib"] < 512 and gpu["utilization"] < 5:
            return gpu, inventory
    return None, inventory


def worker_environment(gpu):
    return {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(gpu["index"]),
        "REZE_EXPECTED_GPU_UUID": gpu["uuid"],
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "VLLM_NO_USAGE_STATS": "1",
        "WANDB_MODE": "disabled",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "PYTHONUNBUFFERED": "1",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    with (run / "dispatch.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        if (run / "complete.json").exists() or (run / "stopped.json").exists():
            return
        binding, rows, binding_hash = load_run(run)
        preflight = json.loads((run / "media_preflight.json").read_text())
        if preflight["status"] != "passed" or preflight["binding_sha256"] != binding_hash:
            raise ValueError("Native media preflight has not passed for this run")
        if not (run / "inference_complete.json").exists():
            attempts_path = run / "worker_attempts.json"
            attempts = json.loads(attempts_path.read_text()) if attempts_path.exists() else []
            for attempt in attempts:
                if attempt["status"] == "running":
                    attempt.update(status="interrupted", seconds=max(0, time.time() - attempt["started_epoch"]))
            remaining = binding["maximum_active_worker_seconds"] - sum(a["seconds"] for a in attempts)
            if len(attempts) >= binding["maximum_worker_attempts"] or remaining <= 0:
                state = {"status": "stopped_at_budget", "updated_at": utc_now(), "attempts": attempts}
                atomic_json(run / "stopped.json", state)
                atomic_json(run / "status.json", state)
                return
            gpu, inventory = free_gpu(binding["gpu_candidates"])
            if gpu is None:
                atomic_json(
                    run / "status.json",
                    {
                        "status": "waiting_for_idle_gpu",
                        "updated_at": utc_now(),
                        "gpu_inventory": inventory,
                        "completed_queries": len(list((run / "predictions").glob("*.json"))),
                        "expected_queries": len(rows),
                        "planned_requests": binding["expected_model_requests"],
                    },
                )
                return
            attempt = {
                "number": len(attempts) + 1,
                "status": "running",
                "started_epoch": time.time(),
                "started_at": utc_now(),
                "gpu": gpu,
                "seconds": 0,
            }
            attempts.append(attempt)
            atomic_json(attempts_path, attempts)
            before = time.monotonic()
            with (run / f"worker_attempt_{len(attempts)}.log").open("a") as log:
                child = subprocess.Popen(
                    [sys.executable, str(HERE / "worker.py"), "--run", str(run)],
                    env=worker_environment(gpu),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                attempt["pid"] = child.pid
                atomic_json(attempts_path, attempts)
                try:
                    while True:
                        try:
                            code = child.wait(timeout=min(15, max(0.1, remaining)))
                            break
                        except subprocess.TimeoutExpired:
                            last_activity = max(
                                (run / "status.json").stat().st_mtime,
                                Path(log.name).stat().st_mtime,
                                attempt["started_epoch"],
                            )
                            if time.monotonic() - before >= remaining:
                                attempt["stop_reason"] = "active_time_budget"
                                raise
                            if time.time() - last_activity >= binding["maximum_worker_stall_seconds"]:
                                attempt["stop_reason"] = "no_progress_for_900_seconds"
                                raise
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                    code = 124
            attempt.update(
                status="complete" if code == 0 else "failed",
                returncode=code,
                seconds=time.monotonic() - before,
                finished_at=utc_now(),
            )
            atomic_json(attempts_path, attempts)
            if code:
                atomic_json(
                    run / "status.json",
                    {
                        "status": "worker_failed_pending_bounded_retry",
                        "updated_at": utc_now(),
                        "attempt": attempt,
                    },
                )
                return
        result = subprocess.run(
            [sys.executable, str(HERE / "evaluate.py"), "--run", str(run)],
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
            capture_output=True,
            text=True,
            timeout=1800,
        )
        (run / "evaluation.log").write_text(result.stdout + result.stderr)
        if result.returncode:
            state = {"status": "evaluation_failed", "updated_at": utc_now(), "returncode": result.returncode}
            atomic_json(run / "stopped.json", state)
            atomic_json(run / "status.json", state)
        else:
            atomic_json(
                run / "dispatch_completion.json",
                {
                    "finished_at": utc_now(),
                    "evaluation_sha256": sha256(run / "evaluation.json"),
                },
            )


if __name__ == "__main__":
    main()
