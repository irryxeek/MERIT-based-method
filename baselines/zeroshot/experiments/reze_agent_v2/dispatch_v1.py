"""Single-GPU, timer-driven V1 scheduling after the QFVS predecessor exits."""

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from core import legacy, load_run

HERE = Path(__file__).resolve().parent


def idle_gpu(candidates):
    occupied = set(
        subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"], text=True
        ).splitlines()
    )
    inventory = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True
    )
    for line in inventory.splitlines():
        index, uuid, memory, utilization = [x.strip() for x in line.split(",")]
        if uuid in candidates and uuid not in occupied and int(memory) < 512 and int(utilization) < 5:
            return {"index": int(index), "uuid": uuid}
    return None


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
        parent = Path(binding["after_run"])
        if legacy.sha256(parent / "binding.json") != binding["after_binding_sha256"]:
            raise ValueError("Predecessor identity changed")
        if not any((parent / name).exists() for name in ("comparison.json", "stopped.json")):
            legacy.atomic_json(
                run / "status.json",
                {
                    "status": "waiting_for_Q1",
                    "updated_at": legacy.utc_now(),
                    "completed_queries": 0,
                    "expected_queries": len(rows),
                },
            )
            return
        if not (run / "inference_complete.json").exists():
            attempts_path = run / "worker_attempts.json"
            attempts = json.loads(attempts_path.read_text()) if attempts_path.exists() else []
            for attempt in attempts:
                if attempt["status"] == "running":
                    attempt.update(status="interrupted", seconds=max(0, time.time() - attempt["started_epoch"]))
            remaining = binding["maximum_active_worker_seconds"] - sum(a["seconds"] for a in attempts)
            if remaining <= 0 or len(attempts) >= binding["maximum_worker_attempts"]:
                state = {"status": "stopped_at_budget", "updated_at": legacy.utc_now(), "attempts": attempts}
                legacy.atomic_json(run / "stopped.json", state)
                legacy.atomic_json(run / "status.json", state)
                return
            gpu = idle_gpu(binding["gpu_candidates"])
            if gpu is None:
                legacy.atomic_json(
                    run / "status.json", {"status": "waiting_for_idle_gpu", "updated_at": legacy.utc_now()}
                )
                return
            attempt = {
                "number": len(attempts) + 1,
                "status": "running",
                "seconds": 0,
                "started_epoch": time.time(),
                "started_at": legacy.utc_now(),
                "gpu": gpu,
            }
            attempts.append(attempt)
            legacy.atomic_json(attempts_path, attempts)
            env = {
                **os.environ,
                "CUDA_VISIBLE_DEVICES": str(gpu["index"]),
                "REZE_EXPECTED_GPU_UUID": gpu["uuid"],
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "VLLM_NO_USAGE_STATS": "1",
                "WANDB_MODE": "disabled",
                "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                "OMP_NUM_THREADS": "2",
                "MKL_NUM_THREADS": "2",
                "TOKENIZERS_PARALLELISM": "false",
                "PYTHONUNBUFFERED": "1",
            }
            before = time.monotonic()
            with (run / f"worker_attempt_{len(attempts)}.log").open("a") as log:
                child = subprocess.Popen(
                    [sys.executable, str(HERE / "worker_v1.py"), "--run", str(run)],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                attempt["pid"] = child.pid
                legacy.atomic_json(attempts_path, attempts)
                try:
                    while True:
                        try:
                            code = child.wait(timeout=15)
                            break
                        except subprocess.TimeoutExpired:
                            last = max(
                                (run / "status.json").stat().st_mtime,
                                Path(log.name).stat().st_mtime,
                                attempt["started_epoch"],
                            )
                            if time.monotonic() - before >= remaining or time.time() - last >= 900:
                                attempt["stop_reason"] = "active_time_or_stall_budget"
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
                finished_at=legacy.utc_now(),
            )
            legacy.atomic_json(attempts_path, attempts)
            if code:
                if not (run / "stopped.json").exists():
                    legacy.atomic_json(
                        run / "status.json",
                        {
                            "status": "worker_failed_pending_bounded_retry",
                            "updated_at": legacy.utc_now(),
                            "attempt": attempt,
                        },
                    )
                return
        result = subprocess.run(
            [sys.executable, str(HERE / "evaluate_v1.py"), "--run", str(run)],
            capture_output=True,
            text=True,
            timeout=1800,
        )
        (run / "evaluation.log").write_text(result.stdout + result.stderr)
        if result.returncode:
            state = {"status": "evaluation_failed", "updated_at": legacy.utc_now(), "returncode": result.returncode}
            legacy.atomic_json(run / "stopped.json", state)
            legacy.atomic_json(run / "status.json", state)


if __name__ == "__main__":
    main()
