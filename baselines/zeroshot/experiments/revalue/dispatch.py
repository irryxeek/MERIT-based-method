"""Bounded idle-GPU queue; R0 gates R1, with no device reservation."""

import argparse
import fcntl
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from io_utils import HERE, load_run, read_json, utc_now, write_json
from repair import worker_remaining


def idle_gpu(candidates):
    raw = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True
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
        uuid.strip()
        for uuid in subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"], text=True
        ).splitlines()
    }
    if candidates == "all":
        candidates = sorted(inventory, key=lambda uuid: inventory[uuid]["index"])
    elif not isinstance(candidates, list):
        raise ValueError("gpu_candidates must be 'all' or a list of GPU UUIDs")
    for uuid in candidates:
        gpu = inventory.get(uuid)
        if gpu and uuid not in occupied and gpu["memory_mib"] < 512 and gpu["utilization"] < 5:
            return gpu, inventory
    return None, inventory


def environment(gpu):
    return {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(gpu["index"]),
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "REZE_EXPECTED_GPU_UUID": gpu["uuid"],
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
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
    }


def stop(run, reason, stage):
    state = {"status": "stopped", "reason": reason, "stage": stage, "updated_at": utc_now()}
    write_json(run / "stopped.json", state)
    write_json(run / "status.json", state)


def kill_own_child(child):
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        child.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()


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
        elapsed = time.time() - datetime.fromisoformat(binding["created_at"]).timestamp()
        if elapsed > binding["queue_expiry_hours"] * 3600:
            stop(run, "72-hour campaign expiry; no indefinite queue", "queue")
            return
        stages = binding.get("stage_order", ["R0", "R1"])
        if stages not in (["R0"], ["R0", "R1"]):
            raise ValueError("Unsupported execution stage order")
        for stage in stages:
            directory = run / stage
            directory.mkdir(exist_ok=True)
            if stage == "R0" and (directory / "gate.json").exists():
                if read_json(directory / "gate.json")["status"] != "passed":
                    stop(run, "R0 engineering gate failed; inspect immutable failures before a new version", stage)
                    return
                continue
            if stage == "R1" and (directory / "inference_complete.json").exists():
                break
            attempts_path = directory / "worker_attempts.json"
            attempts = read_json(attempts_path) if attempts_path.exists() else []
            for attempt in attempts:
                if attempt["status"] == "running":
                    # Bound one missed heartbeat plus systemd's 35 s cleanup;
                    # never substitute wall time that includes server downtime.
                    attempt.update(status="interrupted", seconds=attempt["seconds"] + 50)
            write_json(attempts_path, attempts)
            remaining, launches_left = worker_remaining(binding, stage, attempts)
            if remaining <= 35 or launches_left <= 0:
                stop(run, "Active GPU time/worker attempt limit; no budget reset", stage)
                return
            gpu, inventory = idle_gpu(binding["gpu_candidates"])
            if gpu is None:
                write_json(
                    run / "status.json",
                    {
                        "status": "waiting_for_idle_gpu",
                        "stage": stage,
                        "updated_at": utc_now(),
                        "gpu_inventory": inventory,
                        "completed_queries": len(list((directory / "records").glob("*.json"))),
                        "expected_queries": len(rows),
                        "binding_sha256": binding_hash,
                    },
                )
                return
            # Second check narrows the gap between observing idle state and starting.
            confirmed, _ = idle_gpu([gpu["uuid"]])
            if confirmed is None:
                return
            attempt = {
                "number": len(attempts) + 1,
                "status": "running",
                "started_at": utc_now(),
                "started_epoch": time.time(),
                "gpu": gpu,
                "seconds": 0.0,
            }
            attempts.append(attempt)
            write_json(attempts_path, attempts)
            started = time.monotonic()
            with (directory / f"worker_{len(attempts)}.log").open("a") as stream:
                child = subprocess.Popen(
                    [sys.executable, str(HERE / "worker.py"), "--run", str(run), "--stage", stage],
                    env=environment(gpu),
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                attempt["pid"] = child.pid
                write_json(attempts_path, attempts)
                write_json(
                    run / "status.json", {"status": "running", "stage": stage, "gpu": gpu, "updated_at": utc_now()}
                )
                try:
                    while True:
                        spent = time.monotonic() - started
                        timeout = min(15, max(0.05, remaining - 35 - spent))
                        try:
                            code = child.wait(timeout=timeout)
                            break
                        except subprocess.TimeoutExpired:
                            attempt["seconds"] = time.monotonic() - started
                            write_json(attempts_path, attempts)
                            status_path = directory / "status.json"
                            activity = max(
                                attempt["started_epoch"],
                                Path(stream.name).stat().st_mtime,
                                status_path.stat().st_mtime if status_path.exists() else 0,
                            )
                            if attempt["seconds"] >= remaining - 35:
                                attempt["stop_reason"] = "active GPU time cap"
                                raise
                            if time.time() - activity >= binding["maximum_worker_stall_seconds"]:
                                attempt["stop_reason"] = "no progress for 900 seconds"
                                raise
                except subprocess.TimeoutExpired:
                    kill_own_child(child)
                    code = 124
                finally:
                    if child.poll() is None:
                        kill_own_child(child)
            attempt.update(
                status="complete" if code == 0 else "failed",
                returncode=code,
                seconds=time.monotonic() - started,
                finished_at=utc_now(),
            )
            write_json(attempts_path, attempts)
            if code:
                if code == 124:
                    stop(run, attempt["stop_reason"], stage)
                    # Preserve a full-denominator partial record, with no further GPU calls.
                    from budget import Ledger
                    from worker import finish

                    ledger = Ledger(directory / "ledger.json", binding["budgets"][stage])
                    finish(run, stage, binding, rows, binding_hash, ledger, attempt["stop_reason"])
                    if stage == "R1":
                        result = subprocess.run(
                            [sys.executable, str(HERE / "evaluate.py"), "--run", str(run)],
                            capture_output=True,
                            text=True,
                            timeout=1800,
                        )
                        (run / "evaluation.log").write_text(result.stdout + result.stderr)
                elif code == 75:
                    write_json(
                        run / "status.json",
                        {
                            "status": "worker_failed_pending_bounded_retry",
                            "stage": stage,
                            "attempt": attempt,
                            "updated_at": utc_now(),
                        },
                    )
                else:
                    stop(run, f"Non-retriable implementation/provenance failure, exit {code}", stage)
                return
            if stage == "R0" and read_json(directory / "gate.json")["status"] != "passed":
                stop(run, "R0 engineering gate failed; R1 not started", stage)
                return
        if stages == ["R0"]:
            state = {
                "status": "engineering_complete",
                "stage": "R0",
                "updated_at": utc_now(),
                "binding_sha256": binding_hash,
                "R1_started": False,
            }
            write_json(run / "complete.json", state)
            write_json(run / "status.json", state)
            return
        result = subprocess.run(
            [sys.executable, str(HERE / "evaluate.py"), "--run", str(run)], capture_output=True, text=True, timeout=1800
        )
        (run / "evaluation.log").write_text(result.stdout + result.stderr)
        if result.returncode:
            stop(run, f"Official evaluation failed with code {result.returncode}", "R1")


if __name__ == "__main__":
    main()
