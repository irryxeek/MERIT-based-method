"""Use idle GPUs with disjoint query claims and one six-GPU-hour stage budget."""

import argparse
import fcntl
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from dispatch import environment, idle_gpu
from io_utils import HERE, read_json, utc_now, write_json
from r2_protocol import load_run
from r2_worker import finish


def active_seconds(attempts):
    return sum(a["seconds"] for a in attempts)


def stop_children(live):
    for child, _, _, _ in live.values():
        if child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 35
    for child, _, _, _ in live.values():
        try:
            child.wait(timeout=max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()


def evaluate(run):
    result = subprocess.run(
        [sys.executable, str(HERE / "r2_evaluate.py"), "--run", str(run)], capture_output=True, text=True, timeout=1800
    )
    (run / "evaluation.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        state = {"status": "evaluation_failed", "code": result.returncode, "updated_at": utc_now()}
        write_json(run / "stopped.json", state)
        write_json(run / "status.json", state)


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
        directory = run / "R2"
        if (directory / "inference_complete.json").exists():
            evaluate(run)
            return
        attempts_path = directory / "worker_attempts.json"
        attempts = read_json(attempts_path) if attempts_path.exists() else []
        for attempt in attempts:
            if attempt["status"] == "running":
                proc = Path(f"/proc/{attempt['pid']}/cmdline")
                if proc.exists() and str(run).encode() in proc.read_bytes() and b"r2_worker.py" in proc.read_bytes():
                    raise RuntimeError("An orphan R2 worker is still alive; do not duplicate its workload")
                attempt.update(status="interrupted", seconds=attempt["seconds"] + 50)
        write_json(attempts_path, attempts)
        live = {}
        reason = None
        try:
            while True:
                now = time.monotonic()
                for number, (child, attempt, start, stream) in list(live.items()):
                    attempt["seconds"] = now - start
                    code = child.poll()
                    if code is not None:
                        attempt.update(
                            status="complete" if code == 0 else "failed", returncode=code, finished_at=utc_now()
                        )
                        stream.close()
                        del live[number]
                        if code not in (0, 75):
                            reason = f"Non-retriable R2 worker failure: {number}, exit {code}"
                    else:
                        status = directory / "workers" / f"{number}.status.json"
                        activity = max(
                            attempt["started_epoch"],
                            Path(stream.name).stat().st_mtime,
                            status.stat().st_mtime if status.exists() else 0,
                        )
                        if time.time() - activity >= binding["maximum_worker_stall_seconds"]:
                            reason = f"Worker {number} made no progress for 900 seconds"
                write_json(attempts_path, attempts)
                completed = len(list((directory / "records").glob("*.json")))
                if completed == len(rows) and not live:
                    break
                if (directory / "stop_request.json").exists():
                    reason = read_json(directory / "stop_request.json")["reason"]
                elapsed = time.time() - datetime.fromisoformat(binding["created_at"]).timestamp()
                if elapsed > binding["queue_expiry_hours"] * 3600:
                    reason = "72-hour campaign expiry"
                remaining = binding["maximum_active_worker_seconds"] - active_seconds(attempts)
                if remaining <= max(60, 60 * len(live)):
                    reason = "Summed active GPU time cap, including shutdown reserve"
                if not live and len(attempts) >= binding["maximum_worker_launches"]:
                    reason = "Bounded worker launch budget exhausted"
                if reason:
                    write_json(directory / "stop_request.json", {"reason": reason, "at": utc_now()})
                    stop_children(live)
                    for child, attempt, start, stream in live.values():
                        attempt.update(
                            seconds=time.monotonic() - start,
                            status="stopped",
                            returncode=child.returncode,
                            finished_at=utc_now(),
                            stop_reason=reason,
                        )
                        stream.close()
                    live.clear()
                    write_json(attempts_path, attempts)
                    break
                probe_error = None
                while (
                    len(live) < min(binding["maximum_concurrent_workers"], len(rows) - completed)
                    and len(attempts) < binding["maximum_worker_launches"]
                    and remaining > 60 * (len(live) + 1) + 60
                ):
                    try:
                        _, inventory = idle_gpu("all")
                        occupied = {a[1]["gpu"]["uuid"] for a in live.values()}
                        candidates = sorted(
                            (k for k in inventory if k not in occupied), key=lambda k: inventory[k]["index"]
                        )
                        gpu, _ = idle_gpu(candidates)
                        if gpu is None or idle_gpu([gpu["uuid"]])[0] is None:
                            break
                    except (OSError, ValueError, subprocess.CalledProcessError) as error:
                        probe_error = str(error)
                        break
                    number = str(len(attempts) + 1)
                    (directory / "workers").mkdir(exist_ok=True)
                    stream = (directory / "workers" / f"{number}.log").open("a")
                    start = time.monotonic()
                    child = subprocess.Popen(
                        [sys.executable, str(HERE / "r2_worker.py"), "--run", str(run), "--worker-id", number],
                        env=environment(gpu),
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    attempt = {
                        "number": number,
                        "status": "running",
                        "started_at": utc_now(),
                        "started_epoch": time.time(),
                        "gpu": gpu,
                        "seconds": 0.0,
                        "pid": child.pid,
                    }
                    attempts.append(attempt)
                    live[number] = (child, attempt, start, stream)
                    write_json(attempts_path, attempts)
                write_json(
                    run / "status.json",
                    {
                        "status": "running" if live else "waiting_for_idle_gpu",
                        "stage": "R2",
                        "updated_at": utc_now(),
                        "completed_queries": completed,
                        "expected_queries": len(rows),
                        "binding_sha256": binding_hash,
                        "active_gpus": [v[1]["gpu"] for v in live.values()],
                        "active_workers": len(live),
                        "summed_gpu_seconds": active_seconds(attempts),
                        "gpu_probe_error": probe_error,
                    },
                )
                if not live:
                    return
                time.sleep(10)
        finally:
            if live:
                stop_children(live)
                for child, attempt, start, stream in live.values():
                    attempt.update(seconds=time.monotonic() - start, status="interrupted", returncode=child.returncode)
                    stream.close()
                write_json(attempts_path, attempts)
        finish(run, binding, rows, binding_hash, reason)
        evaluate(run)


if __name__ == "__main__":
    main()
