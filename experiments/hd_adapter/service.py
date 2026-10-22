"""Bounded supervisor for this experiment's own model unit and worker process."""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from merit_qvh.core import write_json

from .prepare import verify

MODEL_UNIT = os.environ.get("MERIT_MODEL_UNIT", "merit-publication-model.service")


def guard(config, *, require_idle=True):
    observed = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    rows = {int(p[0]): (p[1].strip(), int(p[2])) for line in observed.splitlines() if (p := line.split(","))}
    wanted = config["hd"]["server"]["gpu_uuids"]
    if len(wanted) != config["hd"]["server"]["tensor_parallel_size"] or len(set(wanted)) != len(wanted):
        raise ValueError("Configure one unique GPU UUID per tensor-parallel rank")
    by_uuid = {uuid: (index, memory) for index, (uuid, memory) in rows.items()}
    indices = []
    for uuid in wanted:
        if uuid not in by_uuid:
            raise RuntimeError(f"Configured GPU is absent: {uuid}")
        index, memory = by_uuid[uuid]
        if require_idle and memory > 200:
            raise RuntimeError(f"Configured GPU {index} is occupied; do not reclaim it")
        indices.append(index)
    return indices


def healthy(endpoint):
    try:
        with urllib.request.urlopen(endpoint.removesuffix("/v1") + "/health", timeout=10) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError):
        return False


def stop_child(child):
    if child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=30)


def supervise(root, endpoint):
    config, _ = verify(root)
    state_path = root / "supervisor.json"
    state = (
        json.loads(state_path.read_text())
        if state_path.exists()
        else {
            "started_at": time.time(),
            "launches": 0,
            "deadline": time.time() + config["hd"]["automation"]["max_wall_seconds"],
        }
    )
    maximum = config["hd"]["automation"]["max_worker_restarts"] + 1
    child = None
    try:
        while state["launches"] < maximum and time.time() < state["deadline"]:
            if (root / "complete.json").exists():
                state["status"] = "complete"
                return
            if (root / "circuit_breaker.json").exists():
                state["status"] = "circuit_breaker"
                return
            state.update(launches=state["launches"] + 1, status="starting", at=time.time())
            write_json(state_path, state)
            subprocess.run(["systemctl", "--user", "start", MODEL_UNIT], check=True)
            startup_deadline = min(time.time() + 900, state["deadline"])
            while not healthy(endpoint):
                if time.time() > startup_deadline:
                    raise RuntimeError("Model did not become healthy within startup budget")
                time.sleep(5)
            env = dict(os.environ)
            snapshot = root / "source_snapshot"
            env.update(
                CUDA_VISIBLE_DEVICES=str(guard(config, require_idle=False)[0]),
                TOKENIZERS_PARALLELISM="false",
                PYTHONPATH=f"{snapshot / 'src'}:{snapshot / 'experiments'}",
            )
            worker_started = time.time()
            with (root / "worker.log").open("a") as log:
                child = subprocess.Popen(
                    [
                        sys.executable,
                        "-u",
                        "-m",
                        "hd_adapter.runner",
                        "run",
                        "--run-root",
                        str(root),
                        "--endpoint",
                        endpoint,
                    ],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                while child.poll() is None:
                    activity = max(
                        [worker_started]
                        + [
                            p.stat().st_mtime
                            for pattern in ("requests/*/*.json", "queries/*.json", "media_probes/*.json")
                            for p in root.glob(pattern)
                        ]
                    )
                    state.update(
                        status="running", at=time.time(), worker_pid=child.pid, last_work_activity=activity
                    )
                    write_json(state_path, state)
                    if time.time() > state["deadline"]:
                        stop_child(child)
                        state["status"] = "wall_budget_exhausted"
                        return
                    if (
                        not healthy(endpoint)
                        or time.time() - activity > config["hd"]["automation"]["stall_seconds"]
                    ):
                        stop_child(child)
                        break
                    time.sleep(15)
            state["last_worker_exit"] = child.returncode
            subprocess.run(["systemctl", "--user", "stop", MODEL_UNIT], check=True)
            if child.returncode == 0 and (root / "complete.json").exists():
                state["status"] = "complete"
                return
            time.sleep(5)
        state["status"] = "restart_or_wall_budget_exhausted"
    finally:
        if child is not None:
            stop_child(child)
        state["at"] = time.time()
        write_json(state_path, state)
        subprocess.run(["systemctl", "--user", "stop", MODEL_UNIT], check=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("guard", "supervise"))
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8037/v1")
    args = parser.parse_args()
    if args.command == "guard":
        configuration, _ = verify(args.run_root)
        guard(configuration)
    else:
        supervise(args.run_root, args.endpoint)
