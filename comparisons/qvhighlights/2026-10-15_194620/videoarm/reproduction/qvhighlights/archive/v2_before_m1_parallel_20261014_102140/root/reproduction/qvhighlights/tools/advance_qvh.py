"""Authorized independent phase worker; no SVHighlights units or state are touched."""

import _bootstrap  # noqa: F401
import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from videoarm.qvhighlights.common import ARTIFACTS, atomic_json
from report_qvh import costs
from refresh_progress import refresh


def execute(name, arguments):
    script = ARTIFACTS / "tools" / name
    log = ARTIFACTS / "logs" / (Path(name).stem + "_worker.log")
    with log.open("a") as f:
        f.write("\n" + json.dumps({"time": time.time(), "arguments": arguments}) + "\n")
        f.flush()
        subprocess.run(
            [sys.executable, "-u", str(script), *arguments],
            check=True,
            stdout=f,
            stderr=subprocess.STDOUT,
        )


def run(task, split, method, run_id, config="default_v2.json"):
    execute(
        "run_qvh.py",
        [
            "--task",
            task,
            "--method",
            method,
            "--split",
            split,
            "--run-id",
            run_id,
            "--config",
            str(ARTIFACTS / "configs" / config),
            "--resume",
        ],
    )


def score_runs(task, split, methods, ids, root):
    for method in methods:
        path = root / "runs" / ids[method]
        execute(
            "score_qvh.py",
            [
                "--task",
                task,
                "--manifest",
                str(ARTIFACTS / f"manifests/{split}.jsonl"),
                "--predictions",
                str(path / "predictions.jsonl"),
                "--allow-partial",
                "--out",
                str(path / "scores.json"),
            ],
        )
    if {"B1", "M1"} <= set(methods):
        paths = [root / "runs" / ids[m] for m in methods]
        arguments = [
            "--manifest",
            str(ARTIFACTS / f"manifests/{split}.jsonl"),
            "--b1-score",
            str(root / "runs" / ids["B1"] / "scores.json"),
            "--m1-score",
            str(root / "runs" / ids["M1"] / "scores.json"),
            "--out",
            str(ARTIFACTS / "checks" / f"paired_{task}_{split}_{ids['M1']}.json"),
        ]
        for p in paths:
            arguments += ["--run", str(p)]
        execute("report_qvh.py", arguments)
    execute("write_handoff.py", [])
    refresh(root)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait-resources", action="store_true")
    a = ap.parse_args()
    root = Path(os.getenv("QVH_DATA_ROOT", str(ARTIFACTS.parents[1] / "qvh_data")))
    root.mkdir(parents=True, exist_ok=True)
    worker_lock = (root / "worker.lock").open("a")
    fcntl.flock(worker_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    worker = {"status": "waiting_tp2", "phase": "P2", "started_unix": time.time()}
    atomic_json(ARTIFACTS / "checks/worker_status.json", worker)
    owns_model = False

    def pause(signum, frame):
        worker.update(status="paused_by_user_or_service_stop", updated_unix=time.time())
        raise KeyboardInterrupt()

    signal.signal(signal.SIGTERM, pause)
    signal.signal(signal.SIGINT, pause)
    try:
        while True:
            execute("launch_qvh_service.py", [])
            gpu = json.loads((ARTIFACTS / "checks/gpu_readiness.json").read_text())
            verified = ARTIFACTS / "checks/backend_verified.json"
            if verified.exists():
                try:
                    execute("verify_qvh_service.py", [])
                    owns_model = True
                    break
                except subprocess.CalledProcessError:
                    verified.unlink()
            if not gpu["blocked"]:
                execute("launch_qvh_service.py", ["--launch"])
                owns_model = True
                for i in range(120):
                    try:
                        execute("verify_qvh_service.py", [])
                        break
                    except subprocess.CalledProcessError:
                        worker.update(
                            status="model_starting", heartbeat_unix=time.time()
                        )
                        atomic_json(ARTIFACTS / "checks/worker_status.json", worker)
                        time.sleep(10)
                else:
                    raise RuntimeError(
                        "owned model startup failed after 20 minutes; logs retained"
                    )
                break
            worker.update(heartbeat_unix=time.time(), free_indices=gpu["free_indices"])
            atomic_json(ARTIFACTS / "checks/worker_status.json", worker)
            refresh(root)
            if not a.wait_resources:
                print(
                    "Blocked: two free GPUs required; CPU work and state are preserved"
                )
                return
            time.sleep(30)
        subprocess.run(
            [
                "systemd-run",
                "--user",
                "--collect",
                "--unit=videoarm-qvh-telemetry",
                sys.executable,
                "-u",
                str(ARTIFACTS / "tools/record_gpu_telemetry.py"),
            ],
            check=True,
        )
        worker.update(status="running", phase="P2_live_probes")
        atomic_json(ARTIFACTS / "checks/worker_status.json", worker)
        execute("probe_qvh_backend.py", [])
        for task in ("mr", "hd"):
            for split in ("smoke8", "dev50"):
                ids = {
                    m: f"dev_v2_{task}_{split}_{m}" for m in ("B0", "B1", "M1", "A1")
                }
                for method, run_id in ids.items():
                    worker.update(
                        phase="P2", task=task, split=split, method=method, run_id=run_id
                    )
                    atomic_json(ARTIFACTS / "checks/worker_status.json", worker)
                    run(task, split, method, run_id)
                    if (
                        split == "smoke8"
                        and costs(root / "runs" / run_id)["succeeded"] == 0
                    ):
                        raise RuntimeError(
                            f"no valid autonomous smoke predictions: {run_id}; inspect retained failures before dependent runs"
                        )
                if split == "smoke8":
                    report = json.loads(
                        (ARTIFACTS / "checks/development_amendment_v2.json").read_text()
                    )
                    report.setdefault("live_smoke_tasks_completed", []).append(task)
                    if set(report["live_smoke_tasks_completed"]) == {"mr", "hd"}:
                        report["validation_status"] = "offline_and_live_v2_verified"
                    atomic_json(
                        ARTIFACTS / "checks/development_amendment_v2.json", report
                    )
                score_runs(task, split, list(ids), ids, root)
            ids = {m: f"dev_low_v2_{task}_dev50_{m}" for m in ("B1", "M1")}
            for method, run_id in ids.items():
                run(task, "dev50", method, run_id, "dev_low_v2.json")
            score_runs(task, "dev50", list(ids), ids, root)
        estimate = {}
        for task in ("mr", "hd"):
            for method in ("B0", "B1", "M1"):
                p = root / "runs" / f"dev_v2_{task}_dev50_{method}"
                c = costs(p)
                total = c["totals"]
                seconds = total["active_seconds"] / c["planned"]
                estimate[f"{task}_{method}"] = {
                    "measured_dev_queries": c["planned"],
                    "successful_dev_queries": c["succeeded"],
                    "seconds_per_planned_query": seconds,
                    "val_hours": seconds * 1550 / 3600,
                    "test_hours": seconds * 1542 / 3600 if method != "B0" else None,
                    "allocated_gpu_hours_val_tp2": 2 * seconds * 1550 / 3600,
                    "tokens_per_query": (total["input_tokens"] + total["output_tokens"])
                    / c["planned"],
                    "dev_disk_bytes": sum(
                        f.stat().st_size for f in p.rglob("*") if f.is_file()
                    ),
                    "caveat": "development throughput estimate; partial failures may bias low; service startup excluded",
                }
        atomic_json(ARTIFACTS / "checks/full_run_estimate.json", estimate)
        while True:
            media = ARTIFACTS / "checks/media_all_progress.json"
            if media.exists():
                p = json.loads(media.read_text())
                if p["completed"] == p["planned"]:
                    break
            worker.update(phase="P3_wait_full_media_audit", heartbeat_unix=time.time())
            atomic_json(ARTIFACTS / "checks/worker_status.json", worker)
            time.sleep(30)
        execute("freeze_qvh.py", [])
        freeze = json.loads((ARTIFACTS / "checks/formal_freeze.json").read_text())
        for split in ("val", "test"):
            for task in ("mr", "hd"):
                ids = freeze["run_ids"][task][split]
                for method, run_id in ids.items():
                    worker.update(
                        phase="P3" if split == "val" else "P4",
                        task=task,
                        split=split,
                        method=method,
                        run_id=run_id,
                    )
                    atomic_json(ARTIFACTS / "checks/worker_status.json", worker)
                    run(task, split, method, run_id)
                score_runs(task, split, list(ids), ids, root)
        worker.update(status="completed", phase="P5", completed_unix=time.time())
    except KeyboardInterrupt:
        worker.update(status="paused_by_user_or_service_stop", updated_unix=time.time())
    except Exception as e:
        worker.update(
            status="blocked_or_failed",
            error=f"{type(e).__name__}: {e}",
            updated_unix=time.time(),
        )
        raise
    finally:
        atomic_json(ARTIFACTS / "checks/worker_status.json", worker)
        if owns_model:
            subprocess.run(
                ["systemctl", "--user", "stop", "videoarm-qvh-model.service"],
                check=False,
            )
        execute("write_handoff.py", [])
        refresh(root)


if __name__ == "__main__":
    main()
