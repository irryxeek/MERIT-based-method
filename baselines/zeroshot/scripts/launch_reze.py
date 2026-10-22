"""Supervise independent GPU workers, resume checkpoints, and evaluate automatically."""

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

from common import ROOT, STORE, atomic_json, read_jsonl, sha256, utc_now, write_jsonl
from reze_run_io import load_bound_run, read_record

SOURCES = [
    "scripts/common.py",
    "scripts/reze_core.py",
    "scripts/reze_media.py",
    "scripts/reze_backend.py",
    "scripts/reze_run_io.py",
    "scripts/run_reze_worker.py",
    "scripts/launch_reze.py",
    "scripts/evaluate.py",
    "configs/reze_protocol_v2.json",
    "configs/protocol_v1.json",
    "manifests/reze_model_audit.json",
    "manifests/data_audit.json",
]


def gpu_inventory():
    text = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"], text=True
    )
    return {
        int(p[0]): {"index": int(p[0]), "uuid": p[1].strip(), "used_mib": int(p[2])}
        for line in text.splitlines()
        if (p := line.split(","))
    }


def initialize(run, smoke, gpu_indices):
    manifest = ROOT / "manifests" / ("reze_smoke_inference.jsonl" if smoke else "reze_val_inference.jsonl")
    rows = read_jsonl(manifest)
    inventory = gpu_inventory()
    if (run / "binding.json").exists():
        binding, bound_rows = load_bound_run(run)
        assert rows == bound_rows and binding["smoke"] == smoke
        assert gpu_indices == [g["index"] for g in binding["gpus"]]
        assert all(inventory[g["index"]]["uuid"] == g["uuid"] for g in binding["gpus"])
        return binding, rows
    assert len(rows) == (5 if smoke else 1550)
    if not smoke:
        gate = STORE / "runs/reze_sustained_v3"
        evidence = json.loads((gate / "completion.json").read_text())
        assert evidence["status"] == "complete" and evidence["queries"] == 21
        engine_sources = json.loads((gate / "source_binding.json").read_text())["sha256"]
        for name in ["scripts/reze_backend.py", "scripts/reze_media.py", "scripts/reze_core.py"]:
            assert sha256(ROOT / name) == engine_sources[name]
    assert all(inventory[g]["used_mib"] < 300 for g in gpu_indices), "Requested GPU is occupied"
    hashes = {}
    for vid in {r["vid"] for r in rows}:
        video = Path(next(row["video_path"] for row in rows if row["vid"] == vid))
        hashes[vid] = sha256(video)
    binding = {
        "method": "REZE_Qwen3-VL-8B-Instruct_independent_appendix_implementation",
        "created_at": utc_now(),
        "smoke": smoke,
        "queries": len(rows),
        "manifest_sha256": sha256(manifest),
        "gpus": [inventory[g] for g in gpu_indices],
        "source_sha256": {name: sha256(ROOT / name) for name in SOURCES},
        "video_sha256": hashes,
        "restart_limit_per_worker": 2,
        "stall_timeout_seconds": 900,
    }
    if not smoke:
        binding["sustained_gate_sha256"] = sha256(gate / "completion.json")
    for name in SOURCES:
        target = run / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    shutil.copy2(manifest, run / "inference.jsonl")
    atomic_json(run / "binding.json", binding)
    return binding, rows


def collect(run, binding, rows, started):
    binding_hash = sha256(run / "binding.json")
    predictions = []
    counts = {
        "clips": 0,
        "frames": 0,
        "prompt_tokens": 0,
        "generate_seconds_sum": 0.0,
        "decode_preprocess_seconds_sum": 0.0,
        "score_modes": {},
        "generated_tokens": {},
    }
    receipt = {}
    for row in rows:
        record = read_record(run, row, binding, binding_hash)
        predictions.append(record["prediction"])
        counts["clips"] += len(record["clips"])
        counts["frames"] += record["media"]["model_input_frames"]
        counts["prompt_tokens"] += sum(c["prompt_tokens"] for c in record["clips"])
        counts["generate_seconds_sum"] += record["generate_seconds"]
        counts["decode_preprocess_seconds_sum"] += record["media"]["decode_preprocess_seconds"]
        for clip in record["clips"]:
            for field, key in [("score_modes", "score_mode"), ("generated_tokens", "generated_text")]:
                value = clip[key]
                counts[field][value] = counts[field].get(value, 0) + 1
        receipt[str(row["qid"])] = sha256(run / "records" / f"{row['qid']}.json")
    assert {p.stem for p in (run / "records").glob("*.json")} == set(receipt)
    write_jsonl(run / "predictions.jsonl", predictions)
    atomic_json(run / "record_receipt.json", receipt)
    commands = [sys.executable, str(ROOT / "scripts/evaluate.py"), "--run", str(run)]
    if binding["smoke"]:
        commands.append("--smoke")
    with (run / "evaluation.log").open("a") as stream:
        subprocess.run(commands, stdout=stream, stderr=subprocess.STDOUT, check=True, cwd=ROOT)
    runtime = {
        **counts,
        "output_tokens": counts["clips"],
        "wall_seconds_this_launch": time.time() - started,
        "worker_failures": len(list((run / "workers").glob("*/failure_attempt_*.json"))),
        "worker_runs": [json.loads(p.read_text()) for p in sorted((run / "workers").glob("*/completion.json"))],
        "external_api_calls": 0,
    }
    atomic_json(run / "runtime.json", runtime)
    completion = {
        "status": "complete",
        "completed_at": utc_now(),
        "completed_queries": len(predictions),
        "prediction_failures": 0,
        "predictions_sha256": sha256(run / "predictions.jsonl"),
        "binding_sha256": binding_hash,
        "record_receipt_sha256": sha256(run / "record_receipt.json"),
        "wall_seconds_this_launch": time.time() - started,
    }
    atomic_json(run / "completion.json", completion)
    return completion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    assert len(set(args.gpus)) == len(args.gpus)
    run = args.run.resolve()
    run.mkdir(parents=True, exist_ok=True)
    lock = (run / "coordinator.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.time()
    binding, rows = initialize(run, args.smoke, args.gpus)
    processes, ledger = {}, []
    ledger_path = run / "launch_ledger.json"
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text())

    def start_worker(worker):
        gpu = binding["gpus"][worker]
        assert gpu_inventory()[gpu["index"]]["used_mib"] < 300, f"GPU {gpu['index']} is now occupied"
        past = [x for x in ledger if x["worker"] == worker]
        attempt = len(past)
        if attempt > binding["restart_limit_per_worker"]:
            raise RuntimeError(f"Worker {worker} exhausted automatic restarts")
        folder = run / "workers" / str(worker)
        folder.mkdir(parents=True, exist_ok=True)
        log = (folder / f"attempt_{attempt}.log").open("a")
        env = dict(os.environ)
        env.update(
            CUDA_VISIBLE_DEVICES=str(gpu["index"]),
            CUDA_DEVICE_ORDER="PCI_BUS_ID",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            HF_HUB_DISABLE_TELEMETRY="1",
            VLLM_NO_USAGE_STATS="1",
            VLLM_WORKER_MULTIPROC_METHOD="spawn",
            OMP_NUM_THREADS="2",
            TOKENIZERS_PARALLELISM="false",
            PYTHONUNBUFFERED="1",
        )
        process = subprocess.Popen(
            [
                sys.executable,
                str(ROOT / "scripts/run_reze_worker.py"),
                "--run",
                str(run),
                "--worker",
                str(worker),
                "--attempt",
                str(attempt),
            ],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        log.close()
        item = {"worker": worker, "attempt": attempt, "pid": process.pid, "started_at": utc_now()}
        ledger.append(item)
        atomic_json(ledger_path, ledger)
        processes[worker] = (process, item, time.time())

    try:
        for worker in range(len(args.gpus)):
            if not (run / "workers" / str(worker) / "completion.json").exists():
                start_worker(worker)
        while processes:
            for worker, (process, item, launch_time) in list(processes.items()):
                progress_path = run / "workers" / str(worker) / "progress.json"
                last_progress = max(launch_time, progress_path.stat().st_mtime if progress_path.exists() else 0)
                if process.poll() is None and time.time() - last_progress > binding["stall_timeout_seconds"]:
                    import signal

                    os.killpg(process.pid, signal.SIGKILL)
                    item["stalled"] = True
                    process.wait()
                if process.poll() is not None:
                    item.update(returncode=process.returncode, finished_at=utc_now())
                    atomic_json(ledger_path, ledger)
                    del processes[worker]
                    if process.returncode:
                        time.sleep(10)
                        start_worker(worker)
            statuses = [json.loads(p.read_text()) for p in sorted((run / "workers").glob("*/progress.json"))]
            atomic_json(
                run / "progress.json",
                {
                    "status": "running",
                    "updated_at": utc_now(),
                    "elapsed_seconds": time.time() - started,
                    "expected_queries": len(rows),
                    "completed_queries": sum(s["completed_queries"] for s in statuses),
                    "clips": sum(s["clips"] for s in statuses),
                    "workers": statuses,
                },
            )
            if processes:
                time.sleep(10)
        completion = collect(run, binding, rows, started)
        atomic_json(run / "progress.json", completion)
        print(json.dumps(completion), flush=True)
    except BaseException as exc:
        atomic_json(
            run / "coordinator_failure.json",
            {
                "time": utc_now(),
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "live_worker_pids": [p.pid for p, _, _ in processes.values() if p.poll() is None],
            },
        )
        raise


if __name__ == "__main__":
    main()
