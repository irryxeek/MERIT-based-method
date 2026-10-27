"""Freeze, run and evaluate the authorized TVSum val10 four-arm transfer pilot."""

import argparse
import fcntl
import importlib.metadata
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from core import (
    ARMS,
    CALLS_PER_VIDEO,
    LIMITS,
    MAX_SHARD_TOKENS,
    ROOT,
    TransferBackend,
    TransferLedger,
    make_scores,
    one_video,
    rerank,
    verify_scores,
)
from dispatch import environment, idle_gpu
from io_utils import digest, read_json, sha256, utc_now, write_json
from observer_output_probe import process_group_members, signal_group

GPU_SECONDS_PER_WORKER = 3600


def model_check(receipt):
    audit = receipt["audit"]
    if audit["revision"] != "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b":
        raise ValueError("Unexpected model revision")
    model = Path(os.environ["REZE_MODEL_PATH"]).resolve()
    if model != Path(audit["model_path"]).resolve():
        raise ValueError("Configured model differs from frozen weight receipt")
    for name, stat in receipt["stats"].items():
        current = (model / name).stat()
        if {"size": current.st_size, "mtime_ns": current.st_mtime_ns} != stat:
            raise ValueError(f"Model stat changed: {name}")
        if current.st_size < 100_000_000 and sha256(model / name) != audit["files"][name]["sha256"]:
            raise ValueError(f"Model/tokenizer content changed: {name}")


def freeze_source(run):
    files = []
    for folder in ("scripts", "experiments/revalue", "experiments/reze_evidence", "experiments/reze_tvsum",
                   "experiments/revalue_crossdataset"):
        files.extend(p for p in (ROOT / folder).rglob("*") if p.is_file()
                     and p.suffix in {".py", ".json", ".md", ".LICENSE"}
                     and not any(part.startswith(".") or part == "__pycache__" for part in p.relative_to(ROOT).parts))
    hashes = {}
    for source in files:
        relative = source.relative_to(ROOT)
        destination = run / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        hashes[str(relative)] = sha256(destination)
    return hashes


def prepare(args):
    from tvsum_adapter import prepare_inputs

    run = args.run.resolve()
    if run.exists():
        raise FileExistsError("Use a new run; never overwrite historical artifacts")
    receipt = read_json(args.model_receipt)
    model_check(receipt)
    run.mkdir(parents=True)
    shutil.copy2(args.protocol, run / "protocol.md")
    write_json(run / "model_verification.json", receipt)
    rows = prepare_inputs(args.baseline.resolve(), run / "inputs")
    if len(rows) != 10 or len({r["qid"] for r in rows}) != 10:
        raise ValueError("Exact val10 required")
    rows.sort(key=lambda r: digest("tvsum-transfer-v1:" + r["qid"]))
    write_json(run / "inference.json", rows)
    inputs = {str(p.relative_to(run)): sha256(p) for p in (run / "inputs").rglob("*") if p.is_file()}
    for name in ("protocol.md", "model_verification.json", "inference.json"):
        inputs[name] = sha256(run / name)
    manifest = {"created_at": utc_now(), "protocol": "revalue-tvsum-val10-transfer-v1", "baseline": str(args.baseline.resolve()),
                "source_sha256": freeze_source(run), "input_sha256": inputs, "shards": [[r["qid"] for r in rows[i::2]] for i in range(2)],
                "model_revision": receipt["audit"]["revision"], "model_validation": "prior full-hash receipt plus unchanged file stats and fresh small-file hashes",
                "limits": LIMITS, "formal_requests": 10 * CALLS_PER_VIDEO, "total_token_cap": MAX_SHARD_TOKENS * 2,
                "maximum_gpu_seconds": GPU_SECONDS_PER_WORKER * 2, "worker_launches_per_shard": 1,
                "inference_uses_labels": False, "new_source_frames": 0, "observation_enabled": False,
                "model_runtime": "shared frozen ReVALUE engine; original P_pair prompt/logprobs/ranking; no batch-throughput comparison",
                "passport": {"Origin Skill": "academic-research-suite 3.23.2 / experiment-agent", "Origin Mode": "run",
                             "Origin Date": "2026-10-27", "Verification Status": "UNVERIFIED", "Version Label": "tvsum_transfer_v1"},
                "packages": {n: importlib.metadata.version(n) for n in ("torch", "transformers", "vllm", "numpy", "pydantic")}}
    write_json(run / "manifest.json", manifest)
    write_json(run / "status.json", {"status": "prepared", "formal_requests": 670, "videos": 10, "at": utc_now()})
    print(f"Prepared {run}: 10 videos; 600 comparisons + 70 task/fact/memory requests", flush=True)


def verify(run):
    manifest = read_json(run / "manifest.json")
    for name, expected in manifest["source_sha256"].items():
        if sha256(run / "source_snapshot" / name) != expected:
            raise ValueError(f"Frozen source drift: {name}")
    for name, expected in manifest["input_sha256"].items():
        if sha256(run / name) != expected:
            raise ValueError(f"Frozen input drift: {name}")
    model_check(read_json(run / "model_verification.json"))
    rows = read_json(run / "inference.json")
    receipts = read_json(run / "inputs/adapter_manifest.json")["parent_receipts"]
    for row in rows:
        receipts.extend(row["parent_receipts"])
    for entry in receipts:
        if sha256(entry["path"]) != entry["sha256"]:
            raise ValueError(f"Parent cache drift: {entry['path']}")
    return manifest, rows


def worker(args):
    import torch
    from tvsum_adapter import load_candidates

    manifest, rows = verify(args.run)
    actual = "GPU-" + str(torch.cuda.get_device_properties(0).uuid).removeprefix("GPU-")
    if torch.cuda.device_count() != 1 or actual != os.environ["REZE_EXPECTED_GPU_UUID"]:
        raise ValueError("Visible GPU mapping changed")
    directory = args.run / f"shard_{args.slot}"
    directory.mkdir(exist_ok=True)
    ledger = TransferLedger(directory / "ledger.json", LIMITS)
    backend = TransferBackend(directory, ledger)
    expected = manifest["shards"][args.slot]
    for row in rows:
        if row["qid"] not in expected:
            continue
        if time.time() > args.deadline - 60:
            raise RuntimeError("Lifecycle budget exhausted; preserve partial results")
        path = directory / "records" / f"{row['qid']}.json"
        if path.exists():
            raise ValueError("One launch only; no unreviewed resume")
        targets, traces = load_candidates(args.run / "inputs", row)
        record = one_video(row, targets, traces, backend)
        record["manifest_sha256"] = sha256(args.run / "manifest.json")
        write_json(path, record)
        print(f"Completed {row['qid']}: " + str({arm: record['arms'][arm]['status'] for arm in ARMS}), flush=True)
    write_json(directory / "inference_complete.json", {"status": "complete", "qids": expected,
        "records_sha256": {qid: sha256(directory / "records" / f"{qid}.json") for qid in expected},
        "manifest_sha256": sha256(args.run / "manifest.json")})


def cleanup(process):
    remaining = process_group_members(process.pid)
    if remaining:
        signal_group(process.pid, signal.SIGTERM)
        end = time.monotonic() + 10
        while process_group_members(process.pid) and time.monotonic() < end:
            time.sleep(0.2)
        if process_group_members(process.pid):
            signal_group(process.pid, signal.SIGKILL)
    process.wait(timeout=5)
    return process_group_members(process.pid)


def supervise(args):
    run = args.run.resolve()
    def interrupted(signum, _frame):
        raise InterruptedError(f"Supervisor signal {signum}; clean up owned workers")
    for signum in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(signum, interrupted)
    with (run / "supervisor.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify(run)
        if (run / "launch.json").exists():
            raise ValueError("Preserve previous launch and budget; explicit recovery protocol required")
        candidates = "all"
        first, inventory = idle_gpu(candidates)
        if first is None:
            raise RuntimeError("No idle GPU; no model launched")
        remaining = [uuid for uuid in inventory if uuid != first["uuid"]]
        second, _ = idle_gpu(remaining)
        if second is None:
            raise RuntimeError("Two independent shards require two idle GPUs; no model launched")
        gpus = [first, second]
        if any(idle_gpu([gpu["uuid"]])[0] is None for gpu in gpus):
            raise RuntimeError("GPU occupancy changed before launch")
        script = run / "source_snapshot/experiments/revalue_crossdataset/run.py"
        workers, logs = [], []
        try:
            for slot, gpu in enumerate(gpus):
                directory = run / f"shard_{slot}"
                directory.mkdir(exist_ok=True)
                log = (directory / "worker.log").open("w")
                logs.append(log)
                command = [sys.executable, str(script), "worker", "--run", str(run), "--slot", str(slot),
                           "--deadline", str(time.time() + GPU_SECONDS_PER_WORKER)]
                started = time.monotonic()
                process = subprocess.Popen(command, env=environment(gpu), stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                workers.append({"process": process, "started": started, "gpu": gpu, "slot": slot,
                                "command": command, "status": "running", "last_progress": None, "unchanged": 0})
                write_json(run / "launch.json", {"at": utc_now(), "workers": [
                    {"slot": w["slot"], "gpu": w["gpu"], "pid": w["process"].pid, "command": w["command"]} for w in workers]})
            while any(w["status"] == "running" for w in workers):
                for item in workers:
                    if item["status"] != "running":
                        continue
                    process = item["process"]
                    elapsed = time.monotonic() - item["started"]
                    directory = run / f"shard_{item['slot']}"
                    done = process.poll() is not None
                    timeout = elapsed >= GPU_SECONDS_PER_WORKER - 40
                    ledger_path = directory / "ledger.json"
                    progress = (ledger_path.stat().st_mtime_ns if ledger_path.exists() else None,
                                (directory / "worker.log").stat().st_size)
                    item["unchanged"] = item["unchanged"] + 1 if progress == item["last_progress"] else 0
                    item["last_progress"] = progress
                    if item["unchanged"] == 3:
                        print(f"Shard {item['slot']}: 90-second progress stall advisory", flush=True)
                    if done or timeout:
                        if timeout and not done:
                            print(f"Shard {item['slot']}: hard lifecycle limit; terminating owned group", flush=True)
                        left = cleanup(process)
                        resource = {"gpu": item["gpu"], "exit_code": process.returncode, "timed_out": timeout and not done,
                                    "seconds": time.monotonic() - item["started"], "remaining_owned_pids": left,
                                    "finished_at": utc_now()}
                        write_json(directory / "resource_receipt.json", resource)
                        item["status"] = "complete" if process.returncode == 0 and not left else "failed"
                write_json(run / "status.json", {"status": "running" if any(w["status"] == "running" for w in workers) else "inference_finished",
                    "at": utc_now(), "completed_videos": sum(len(list((run / f"shard_{i}/records").glob("*.json"))) for i in range(2)),
                    "workers": [{"slot": w["slot"], "gpu": w["gpu"], "state": w["status"],
                                 "elapsed_seconds": time.monotonic() - w["started"], "stall_advisory": w["unchanged"] >= 3} for w in workers]})
                if any(w["status"] == "running" for w in workers):
                    remaining_seconds = min(GPU_SECONDS_PER_WORKER - 40 - (time.monotonic() - w["started"])
                                            for w in workers if w["status"] == "running")
                    time.sleep(min(30, max(0.1, remaining_seconds)))
            if all(w["status"] == "complete" for w in workers):
                analyze(args)
            else:
                write_json(run / "stopped.json", {"at": utc_now(), "reason": "Worker failed; attempts and partial records retained"})
                raise RuntimeError("Transfer run incomplete")
        finally:
            for item in workers:
                if item["process"].poll() is None or process_group_members(item["process"].pid):
                    cleanup(item["process"])
                resource_path = run / f"shard_{item['slot']}/resource_receipt.json"
                if not resource_path.exists():
                    write_json(resource_path, {"gpu": item["gpu"], "exit_code": item["process"].returncode,
                        "interrupted": True, "seconds": time.monotonic() - item["started"],
                        "remaining_owned_pids": process_group_members(item["process"].pid), "finished_at": utc_now()})
            for log in logs:
                log.close()


def analyze(args):
    from tvsum_adapter import evaluate_predictions

    run = args.run.resolve()
    manifest, rows = verify(run)
    records = {}
    ledgers, resources = [], []
    for slot in range(2):
        directory = run / f"shard_{slot}"
        completion = read_json(directory / "inference_complete.json")
        if completion["manifest_sha256"] != sha256(run / "manifest.json") or completion["qids"] != manifest["shards"][slot]:
            raise ValueError("Shard completion mismatch")
        for qid in completion["qids"]:
            path = directory / "records" / f"{qid}.json"
            if sha256(path) != completion["records_sha256"][qid]:
                raise ValueError("Completed prediction record changed")
            record = read_json(path)
            if record["manifest_sha256"] != sha256(run / "manifest.json") or record["qid"] != qid:
                raise ValueError("Record mismatch")
            records[qid] = record
        ledger = read_json(directory / "ledger.json")
        for attempt in ledger["attempts"]:
            if attempt["status"] not in {"complete", "invalid"}:
                raise ValueError("Unsettled request")
            raw = read_json(directory / "attempts" / f"{attempt['number']:05d}.json")
            if digest(raw) != attempt["record_sha256"]:
                raise ValueError("Raw request receipt changed")
        ledgers.extend(ledger["attempts"])
        resources.append(read_json(directory / "resource_receipt.json"))
    if set(records) != {r["qid"] for r in rows}:
        raise ValueError("Full denominator required")
    for row in rows:
        record = records[row["qid"]]
        if record["B0"] != row["scores"] or record["candidate_ids"] != row["candidate_ids"]:
            raise ValueError("Prediction baseline/candidate drift")
        for arm in ARMS:
            output = record["arms"][arm]
            verify_scores(row["scores"], row["candidate_ids"], output["scores"])
            if output["status"] == "complete":
                if arm == "P_pair":
                    replay, _ = make_scores(row["scores"], row["candidate_ids"], output["comparisons"], arm)
                else:
                    replay, _ = rerank(row["scores"], row["candidate_ids"], output["comparisons"])
            else:
                replay = row["scores"] if arm == "P_pair" else record["arms"]["P_pair"]["scores"]
            if replay != output["scores"]:
                raise ValueError("Ranking replay differs from predictions")
    predictions = {arm: {r["vid"]: records[r["qid"]]["B0"] if arm == "B0"
                        else records[r["qid"]]["arms"][arm]["scores"] for r in rows} for arm in ["B0", *ARMS]}
    write_json(run / "predictions.json", predictions)
    write_json(run / "prediction_receipt.json", {"sha256": sha256(run / "predictions.json"), "at": utc_now(),
                                                "labels_read_after_this_receipt": True})
    metrics = {arm: evaluate_predictions(values, Path(manifest["baseline"])) for arm, values in predictions.items()}
    cost = {"new_attempts": len(ledgers), "new_actual_or_reserved_tokens": sum(a.get("charged_tokens", a["reserved_tokens"]) for a in ledgers),
            "gpu_lifecycle_seconds": sum(r["seconds"] for r in resources), "new_model_frame_presentations": sum(a["model_frames"] for a in ledgers),
            "reused_initialization_per_video": {r["vid"]: r["baseline_cost"] for r in rows},
            "incremental_by_arm": {arm: {"tokens": sum(a.get("charged_tokens", a["reserved_tokens"]) for a in ledgers if arm in a["arms"]),
                                          "requests_including_shared": sum(arm in a["arms"] for a in ledgers)} for arm in ARMS},
            "fallback_P_pair_tokens": {arm: sum(a.get("charged_tokens", a["reserved_tokens"]) for a in ledgers
                if "P_pair" in a["arms"] and records[a["key"].split('/')[0]]["arms"][arm]["status"] == "fallback") for arm in ("C_direct4", "E_memory")}}
    if cost["new_attempts"] > manifest["formal_requests"] or cost["new_actual_or_reserved_tokens"] > manifest["total_token_cap"]:
        raise ValueError("Run budget exceeded")
    result = {"status": "complete", "at": utc_now(), "scope": "TVSum UMT fixed val10 development transfer",
              "metrics": metrics, "cost": cost, "fallbacks": {arm: [qid for qid, r in records.items() if r["arms"][arm]["status"] != "complete"] for arm in ARMS},
              "changed_videos_vs_B0": {arm: [vid for vid in predictions[arm] if predictions[arm][vid] != predictions['B0'][vid]] for arm in ARMS},
              "predictions_sha256": sha256(run / "predictions.json"), "resources": resources,
              "source_content_reliability_independently_validated": False}
    write_json(run / "evaluation.json", result)
    write_json(run / "complete.json", {"status": "complete", "evaluation_sha256": sha256(run / "evaluation.json"), "at": utc_now()})
    write_json(run / "status.json", {"status": "complete", "completed_videos": 10, "at": utc_now(), "evaluation": str(run / "evaluation.json")})
    print("All 10 videos and four conditions evaluated; " + str(cost["new_attempts"]) + " new requests", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run", "worker", "analyze"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--model-receipt", type=Path)
    parser.add_argument("--slot", type=int, choices=(0, 1))
    parser.add_argument("--deadline", type=float)
    args = parser.parse_args()
    {"prepare": prepare, "run": supervise, "worker": worker, "analyze": analyze}[args.mode](args)


if __name__ == "__main__":
    main()
