"""Paired object-presence diagnostic on frozen frames, with no memory promotion."""

import argparse
import fcntl
import hashlib
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from backend import MODEL, Backend, SemanticFailure
from budget import BudgetStop, Ledger
from dispatch import environment, idle_gpu
from frame_observer import still_payload
from io_utils import ROOT, compact, digest, legacy, parents, read_json, sha256, utc_now, write_json
from object_presence_contract import (
    ARMS,
    OUTPUT_CAP,
    ObjectPresence,
    case_jobs,
    prompt,
    validate_case_sources,
    validate_presence,
)
from observer_output_probe import process_group_members, signal_group
from r2_observer import read_frames, source_timeline

COUNT = 60
MAX_REQUEST = 2048
MAX_TOTAL = COUNT * MAX_REQUEST
MAX_GPU_SECONDS = 900
LIMITS = {"requests": COUNT, "extra_attempts": 0, "arm_incremental_tokens": {arm: MAX_TOTAL // 2 for arm in ARMS}}


class PresenceLedger(Ledger):
    def __init__(self, path, *, deadline):
        super().__init__(path, LIMITS)
        self.deadline = deadline

    def reserve(self, key, prompt_tokens, output_cap, arms, request_hash, frames):
        if time.time() >= self.deadline - 60:
            raise BudgetStop("60-second shutdown reserve")
        if prompt_tokens + output_cap > MAX_REQUEST:
            raise BudgetStop("2048-token request reservation exceeded")
        if self.cost() + prompt_tokens + output_cap > MAX_TOTAL:
            raise BudgetStop("122880-token total reservation exceeded")
        if len(arms) != 1 or arms[0] not in ARMS:
            raise ValueError("Each request must charge one frozen arm")
        return super().reserve(key, prompt_tokens, output_cap, arms, request_hash, frames)


def verify(run):
    manifest = read_json(run / "manifest.json")
    for path, expected in manifest["parent_receipts"].items():
        if sha256(path) != expected:
            raise ValueError(f"Parent artifact changed: {path}")
    for name, expected in manifest["source_sha256"].items():
        if sha256(ROOT / name) != expected or sha256(run / "source_snapshot" / name) != expected:
            raise ValueError(f"Source changed: {name}")
    for name, expected in manifest["input_sha256"].items():
        if sha256(run / name) != expected:
            raise ValueError(f"Frozen input changed: {name}")
    for name, expected in read_json(Path(manifest["parent_run"]) / "model_verification.json")["stats"].items():
        stat = (Path(MODEL) / name).stat()
        if (stat.st_size, stat.st_mtime_ns) != (expected["size"], expected["mtime_ns"]):
            raise ValueError("Model drift")
    return manifest


def prepare(args):
    run, reference = args.run.resolve(), args.reference.resolve()
    if run.exists():
        raise FileExistsError("Preserve old runs and choose a new version")
    original = read_json(reference / "manifest.json")
    completion = read_json(reference / "complete.json")
    for name, expected in completion["receipts"].items():
        if sha256(reference / name) != expected:
            raise ValueError("Reference completion receipt changed")
    receipts = dict(original["parent_receipts"])
    for path, expected in receipts.items():
        if sha256(path) != expected:
            raise ValueError(f"Inherited receipt changed: {path}")
    fixture = read_json(args.cases)
    cases = fixture["cases"]
    if len(cases) != 30 or len({c["image_key"] for c in cases}) != 15:
        raise ValueError("Protocol requires 30 cases on 15 images")
    if set(Counter(c["image_key"] for c in cases).values()) != {2}:
        raise ValueError("Each image must have two cases")
    for case in cases:
        if case["reference_status"] not in {"supported", "contradicted", "cannot_determine"}:
            raise ValueError("Reference labels must be explicit before inference")
    old_jobs = {f"{j['qid']}_{j['arm']}_{j['number']}_{j['position']:02d}": j for j in read_json(reference / "jobs.json")}
    validate_case_sources(cases, old_jobs)
    if len({c["qid"] for c in cases}) != 7:
        raise ValueError("Protocol requires seven source queries")
    ledger = {a["key"]: a for a in read_json(reference / "probe/ledger.json")["attempts"]}
    selected = {}
    for case in cases:
        name = case["image_key"]
        job = old_jobs[name]
        if name in selected:
            continue
        record_path = reference / "records" / f"{name}.json"
        record = read_json(record_path)
        response = read_json(record["response_path"])
        if case["qid"] != job["qid"] or digest(response) != ledger[record["key"]]["record_sha256"]:
            raise ValueError("Selected reference is not authenticated")
        if response["request_sha256"] != digest(response["specification"]) or record["result"] != response["parsed"]:
            raise ValueError("Selected reference response mismatch")
        if record["proof"] != job["record"]["proof"]:
            raise ValueError("Selected frame proof mismatch")
        from PIL import Image

        with Image.open(reference / "images" / f"{name}.png") as image:
            if hashlib.sha256(image.convert("RGB").tobytes()).hexdigest() != record["proof"]["frames"][0]["rgb_sha256"]:
                raise ValueError("Prelabelled image pixels differ from frozen source")
        selected[name] = {"job": job, "proof": record["proof"]}
        for path in (record_path, Path(record["response_path"]), reference / "images" / f"{name}.png"):
            receipts[str(path)] = sha256(path)
    if len({(v["job"]["qid"], v["proof"]["frames"][0]["frame_index"]) for v in selected.values()}) != 15:
        raise ValueError("Protocol requires 15 distinct source-frame identities")
    for name in ("manifest.json", "complete.json", "jobs.json", "review_pairs.json", "probe/ledger.json"):
        receipts[str(reference / name)] = sha256(reference / name)
    run.mkdir(parents=True)
    shutil.copy2(args.protocol, run / "protocol.md")
    shutil.copy2(args.cases, run / "reference_labels.json")
    write_json(run / "jobs.json", case_jobs(cases))
    write_json(run / "frames.json", selected)
    write_json(run / "output_schema.json", ObjectPresence.model_json_schema())
    source_paths = {ROOT / name for name in original["source_sha256"]}
    source_paths.update(Path(__file__).parent.rglob("*.py"))
    source_paths.add(args.cases.resolve())
    sources = {}
    for path in sorted(source_paths):
        name = str(path.relative_to(ROOT))
        target = run / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        sources[name] = sha256(path)
    manifest = {
        "created_at": utc_now(), "protocol": "object-presence-paired-v1", "reference": str(reference),
        "parent_run": original["parent_run"], "model_revision": original["model_revision"],
        "parent_receipts": receipts, "source_sha256": sources,
        "input_sha256": {name: sha256(run / name) for name in ("protocol.md", "jobs.json", "frames.json", "reference_labels.json", "output_schema.json")},
        "limits": LIMITS, "maximum_reserved_tokens_per_request": MAX_REQUEST, "maximum_total_tokens": MAX_TOTAL,
        "maximum_gpu_seconds": MAX_GPU_SECONDS, "maximum_worker_launches": 1, "output_cap": OUTPUT_CAP,
        "new_hd_predictions": False, "automatic_memory_promotion": False,
        "passport": {"Origin Skill": "experiment-agent", "Origin Mode": "run", "Origin Date": utc_now(), "Verification Status": "UNVERIFIED", "Version Label": "object_presence_v1"},
    }
    write_json(run / "manifest.json", manifest)
    verify(run)
    write_json(run / "status.json", {"status": "prepared", "expected": COUNT, "at": utc_now()})
    print(compact({"status": "prepared", "requests": COUNT, "manifest_sha256": sha256(run / "manifest.json")}))


def worker(args):
    run = args.run.resolve()
    manifest = verify(run)
    jobs, frozen_frames = read_json(run / "jobs.json"), read_json(run / "frames.json")
    import torch

    torch.set_num_threads(2)
    actual = "GPU-" + str(torch.cuda.get_device_properties(0).uuid).removeprefix("GPU-")
    if torch.cuda.device_count() != 1 or actual != os.environ["REZE_EXPECTED_GPU_UUID"]:
        raise ValueError("GPU UUID mismatch")
    directory = run / "probe"
    directory.mkdir(exist_ok=True)
    ledger = PresenceLedger(directory / "ledger.json", deadline=args.deadline)
    backend = Backend(directory, ledger)
    parent = Path(manifest["parent_run"])
    binding = read_json(parent / "binding.json")
    rows = {r["qid"]: r for r in legacy.read_jsonl(parent / "inference.jsonl")}
    current_qid, current_image, reader = None, None, None
    for index, case in enumerate(jobs):
        source = frozen_frames[case["image_key"]]
        job, qid = source["job"], case["qid"]
        if qid != job["qid"]:
            raise ValueError("Frozen claim query differs from source query")
        if qid != current_qid:
            if reader is not None:
                del reader
            row = rows[qid]
            baseline, _ = parents(binding, row)
            if sha256(row["video_path"]) != baseline["media"]["source_sha256"]:
                raise ValueError("Source video changed")
            original = read_json(parent / "R2/records" / f"{qid}.json")
            reader, timestamps, _ = source_timeline(row, baseline)
            current_qid = qid
        if case["image_key"] != current_image:
            old = original["observations"][job["arm"]][job["number"]]
            video, trace = read_frames(reader, timestamps, [f["frame_index"] for f in old["trace"]["frames"]], row, old["action"], baseline["media"]["resized_height_width"], job["arm"])
            if trace != old["trace"]:
                raise ValueError("Parent frame timeline changed")
            payload, proof = still_payload(video, trace, job["position"])
            if proof != source["proof"]:
                raise ValueError("Input pixels differ from frozen reference")
            current_image = case["image_key"]
        if case["text"] != prompt(case, case["arm"]):
            raise ValueError("Frozen prompt mismatch")
        record = {"key": case["key"], "case_id": case["case_id"], "arm": case["arm"], "proof": proof}
        try:
            value, raw = backend.call(case["key"], case["text"], [payload], [proof], ObjectPresence, OUTPUT_CAP, [case["arm"]], validate_presence, output_schema=read_json(run / "output_schema.json"))
            record.update(status="complete", result=value, response_path=raw["response_path"])
        except SemanticFailure as error:
            record.update(status="invalid", error=str(error), response_path=str(directory / "attempts" / f"{ledger.attempts_for(case['key'])[-1]['number']:05d}.json"))
        write_json(run / "records" / f"{case['case_id']}_{case['arm']}.json", record)
        write_json(run / "status.json", {"status": "running", "completed": index + 1, "expected": COUNT, "at": utc_now()})
        print(f"Completed {index + 1}/{COUNT}: {case['key']} ({record['status']})", flush=True)
    write_json(run / "worker_complete.json", {"at": utc_now(), "requests": len(jobs), "ledger_sha256": sha256(ledger.path)})


def collect(run, lifecycle):
    jobs = {j["key"]: j for j in read_json(run / "jobs.json")}
    frames = read_json(run / "frames.json")
    records = [read_json(p) for p in sorted((run / "records").glob("*.json"))]
    ledger_path = run / "probe/ledger.json"
    ledger = Ledger(ledger_path, LIMITS) if ledger_path.exists() else None
    attempts = {a["key"]: a for a in ledger.state["attempts"]} if ledger else {}
    for record in records:
        job = jobs[record["key"]]
        raw = read_json(record["response_path"])
        spec = raw["specification"]
        if digest(raw) != attempts[record["key"]]["record_sha256"] or raw["request_sha256"] != digest(spec):
            raise ValueError("Response commitment mismatch")
        if spec["text"] != job["text"] or spec["schema"] != ObjectPresence.model_json_schema() or spec["output_cap"] != OUTPUT_CAP or spec["arms"] != [job["arm"]]:
            raise ValueError("Request differs from frozen prompt/schema/budget")
        if record["proof"] != frames[job["image_key"]]["proof"] or spec["evidence"] != [record["proof"]]:
            raise ValueError("Response source mismatch")
        if record["status"] != raw["status"] or (record["status"] == "complete" and record["result"] != raw["parsed"]):
            raise ValueError("Record and response disagree")
    keys = [r["key"] for r in records]
    worker_receipt = read_json(run / "worker_complete.json") if (run / "worker_complete.json").exists() else {}
    checks = {
        "worker_exit_zero": lifecycle["exit_code"] == 0,
        "supervisor_clean": lifecycle["supervisor_error"] is None and not lifecycle["timed_out"],
        "exact_frozen_keys": len(keys) == len(set(keys)) == COUNT and set(keys) == set(jobs),
        "exact_ledger_keys": ledger is not None and len(ledger.state["attempts"]) == COUNT and set(attempts) == set(jobs),
        "worker_ledger_receipt": ledger is not None and worker_receipt.get("ledger_sha256") == sha256(ledger.path),
        "gpu_within_budget": lifecycle["seconds"] <= MAX_GPU_SECONDS,
        "owned_processes_released": not lifecycle["cleanup"]["remaining_owned_pids_after_cleanup"],
        "token_budget": ledger is None or ledger.cost() <= MAX_TOTAL,
    }
    summary = {"at": utc_now(), "status": "complete" if all(checks.values()) else "partial", "completion_checks": checks,
               "processed": len(records), "valid": sum(r["status"] == "complete" for r in records),
               "actual_tokens": ledger.cost() if ledger else 0, "lifecycle": lifecycle,
               "new_hd_predictions": False, "memory_promoted": False}
    write_json(run / "summary.json", summary)
    write_json(run / "status.json", {"status": summary["status"], "completed": len(records), "expected": COUNT, "at": utc_now()})
    receipts = {name: sha256(run / name) for name in ("manifest.json", "probe/ledger.json", "resource_receipt.json", "summary.json") if (run / name).exists()}
    write_json(run / ("complete.json" if summary["status"] == "complete" else "stopped.json"), {"status": summary["status"], "receipts": receipts, "at": utc_now()})
    print(compact(summary))


def analyze(run):
    verify(run)
    completion = read_json(run / "complete.json")
    for name, expected in completion["receipts"].items():
        if sha256(run / name) != expected:
            raise ValueError("Completion receipt changed")
    cases = read_json(run / "reference_labels.json")["cases"]
    attempts = {a["key"]: a for a in read_json(run / "probe/ledger.json")["attempts"]}
    rows, statistics = [], {}
    for case in cases:
        row = {**case, "responses": {}}
        for arm in ARMS:
            record = read_json(run / "records" / f"{case['case_id']}_{arm}.json")
            raw = read_json(record["response_path"])
            if digest(raw) != attempts[f"{case['case_id']}/{arm}"]["record_sha256"] or record["status"] != raw["status"]:
                raise ValueError("Analysis response drift")
            if record["status"] == "complete" and record["result"] != raw["parsed"]:
                raise ValueError("Analysis parsed result drift")
            row["responses"][arm] = {"technical_status": record["status"], "result": record.get("result"),
                                      "input_tokens": raw.get("prompt_tokens", 0), "output_tokens": raw.get("output_tokens", 0),
                                      "finish_reason": raw.get("finish_reason"), "inference_seconds": raw["inference_seconds"]}
        rows.append(row)
    for arm in ARMS:
        labels = [(r["reference_status"], r["responses"][arm]["result"]["status"] if r["responses"][arm]["technical_status"] == "complete" else "invalid") for r in rows]
        statistics[arm] = {
            "reference_agreement": sum(a == b for a, b in labels), "total": len(labels),
            "positive_supported": sum(a == b == "supported" for a, b in labels), "reference_positive": sum(a == "supported" for a, _ in labels),
            "unsupported_affirmations": sum(a != "supported" and b == "supported" for a, b in labels),
            "cannot_determine": sum(b == "cannot_determine" for _, b in labels),
            "invalid": sum(b == "invalid" for _, b in labels),
            "confusion": dict(Counter(f"{a}->{b}" for a, b in labels)),
            "input_tokens": sum(r["responses"][arm]["input_tokens"] for r in rows),
            "output_tokens": sum(r["responses"][arm]["output_tokens"] for r in rows),
        }
    direct, evidence = statistics["direct"], statistics["evidence_first"]
    known = [r for r in rows if r["stratum"] == "known_failure" and r["reference_status"] != "supported"]
    gates = {
        "all_60_valid": direct["invalid"] + evidence["invalid"] == 0,
        "known_unsupported_not_affirmed": len(known) == 3 and all(r["responses"]["evidence_first"]["result"] is not None and r["responses"]["evidence_first"]["result"]["status"] != "supported" for r in known),
        "positive_retention_at_least_90_percent": evidence["positive_supported"] >= math.ceil(0.9 * evidence["reference_positive"]),
        "false_affirmations_no_worse": evidence["unsupported_affirmations"] <= direct["unsupported_affirmations"],
        "agreement_no_worse": evidence["reference_agreement"] >= direct["reference_agreement"],
    }
    write_json(run / "paired_analysis.json", {"at": utc_now(), "statistics": statistics, "gates": gates, "rows": rows,
        "reference_type": "pre_inference_assistant_visual_labels_not_independent_human_ground_truth",
        "decision": "preliminary_checks_passed_content_review_pending" if all(gates.values()) else "targeted_verifier_not_promoted",
        "explanation_content_review": "pending_separate_receipt",
        "memory_promoted": False, "new_hd_metrics": False})
    print(compact({"statistics": statistics, "gates": gates}))


def supervise(args):
    run = args.run.resolve()
    with (run / "runner.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify(run)
        if (run / "launch.json").exists():
            raise ValueError("Exactly one launch; retain any failure")
        gpu, _ = idle_gpu("all")
        if gpu is None or idle_gpu([gpu["uuid"]])[0] is None:
            write_json(run / "status.json", {"status": "waiting_for_idle_gpu", "at": utc_now()})
            raise SystemExit(75)
        started, launch_time = time.monotonic(), utc_now()
        command = [sys.executable, str(Path(__file__).resolve()), "worker", "--run", str(run), "--deadline", str(time.time() + MAX_GPU_SECONDS)]
        with (run / "worker.log").open("w") as log:
            process = subprocess.Popen(command, env={**os.environ, **environment(gpu)}, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            timed_out, error_text, last_size, unchanged = False, None, -1, 0
            try:
                write_json(run / "launch.json", {"at": launch_time, "pid": process.pid, "process_group": process.pid, "gpu": gpu, "command": command})
                while process.poll() is None:
                    elapsed = time.monotonic() - started
                    if elapsed >= MAX_GPU_SECONDS - 20:
                        print("Hard lifecycle timeout: closing only this run's owned process group", flush=True)
                        timed_out = True
                        break
                    size = (run / "worker.log").stat().st_size
                    unchanged = unchanged + 1 if size == last_size else 0
                    last_size = size
                    write_json(run / "monitor.json", {"at": utc_now(), "pid": process.pid, "alive": True, "elapsed_seconds": elapsed, "log_bytes": size, "stall_advisory": unchanged >= 3})
                    if unchanged == 3:
                        print("Advisory: worker log unchanged for 90 seconds; continuing within fixed budget", flush=True)
                    time.sleep(min(30, max(0.1, MAX_GPU_SECONDS - 20 - elapsed)))
            except BaseException as error:
                error_text = f"{type(error).__name__}: {error}"
            finally:
                remaining = process_group_members(process.pid)
                cleanup = {"remaining_owned_pids_after_worker": remaining}
                if remaining:
                    signal_group(process.pid, signal.SIGTERM)
                    until = min(started + MAX_GPU_SECONDS - 5, time.monotonic() + 15)
                    while process_group_members(process.pid) and time.monotonic() < until:
                        time.sleep(0.2)
                    if process_group_members(process.pid):
                        signal_group(process.pid, signal.SIGKILL)
                        time.sleep(0.2)
                process.wait(timeout=2)
                cleanup["remaining_owned_pids_after_cleanup"] = process_group_members(process.pid)
                lifecycle = {"started_at": launch_time, "finished_at": utc_now(), "gpu": gpu, "pid": process.pid, "exit_code": process.returncode, "timed_out": timed_out, "seconds": time.monotonic() - started, "cleanup": cleanup, "supervisor_error": error_text}
                write_json(run / "resource_receipt.json", lifecycle)
        collect(run, lifecycle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "run", "worker", "analyze"])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--deadline", type=float)
    args = parser.parse_args()
    if args.mode == "analyze":
        analyze(args.run.resolve())
    else:
        {"prepare": prepare, "run": supervise, "worker": worker}[args.mode](args)


if __name__ == "__main__":
    main()
