"""Paired character-limit probe with frozen pixels and one bounded GPU lifecycle."""

import argparse
import fcntl
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
from frame_observer import FRAME_OUTPUT_CAP, FRAME_PROMPT, has_visible_claim, still_payload
from io_utils import ROOT, compact, digest, legacy, parents, read_json, sha256, utc_now, write_json
from observer_output_contract import RelaxedFrameDescription, select_jobs, unbounded_schema, validate_unbounded
from r2_observer import read_frames, source_timeline

MAX_REQUEST = 2048
MAX_TOTAL = 37 * MAX_REQUEST
MAX_GPU_SECONDS = 900
LIMITS = {"requests": 37, "extra_attempts": 0, "arm_incremental_tokens": {"E_uniform": MAX_TOTAL, "E_targeted": MAX_TOTAL}}


class ProbeLedger(Ledger):
    def __init__(self, path, *, deadline):
        super().__init__(path, LIMITS)
        self.deadline = deadline

    def reserve(self, key, prompt_tokens, output_cap, arms, request_hash, frames):
        if time.time() >= self.deadline - 60:
            raise BudgetStop("GPU lifecycle includes a 60-second shutdown reserve")
        reservation = prompt_tokens + output_cap
        if reservation > MAX_REQUEST:
            raise BudgetStop("Per-request 2048-token reservation exceeded")
        if self.cost() + reservation > MAX_TOTAL:
            raise BudgetStop("Global 75776-token reservation exceeded")
        return super().reserve(key, prompt_tokens, output_cap, arms, request_hash, frames)


def verify(run):
    manifest = read_json(run / "manifest.json")
    for path, expected in manifest["parent_receipts"].items():
        if sha256(path) != expected:
            raise ValueError(f"Parent artifact changed: {path}")
    for name, expected in manifest["source_sha256"].items():
        if sha256(ROOT / name) != expected or sha256(run / "source_snapshot" / name) != expected:
            raise ValueError(f"Frozen source changed: {name}")
    for name, expected in manifest["input_sha256"].items():
        if sha256(run / name) != expected:
            raise ValueError(f"Frozen input changed: {name}")
    model = read_json(Path(manifest["parent_run"]) / "model_verification.json")
    for name, expected in model["stats"].items():
        stat = (Path(MODEL) / name).stat()
        if stat.st_size != expected["size"] or stat.st_mtime_ns != expected["mtime_ns"]:
            raise ValueError("Model drift")
    return manifest


def prepare(args):
    run, reference = args.run.resolve(), args.reference.resolve()
    if run.exists():
        raise FileExistsError("Preserve existing runs; choose a new version")
    original = read_json(reference / "manifest.json")
    parent = Path(original["parent_run"])
    jobs = select_jobs(reference)
    if len(jobs) != 37:
        raise ValueError("Frozen probe expects 37 requests")
    # Authenticate the inherited receipts before creating any new freeze.
    # A current-file hash must never silently replace a historical commitment.
    completion = read_json(reference / "complete.json")
    if completion["manifest_sha256"] != sha256(reference / "manifest.json") or completion["ledger_sha256"] != sha256(reference / "probe/ledger.json"):
        raise ValueError("Reference completion receipt changed")
    for path, expected in original["parent_receipts"].items():
        if sha256(path) != expected:
            raise ValueError(f"Inherited parent receipt changed: {path}")
    original_ledger = read_json(reference / "probe/ledger.json")
    attempts = {item["key"]: item for item in original_ledger["attempts"]}
    for job in jobs:
        value = read_json(job["response_path"])
        if digest(value) != attempts[job["key"]]["record_sha256"] or value["request_sha256"] != digest(value["specification"]) or value["parsed"] != job["record"]["result"]:
            raise ValueError("Selected reference response is not the committed result")
    run.mkdir(parents=True)
    shutil.copy2(args.protocol, run / "protocol.md")
    write_json(run / "jobs.json", jobs)
    write_json(run / "output_schema.json", unbounded_schema())
    receipts = dict(original["parent_receipts"])
    for path in (reference / "manifest.json", reference / "complete.json", reference / "probe/ledger.json", reference / "review/assistant_findings.json", parent / "model_verification.json", parent / "binding.json", parent / "inference.jsonl"):
        receipts[str(path)] = sha256(path)
    for job in jobs:
        for key in ("record_path", "response_path"):
            receipts[job[key]] = sha256(job[key])
        path = parent / "R2/records" / f"{job['qid']}.json"
        receipts[str(path)] = sha256(path)
    source_paths = {ROOT / name for name in original["source_sha256"]}
    source_paths.update(Path(__file__).parent.rglob("*.py"))
    sources = {}
    for path in sorted(source_paths):
        name = str(path.relative_to(ROOT))
        target = run / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        sources[name] = sha256(path)
    manifest = {
        "created_at": utc_now(), "protocol": "observer-output-character-limit-v1", "reference": str(reference),
        "parent_run": str(parent), "model_revision": original["model_revision"],
        "parent_receipts": receipts, "source_sha256": sources,
        "input_sha256": {name: sha256(run / name) for name in ("protocol.md", "jobs.json", "output_schema.json")},
        "limits": LIMITS, "maximum_reserved_tokens_per_request": MAX_REQUEST,
        "maximum_total_tokens": MAX_TOTAL, "maximum_gpu_seconds": MAX_GPU_SECONDS,
        "maximum_worker_launches": 1, "output_cap": FRAME_OUTPUT_CAP,
        "change": "Only remove visible.maxLength and uncertainty.maxLength from the frozen output schema",
        "new_hd_predictions": False, "automatic_memory_promotion": False,
        "passport": {"Origin Skill": "experiment-agent", "Origin Mode": "run", "Origin Date": utc_now(), "Verification Status": "UNVERIFIED", "Version Label": "observer_output_v1"},
    }
    write_json(run / "manifest.json", manifest)
    verify(run)
    write_json(run / "status.json", {"status": "prepared", "expected": len(jobs), "at": utc_now()})
    print(compact({"status": "prepared", "requests": len(jobs), "manifest_sha256": sha256(run / "manifest.json")}))


def worker(args):
    run = args.run.resolve()
    manifest = verify(run)
    jobs = read_json(run / "jobs.json")
    import torch
    from PIL import Image

    torch.set_num_threads(2)
    actual = "GPU-" + str(torch.cuda.get_device_properties(0).uuid).removeprefix("GPU-")
    if torch.cuda.device_count() != 1 or actual != os.environ["REZE_EXPECTED_GPU_UUID"]:
        raise ValueError("GPU UUID mismatch")
    directory = run / "probe"
    directory.mkdir(exist_ok=True)
    ledger = ProbeLedger(directory / "ledger.json", deadline=args.deadline)
    backend = Backend(directory, ledger)
    parent = Path(manifest["parent_run"])
    binding = read_json(parent / "binding.json")
    rows = {row["qid"]: row for row in legacy.read_jsonl(parent / "inference.jsonl")}
    current_qid, current_window, completed = None, None, 0
    reader = None
    for job in jobs:
        qid = job["qid"]
        if qid != current_qid:
            if reader is not None:
                del reader
            row = rows[qid]
            baseline, _ = parents(binding, row)
            if sha256(row["video_path"]) != baseline["media"]["source_sha256"]:
                raise ValueError("Source video bytes changed")
            original = read_json(parent / "R2/records" / f"{qid}.json")
            reader, timestamps, _ = source_timeline(row, baseline)
            current_qid = qid
        window = (qid, job["arm"], job["number"])
        if window != current_window:
            old = original["observations"][job["arm"]][job["number"]]
            video, trace = read_frames(reader, timestamps, [f["frame_index"] for f in old["trace"]["frames"]], row, old["action"], baseline["media"]["resized_height_width"], job["arm"])
            if trace != old["trace"]:
                raise ValueError("Parent source frames changed")
            current_window = window
        payload, proof = still_payload(video, trace, job["position"])
        reference = read_json(job["record_path"])
        raw_reference = read_json(job["response_path"])
        expected = raw_reference["specification"]
        if proof != reference["proof"] or FRAME_PROMPT != expected["text"]:
            raise ValueError("Paired observer input changed")
        if backend.request(FRAME_PROMPT, [payload])["prompt"] != expected["rendered_prompt"]:
            raise ValueError("Paired rendered model prompt changed")
        output_schema = unbounded_schema()
        old_schema = read_json(job["response_path"])["specification"]["schema"]
        del old_schema["properties"]["visible"]["maxLength"]
        del old_schema["properties"]["uncertainty"]["maxLength"]
        if output_schema != old_schema:
            raise ValueError("Output probe changed more than two schema fields")
        key = f"{qid}/{job['arm']}/unbounded_frame/{job['number']}/{job['position']}"
        result = {"key": key, "qid": qid, "arm": job["arm"], "number": job["number"], "position": job["position"], "proof": proof, "reference_record": job["record_path"], "reference_response": job["response_path"], "selection_reasons": job["selection_reasons"]}
        try:
            value, raw = backend.call(key, FRAME_PROMPT, [payload], [proof], RelaxedFrameDescription, FRAME_OUTPUT_CAP, [job["arm"]], validate_unbounded, output_schema=output_schema)
            result.update(status="complete", result=value, response_path=raw["response_path"])
            for field in ("text", "system", "rendered_prompt", "evidence", "output_cap", "arms"):
                if raw["specification"][field] != expected[field]:
                    raise ValueError(f"Paired {field} changed")
        except SemanticFailure as error:
            result.update(status="invalid", error=str(error), response_path=str(directory / "attempts" / f"{ledger.attempts_for(key)[-1]['number']:05d}.json"))
        name = f"{qid}_{job['arm']}_{job['number']}_{job['position']:02d}"
        image_path = run / "images" / f"{name}.png"
        image_path.parent.mkdir(exist_ok=True)
        Image.fromarray(video["video"][job["position"]]).save(image_path)
        result["review_image"] = str(image_path)
        write_json(run / "records" / f"{name}.json", result)
        completed += 1
        write_json(run / "status.json", {"status": "running", "completed": completed, "expected": 37, "at": utc_now()})
        print(f"Completed {completed}/37: {key} ({result['status']})", flush=True)
    write_json(run / "worker_complete.json", {"at": utc_now(), "requests": completed, "ledger_sha256": sha256(ledger.path)})


def collect(run, lifecycle):
    jobs = read_json(run / "jobs.json")
    expected = {f"{j['qid']}/{j['arm']}/unbounded_frame/{j['number']}/{j['position']}": j for j in jobs}
    records = [read_json(path) for path in sorted((run / "records").glob("*.json"))]
    ledger_path = run / "probe/ledger.json"
    ledger = Ledger(ledger_path, LIMITS) if ledger_path.exists() else None
    pairs = []
    for record in records:
        if record["key"] not in expected or record["proof"] != expected[record["key"]]["record"]["proof"]:
            raise ValueError("New record differs from the frozen job/source proof")
        before = read_json(record["reference_response"])
        after = read_json(record["response_path"])
        attempt = ledger.attempts_for(record["key"])[-1]
        if digest(after) != attempt["record_sha256"] or after["request_sha256"] != digest(after["specification"]):
            raise ValueError("New response receipt mismatch")
        if record["status"] != after["status"] or (record["status"] == "complete" and record["result"] != after["parsed"]):
            raise ValueError("New record differs from the committed response")
        pairs.append({
            "key": record["key"], "qid": record["qid"], "arm": record["arm"],
            "frame": record["proof"]["frames"][0], "selection_reasons": record["selection_reasons"],
            "before": before["parsed"], "after": after.get("parsed"), "new_status": after["status"],
            "before_tokens": before["prompt_tokens"] + before["output_tokens"],
            "after_tokens": after.get("prompt_tokens", 0) + after.get("output_tokens", 0),
            "finish_reason": after.get("finish_reason"), "review_image": record["review_image"],
            "assistant_review": "pending", "independent_human_review": "not_reviewed",
        })
    valid = [pair for pair in pairs if pair["new_status"] == "complete"]
    keys = [record["key"] for record in records]
    worker_receipt = read_json(run / "worker_complete.json") if (run / "worker_complete.json").exists() else {}
    checks = {
        "worker_exit_zero": lifecycle["exit_code"] == 0,
        "exact_frozen_job_set": len(keys) == 37 and len(set(keys)) == 37 and set(keys) == set(expected),
        "exact_ledger_job_set": ledger is not None and len(ledger.state["attempts"]) == 37 and {a["key"] for a in ledger.state["attempts"]} == set(expected),
        "worker_ledger_receipt": ledger is not None and worker_receipt.get("ledger_sha256") == sha256(ledger.path),
        "gpu_within_budget": lifecycle["seconds"] <= MAX_GPU_SECONDS,
        "owned_processes_released": not lifecycle["cleanup"]["remaining_owned_pids_after_cleanup"],
        "token_budget": ledger is None or ledger.cost() <= MAX_TOTAL,
    }
    summary = {
        "at": utc_now(), "status": "complete" if all(checks.values()) else "partial", "completion_checks": checks,
        "expected_requests": 37, "processed": len(records), "valid": len(valid),
        "statuses": dict(Counter(pair["new_status"] for pair in pairs)),
        "finish_reasons": dict(Counter(pair["finish_reason"] for pair in pairs)),
        "before_visible_empty": sum(not has_visible_claim(pair["before"]["visible"]) for pair in pairs),
        "after_visible_empty": sum(not has_visible_claim(pair["after"]["visible"]) for pair in valid),
        "before_visible_at_character_cap": sum(len(pair["before"]["visible"]) == 160 for pair in pairs),
        "before_uncertainty_at_character_cap": sum(len(pair["before"]["uncertainty"]) == 80 for pair in pairs),
        "after_visible_exceeds_old_cap": sum(len(pair["after"]["visible"]) > 160 for pair in valid),
        "after_uncertainty_exceeds_old_cap": sum(len(pair["after"]["uncertainty"]) > 80 for pair in valid),
        "new_attempts": len(ledger.state["attempts"]) if ledger else 0,
        "new_charged_tokens": ledger.cost() if ledger else 0,
        "paired_historical_tokens": sum(pair["before_tokens"] for pair in pairs),
        "lifecycle": lifecycle, "new_hd_predictions": False, "memory_promoted": False,
        "independent_human_review_count": 0,
    }
    write_json(run / "review_pairs.json", pairs)
    write_json(run / "summary.json", summary)
    write_json(run / "status.json", {"status": summary["status"], "completed": len(records), "expected": 37, "at": utc_now()})
    receipts = {name: sha256(run / name) for name in ("manifest.json", "probe/ledger.json", "review_pairs.json", "resource_receipt.json", "summary.json") if (run / name).exists()}
    write_json(run / ("complete.json" if summary["status"] == "complete" else "stopped.json"), {"status": summary["status"], "receipts": receipts, "at": utc_now()})
    print(compact(summary))


def supervise(args):
    run = args.run.resolve()
    with (run / "runner.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify(run)
        if (run / "launch.json").exists():
            raise ValueError("This probe permits only one worker launch; retain any failed attempt")
        gpu, _ = idle_gpu("all")
        if gpu is None or idle_gpu([gpu["uuid"]])[0] is None:
            write_json(run / "status.json", {"status": "waiting_for_idle_gpu", "at": utc_now()})
            raise SystemExit(75)
        started = time.monotonic()
        launch_time = utc_now()
        command = [sys.executable, str(Path(__file__).resolve()), "worker", "--run", str(run), "--deadline", str(time.time() + MAX_GPU_SECONDS)]
        with (run / "worker.log").open("w") as log:
            process = subprocess.Popen(command, env={**os.environ, **environment(gpu)}, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            timed_out, error_text = False, None
            try:
                write_json(run / "launch.json", {"at": launch_time, "pid": process.pid, "process_group": process.pid, "gpu": gpu, "command": command})
                while process.poll() is None:
                    elapsed = time.monotonic() - started
                    if elapsed >= 880:
                        timed_out = True
                        break
                    write_json(run / "monitor.json", {"at": utc_now(), "pid": process.pid, "alive": True, "elapsed_seconds": elapsed, "log_bytes": (run / "worker.log").stat().st_size})
                    time.sleep(min(30, max(0.1, 880 - elapsed)))
            except BaseException as error:
                error_text = f"{type(error).__name__}: {error}"
            finally:
                # This group was created by this supervisor for this one job.
                # Always close its engines, including monitor-write failures.
                remaining = process_group_members(process.pid)
                cleanup = {"remaining_owned_pids_after_worker": remaining}
                if remaining:
                    signal_group(process.pid, signal.SIGTERM)
                    until = min(started + 895, time.monotonic() + 15)
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


def process_group_members(group):
    result = subprocess.run(["ps", "-eo", "pid=,pgid=,stat="], check=True, capture_output=True, text=True)
    return [int(pid) for line in result.stdout.splitlines() if len(fields := line.split()) == 3 for pid, pgid, state in [fields] if int(pgid) == group and not state.startswith("Z")]


def signal_group(group, sig):
    try:
        os.killpg(group, sig)
    except ProcessLookupError:
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "run", "worker"])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--deadline", type=float)
    args = parser.parse_args()
    {"prepare": prepare, "run": supervise, "worker": worker}[args.mode](args)


if __name__ == "__main__":
    main()
