"""Freeze/run a bounded same-pixel still-observer diagnosis, with two independent GPU slots."""

import argparse
import fcntl
import os
import shutil
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from backend import MODEL, Backend, SemanticFailure
from dispatch import environment, idle_gpu
from frame_observer import (
    FRAME_OUTPUT_CAP,
    FRAME_PROMPT,
    FrameDescription,
    grounded_description,
    still_payload,
    validate_description,
)
from io_utils import HERE, ROOT, digest, legacy, parents, read_json, sha256, utc_now, write_json
from r2_observation_ablation import TimedLedger
from r2_observer import read_frames, source_timeline
from r2_protocol import SharedLedger


def grounding_audit(parent, rows, run):
    from comparison import allowed_references
    from evidence_grounding import ground_observation
    from memory import retrieve
    from r2_planner import candidate_ranges, update_memory
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    regions, cases = Counter(), []
    for row in rows:
        original = read_json(parent / "R2/records" / f"{row['qid']}.json")
        targets = {item["candidate"]: item["target"] for item in candidate_ranges(original["candidate_ids"], row["duration"])}
        for arm in ("E_uniform", "E_targeted"):
            observations = original["observations"][arm]
            legacy_memory = update_memory(original["memories"]["E_memory"], observations, tokenizer)
            if legacy_memory != original["memories"][arm]:
                raise ValueError("Legacy memory replay changed")
            scoped = update_memory(original["memories"]["E_memory"], observations, tokenizer, target_intervals=targets)
            view = retrieve(scoped, tokenizer)
            for item in view["items"]:
                for identity, source in item["sources"].items():
                    if "grounding" in source and source["grounding"]["scope"] != "inside":
                        for c in targets:
                            if identity in allowed_references(c, {c: []}, view):
                                raise ValueError("Context remained directly citable")
            all_facts = {}
            for number, observation in enumerate(observations):
                c = observation["action"]["candidate"]
                facts = ground_observation(observation["result"], observation["trace"], c, targets[c], f"obs{number}_c{c}")
                all_facts.update(facts)
                regions.update(f["grounding"]["scope"] for f in facts.values())
            item = {"qid": row["qid"], "arm": arm, "legacy_replay_exact": True, "tokens": scoped["tokens"], "capacity_evictions": len(scoped["capacity_evictions"])}
            cases.append(item)
            write_json(run / "binding_audit" / f"{row['qid']}_{arm}.json", {**item, "facts": all_facts, "scoped_memory": scoped})
    summary = {"legacy_memories_replayed": len(cases), "scopes": dict(regions), "cases": cases, "new_model_calls": 0, "semantic_accuracy_evaluated": False}
    write_json(run / "binding_audit.json", summary)
    return summary


def prepare(args):
    run, previous = args.run.resolve(), args.reference_probe.resolve()
    if run.exists():
        raise ValueError("Preserve previous attempts; choose a new directory")
    reference = read_json(previous / "manifest.json")
    done = read_json(previous / "complete.json")
    if done["manifest_sha256"] != sha256(previous / "manifest.json") or done["ledger_sha256"] != sha256(previous / "probe/ledger.json"):
        raise ValueError("Reference probe receipt drift")
    parent = Path(reference["parent_run"])
    qids = reference["observation_qids"]
    rows = {row["qid"]: row for row in legacy.read_jsonl(parent / "inference.jsonl")}
    receipts = dict(reference["parent_receipts"])
    for name in ("manifest.json", "complete.json", "probe/ledger.json", "decision.json", "review/assistant_review.json"):
        receipts[str(previous / name)] = sha256(previous / name)
    jobs = []
    record_receipts = read_json(parent / "R2/record_receipt.json")
    for qid in qids:
        original = parent / "R2/records" / f"{qid}.json"
        if sha256(original) != record_receipts[str(qid)]:
            raise ValueError("Parent record changed")
        record = read_json(original)
        for arm in ("E_uniform", "E_targeted"):
            for number, observation in enumerate(record["observations"][arm]):
                path = previous / "observations" / f"{qid}_{arm}_{number}.json"
                if read_json(path)["trace"] != observation["trace"]:
                    raise ValueError("Reference probe source trace mismatch")
                receipts[str(path)] = sha256(path)
                for position, frame in enumerate(observation["trace"]["frames"]):
                    jobs.append({"qid": qid, "arm": arm, "number": number, "position": position, "frame": frame})
    if len(qids) != 7 or len(jobs) != 672:
        raise ValueError("Expected fixed seven queries and all 672 parent frame occurrences")
    for name, expected in receipts.items():
        if sha256(name) != expected:
            raise ValueError("Historical receipt changed")
    run.mkdir()
    legacy.write_jsonl(run / "inference.jsonl", [rows[qid] for qid in qids])
    legacy.write_jsonl(run / "frame_jobs.jsonl", jobs)
    shutil.copy2(args.protocol, run / "implementation_plan.md")
    audit = grounding_audit(parent, [rows[qid] for qid in qids], run)
    names = set(reference["source_sha256"])
    names.update(str(p.relative_to(ROOT)) for p in HERE.rglob("*") if p.suffix in {".py", ".md"} and "__pycache__" not in p.parts)
    source_hashes = {}
    for name in sorted(names):
        target = run / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
        source_hashes[name] = sha256(target)
    manifest = {
        "protocol": "revalue-frame-grounding-v1", "created_at": utc_now(),
        "queue_expires_at": (datetime.now(timezone.utc) + timedelta(hours=72)).isoformat(),
        "parent_run": str(parent), "reference_probe": str(previous), "qids": qids,
        "slots": 2, "maximum_worker_launches_per_slot": 2, "maximum_gpu_seconds_per_slot": 3600,
        "maximum_summed_gpu_seconds": 7200,
        "limits": {"requests": 672, "extra_attempts": 0, "arm_incremental_tokens": {
            "C_direct4": 0, "E_memory": 0, "E_uniform": 336 * 2048, "E_targeted": 336 * 2048,
        }},
        "maximum_reserved_tokens_per_request": 2048, "output_cap": FRAME_OUTPUT_CAP,
        "source_sha256": source_hashes, "parent_receipts": receipts,
        "input_sha256": {name: sha256(run / name) for name in ("inference.jsonl", "frame_jobs.jsonl", "implementation_plan.md", "binding_audit.json")},
        "offline_binding_audit": audit,
        "model_revision": read_json(parent / "binding.json")["model_revision"],
        "review_positions": [0, 4, 8, 12, 16, 20, 23],
        "known_error_frames": {"1129": ["s1122", "s1214", "s1410", "s1461"], "881": ["s2513"], "5172": ["s1291"], "9841": ["s2197"]},
        "comparisons_rerun": False, "new_hd_metrics": False,
    }
    write_json(run / "manifest.json", manifest)
    SharedLedger(run / "probe", manifest["limits"])
    write_json(run / "status.json", {"status": "prepared", "at": utc_now(), "expected_frame_calls": 672})
    print(f"Frozen {run}; manifest={sha256(run / 'manifest.json')}", flush=True)


def verify(run):
    manifest = read_json(run / "manifest.json")
    for field, root in (("source_sha256", ROOT), ("input_sha256", run), ("parent_receipts", Path("/"))):
        for name, expected in manifest[field].items():
            if sha256(root / name) != expected:
                raise ValueError(f"Frozen input drift: {name}")
    model = read_json(Path(manifest["parent_run"]) / "model_verification.json")
    for name, expected in model["stats"].items():
        stat = (Path(MODEL) / name).stat()
        if stat.st_size != expected["size"] or stat.st_mtime_ns != expected["mtime_ns"]:
            raise ValueError("Model drift")
    return manifest


class FrameLedger(TimedLedger):
    def reserve(self, key, prompt_tokens, output_cap, *args, **kwargs):
        if prompt_tokens + output_cap > 2048:
            raise ValueError("Frame request exceeds frozen token reservation")
        return super().reserve(key, prompt_tokens, output_cap, *args, **kwargs)


def worker(args):
    run = args.run.resolve()
    slot = run / f"slot_{args.slot}"
    slot.mkdir(exist_ok=True)
    with (slot / "worker.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        if (slot / "complete.json").exists() or (slot / "stopped.json").exists():
            return
        manifest = verify(run)
        if not 0 <= args.slot < manifest["slots"]:
            raise ValueError("Unknown worker slot")
        if time.time() >= datetime.fromisoformat(manifest["queue_expires_at"]).timestamp():
            write_json(slot / "stopped.json", {"reason": "queue_expired", "at": utc_now()})
            return
        gpu, _ = idle_gpu([args.gpu_uuid])
        if gpu is None or idle_gpu([args.gpu_uuid])[0] is None:
            write_json(slot / "status.json", {"status": "waiting_for_idle_gpu", "at": utc_now()})
            raise SystemExit(75)
        os.environ.update(environment(gpu))
        attempts_path = slot / "worker_attempts.json"
        attempts = read_json(attempts_path) if attempts_path.exists() else []
        if any(a["status"] == "running" for a in attempts):
            raise RuntimeError("Interrupted worker requires elapsed-time reconciliation")
        if len(attempts) >= manifest["maximum_worker_launches_per_slot"]:
            raise ValueError("Worker launch cap")
        previous_seconds = sum(a["seconds"] for a in attempts)
        started = time.monotonic()
        attempt = {"status": "running", "gpu": gpu, "pid": os.getpid(), "started_at": utc_now(), "seconds": 0}
        attempts.append(attempt)
        write_json(attempts_path, attempts)
        try:
            import torch

            torch.set_num_threads(2)
            actual = "GPU-" + str(torch.cuda.get_device_properties(0).uuid).removeprefix("GPU-")
            if torch.cuda.device_count() != 1 or actual != gpu["uuid"]:
                raise ValueError("GPU UUID mismatch")
            ledger = FrameLedger(run / "probe", manifest["limits"], started, previous_seconds, manifest["maximum_gpu_seconds_per_slot"])
            backend = Backend(run / "probe", ledger, status_path=slot / "generation.json")
            parent = Path(manifest["parent_run"])
            binding = read_json(parent / "binding.json")
            rows = legacy.read_jsonl(run / "inference.jsonl")[args.slot::manifest["slots"]]
            completed, invalid = 0, 0
            for row in rows:
                qid = row["qid"]
                baseline, _ = parents(binding, row)
                if sha256(row["video_path"]) != baseline["media"]["source_sha256"]:
                    raise ValueError("Source video bytes changed")
                original = read_json(parent / "R2/records" / f"{qid}.json")
                reader, timestamps, _ = source_timeline(row, baseline)
                for arm in ("E_uniform", "E_targeted"):
                    for number, old in enumerate(original["observations"][arm]):
                        video, trace = read_frames(reader, timestamps, [f["frame_index"] for f in old["trace"]["frames"]], row, old["action"], baseline["media"]["resized_height_width"], arm)
                        if trace != old["trace"]:
                            raise ValueError("Original/resized observation pixels or PTS drift")
                        candidate = old["action"]["candidate"]
                        target = [2.0 * candidate, min(2.0 * candidate + 2, row["duration"])]
                        for position in range(len(trace["frames"])):
                            payload, proof = still_payload(video, trace, position)
                            key = f"{qid}/{arm}/frame_observe/{number}/{position}"
                            record = {"qid": qid, "arm": arm, "number": number, "position": position, "key": key, "proof": proof, "target_interval": target, "candidate": candidate}
                            try:
                                value, raw = backend.call(key, FRAME_PROMPT, [payload], [proof], FrameDescription, FRAME_OUTPUT_CAP, [arm], validate_description, output_schema=FrameDescription.model_json_schema())
                                record.update(status="complete", result=value, response_path=raw["response_path"], grounded_facts=grounded_description(value, proof, candidate, target, f"obs{number}_p{position}_c{candidate}"))
                            except SemanticFailure as error:
                                invalid += 1
                                record.update(status="invalid", error=str(error))
                            write_json(run / "frames" / f"{qid}_{arm}_{number}_{position:02d}.json", record)
                            completed += 1
                            write_json(slot / "status.json", {"status": "running", "processed_frames": completed, "invalid_frames": invalid, "expected": len(rows) * 96, "at": utc_now()})
                            attempt["seconds"] = time.monotonic() - started
                            write_json(attempts_path, attempts)
                        print(f"slot {args.slot}: {completed}/{len(rows) * 96}, invalid {invalid}", flush=True)
                del reader
            state = {"status": "complete", "processed_frames": completed, "invalid_frames": invalid, "at": utc_now()}
            write_json(slot / "complete.json", state)
            write_json(slot / "status.json", state)
            attempt["status"] = "complete"
        except Exception as error:
            write_json(slot / "stopped.json", {"status": "stopped", "error": f"{type(error).__name__}: {error}", "at": utc_now()})
            attempt["status"] = "failed"
            raise
        finally:
            attempt.update(seconds=time.monotonic() - started, finished_at=utc_now())
            write_json(attempts_path, attempts)


def finish(args):
    run = args.run.resolve()
    manifest = verify(run)
    ledger = SharedLedger(run / "probe", manifest["limits"])
    jobs = legacy.read_jsonl(run / "frame_jobs.jsonl")
    records = []
    for job in jobs:
        path = run / "frames" / f"{job['qid']}_{job['arm']}_{job['number']}_{job['position']:02d}.json"
        record = read_json(path)
        if record["proof"]["frames"] != [job["frame"]]:
            raise ValueError("Completed frame proof mismatch")
        if record["status"] == "complete":
            raw = read_json(record["response_path"])
            attempt = ledger.attempts_for(record["key"])[-1]
            if digest(raw) != attempt["record_sha256"] or raw["parsed"] != record["result"]:
                raise ValueError("Completed request receipt mismatch")
        records.append(record)
    if len(ledger.state["attempts"]) != len(jobs):
        raise ValueError("Expected one attempt per frame, without retries")
    state = {
        "status": "complete", "at": utc_now(), "processed_frames": len(records),
        "valid_frames": sum(r["status"] == "complete" for r in records),
        "new_calls": len(ledger.state["attempts"]), "actual_tokens": ledger.cost(),
        "manifest_sha256": sha256(run / "manifest.json"), "ledger_sha256": sha256(ledger.path),
        "same_source_pixels_verified": True, "source_occurrences": 672, "displayed_frames_including_duplicates": 1344,
        "semantic_review": "pending", "new_hd_metrics": False, "comparisons_rerun": False,
    }
    write_json(run / "complete.json", state)
    write_json(run / "status.json", state)
    print(state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "worker", "finish"])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reference-probe", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--slot", type=int)
    parser.add_argument("--gpu-uuid")
    args = parser.parse_args()
    {"prepare": prepare, "worker": worker, "finish": finish}[args.mode](args)


if __name__ == "__main__":
    main()
