"""One immutable inference shard, with atomic checkpoints and attempt accounting."""

import argparse
import fcntl
import json
import os
import subprocess
import time
from pathlib import Path

import protocol


def gpu_memory(uuid):
    raw = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=uuid,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return next(int(line.split(",")[1]) for line in raw.splitlines() if line.split(",")[0].strip() == uuid)


class Ledger:
    def __init__(self, path, limit):
        self.path, self.limit = path, limit
        self.used = (
            sum(r["requests"] for r in protocol.read_jsonl(path) if r["event"] == "reserved") if path.exists() else 0
        )

    def append(self, row):
        with self.path.open("a") as stream:
            stream.write(json.dumps({"time": protocol.utc_now(), **row}, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def reserve(self, qid, requests):
        if self.used + requests > self.limit:
            raise RuntimeError("Attempt budget exhausted; incomplete queries must not become fallback results")
        self.append({"event": "reserved", "qid": qid, "requests": requests})
        self.used += requests


def baseline_one(row, binding, binding_hash, engine, sampling, tokenizer, ids, ledger):
    from reze_backend import make_request, score_requests
    from reze_core import highlight_scores
    from reze_media import prepare_video

    started = time.time()
    clips, media = prepare_video(row)
    if media["source_sha256"] != binding["video_sha256"][row["vid"]]:
        raise ValueError("Video bytes changed after binding")
    requests = [make_request(tokenizer, row["query"], clip) for clip in clips]
    ledger.reserve(row["qid"], len(requests))
    records, elapsed = score_requests(engine, sampling, requests, ids)
    for item, request in zip(records, requests, strict=True):
        item["prompt"] = request["prompt"]
        item["prompt_sha256"] = protocol.digest(request["prompt"])
    ledger.append(
        {
            "event": "returned",
            "qid": row["qid"],
            "requests": len(records),
            "input_tokens": sum(r["prompt_tokens"] for r in records),
            "output_tokens": len(records),
        }
    )
    return {
        "status": "complete",
        "binding_sha256": binding_hash,
        "completed_at": protocol.utc_now(),
        "prediction": {k: row[k] for k in ("qid", "vid", "query")}
        | {
            "pred_saliency_scores": highlight_scores([r["pyes"] for r in records], media["segments"], row["duration"]),
        },
        "media": media,
        "clips": records,
        "cost": {
            "logical_requests": len(records),
            "input_tokens": sum(r["prompt_tokens"] for r in records),
            "output_tokens": len(records),
            "model_frames": media["model_input_frames"],
            "inference_seconds": elapsed,
            "query_seconds": time.time() - started,
        },
    }


def pair_one(row, binding, binding_hash, engine, sampling, tokenizer, ids, ledger):
    from backend import make_request, score_requests
    from candidate_frames import EvidenceUnavailable, prepare_candidates
    from ranking import candidates, make_scores, ordered_pairs

    started = time.time()
    baseline = protocol.read_baseline(binding["parent_run"], row)
    original = baseline["prediction"]["pred_saliency_scores"]
    selected = candidates(original)
    output = {
        "binding_sha256": binding_hash,
        "qid": row["qid"],
        "vid": row["vid"],
        "query": row["query"],
        "candidate_ids": selected,
        "arms": {},
    }
    try:
        targets, traces, decode_seconds = prepare_candidates(row, baseline, selected)
    except EvidenceUnavailable as error:
        output["candidate_frames"] = {}
        for arm in ("P_abs", "P_pair"):
            output["arms"][arm] = {
                "status": "fallback",
                "error": str(error),
                "scores": original,
                "requests": [],
                "cost": {
                    "logical_requests": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "model_frames": 0,
                    "inference_seconds": 0,
                },
            }
        return output
    output["candidate_frames"] = traces
    output["decode_preprocess_seconds"] = decode_seconds
    pairs = ordered_pairs(selected)
    # Submit both arms in alternating order; matching video evidence and pair order.
    specs = [(arm, a, b) for a, b in pairs for arm in ("P_abs", "P_pair")]
    requests = [make_request(tokenizer, row["query"], arm, targets[a], targets[b]) for arm, a, b in specs]
    ledger.reserve(row["qid"], len(requests))
    records, elapsed = score_requests(engine, sampling, requests, ids)
    ledger.append(
        {
            "event": "returned",
            "qid": row["qid"],
            "requests": len(records),
            "input_tokens": sum(r["prompt_tokens"] for r in records),
            "output_tokens": sum(r["output_tokens"] for r in records),
        }
    )
    for arm in ("P_abs", "P_pair"):
        arm_records = []
        for (name, a, b), request, record in zip(specs, requests, records, strict=True):
            if name != arm:
                continue
            arm_records.append(
                {
                    **record,
                    "a": a,
                    "b": b,
                    "prompt": request["prompt"],
                    "prompt_sha256": protocol.digest(request["prompt"]),
                    "evidence_sha256": protocol.digest([traces[str(a)], traces[str(b)]]),
                    "model_frames": len(targets[a]["video"]) + len(targets[b]["video"]),
                    "native_temporal_padding_frames": (len(targets[a]["video"]) % 2 + len(targets[b]["video"]) % 2),
                }
            )
        result = {
            "requests": arm_records,
            "cost": {
                "logical_requests": len(arm_records),
                "input_tokens": sum(r["prompt_tokens"] for r in arm_records),
                "output_tokens": sum(r["output_tokens"] for r in arm_records),
                "model_frames": sum(r["model_frames"] for r in arm_records),
                "inference_seconds": elapsed / 2,
                "time_attribution": "joint_batch_half_not_isolated_latency",
            },
        }
        try:
            for record in arm_records:
                if record.get("error"):
                    raise ValueError(record["error"])
                protocol.verify_logprob(record, strict=True)
            scores, diagnostics = make_scores(original, selected, arm_records, arm)
            result.update(status="complete", scores=scores, ranking=diagnostics)
        except ValueError as error:
            result.update(status="fallback", error=str(error), scores=original)
        output["arms"][arm] = result
    output.update(completed_at=protocol.utc_now(), query_seconds=time.time() - started)
    return output


def stress_one(row, binding, binding_hash, engine, sampling, tokenizer, ids, ledger):
    from backend import make_request, score_requests
    from candidate_frames import prepare_candidates
    from ranking import candidates

    baseline = protocol.read_baseline(binding["parent_run"], row)
    selected = candidates(baseline["prediction"]["pred_saliency_scores"])
    targets, traces, decode_seconds = prepare_candidates(row, baseline, selected)
    specs = [(arm, selected[i], selected[(i + 1) % 5]) for i in range(5) for arm in ("P_abs", "P_pair")]
    requests = [make_request(tokenizer, row["query"], arm, targets[a], targets[b]) for arm, a, b in specs]
    ledger.reserve(row["qid"], len(requests))
    records, seconds = score_requests(engine, sampling, requests, ids)
    for record, (arm, a, b), request in zip(records, specs, requests, strict=True):
        record.update(
            arm=arm,
            a=a,
            b=b,
            prompt=request["prompt"],
            evidence_sha256=protocol.digest([traces[str(a)], traces[str(b)]]),
        )
    ledger.append(
        {
            "event": "returned",
            "qid": row["qid"],
            "requests": len(records),
            "input_tokens": sum(r["prompt_tokens"] for r in records),
            "output_tokens": sum(r["output_tokens"] for r in records),
        }
    )
    return {
        "status": "complete",
        "binding_sha256": binding_hash,
        "qid": row["qid"],
        "candidate_ids": selected,
        "candidate_frames": traces,
        "requests": records,
        "inference_seconds": seconds,
        "decode_preprocess_seconds": decode_seconds,
        "accuracy_not_evaluated": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--shard", type=int, required=True)
    args = parser.parse_args()
    run = args.run
    binding, rows = protocol.load_run(run)
    binding_hash = protocol.sha256(run / "binding.json")
    spec = binding["workers"][args.shard]
    rows = [row for index, row in enumerate(rows) if index % len(binding["workers"]) == args.shard]
    lock = (run / f"worker_{args.shard}.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    import torch

    torch.set_num_threads(2)
    device_uuid = str(torch.cuda.get_device_properties(0).uuid)
    if device_uuid.removeprefix("GPU-") != spec["uuid"].removeprefix("GPU-"):
        raise RuntimeError(f"GPU UUID mismatch: {device_uuid}")
    ledger = Ledger(run / f"attempts_{args.shard}.jsonl", spec["attempt_request_cap"])
    progress_path = run / f"worker_{args.shard}.json"

    def progress(**extra):
        protocol.atomic_json(
            progress_path,
            {
                "time": protocol.utc_now(),
                "pid": os.getpid(),
                "shard": args.shard,
                "total": len(rows),
                "gpu": spec,
                **extra,
            },
        )

    pending, done = [], 0
    for row in rows:
        path = run / "records" / f"{row['qid']}.json"
        if path.exists():
            record = json.loads(path.read_text())
            if binding["mode"] == "baseline":
                protocol.validate_baseline(record, row, binding, binding_hash)
            elif binding["mode"] == "stress":
                protocol.validate_stress_record(record, row, binding, binding_hash)
            else:
                baseline = protocol.read_baseline(binding["parent_run"], row)
                protocol.validate_pair_record(record, row, binding, binding_hash, baseline)
            done += 1
        else:
            pending.append(row)
    if not pending:
        progress(status="complete", done=done)
        return
    progress(status="loading", done=done)
    if binding["mode"] == "baseline":
        from reze_backend import make_engine
    else:
        from backend import make_engine
    audit = json.loads((run / "source_snapshot/manifests/reze_model_audit.json").read_text())
    engine, sampling, tokenizer, ids = make_engine(model_path=audit["model_path"])
    progress(status="ready", done=done, gpu_memory_mib=gpu_memory(spec["uuid"]))
    for row in pending:
        progress(status="processing", done=done, qid=row["qid"], gpu_memory_mib=gpu_memory(spec["uuid"]))
        function = {"baseline": baseline_one, "pair": pair_one, "stress": stress_one}[binding["mode"]]
        record = function(row, binding, binding_hash, engine, sampling, tokenizer, ids, ledger)
        if binding["mode"] == "baseline":
            protocol.validate_baseline(record, row, binding, binding_hash)
        elif binding["mode"] == "stress":
            protocol.validate_stress_record(record, row, binding, binding_hash)
        else:
            protocol.validate_pair_record(
                record, row, binding, binding_hash, protocol.read_baseline(binding["parent_run"], row)
            )
        record["gpu_memory_mib"] = gpu_memory(spec["uuid"])
        record["worker_shard"] = args.shard
        protocol.atomic_json(run / "records" / f"{row['qid']}.json", record)
        done += 1
        progress(status="running", done=done, qid=row["qid"], gpu_memory_mib=record["gpu_memory_mib"])
        print(
            json.dumps(
                {
                    "time": protocol.utc_now(),
                    "qid": row["qid"],
                    "done": done,
                    "total": len(rows),
                    "gpu_memory_mib": record["gpu_memory_mib"],
                }
            ),
            flush=True,
        )
    progress(status="complete", done=done)


if __name__ == "__main__":
    main()
