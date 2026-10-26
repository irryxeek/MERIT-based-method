"""Bounded GPU V1 worker with frame receipts, replay and conservative billing."""

import argparse
import fcntl
import json
import os
import time
import uuid
from pathlib import Path

from core import (
    dispersed_candidates,
    legacy,
    load_run,
    ordered_pairs,
    parent_records,
    permutation_scores,
    validate_record,
)


class Ledger:
    def __init__(self, run, binding):
        self.run, self.binding = run, binding

    def totals(self):
        attempts = [json.loads(p.read_text()) for p in (self.run / "request_attempts").glob("*.json")]
        return {
            "engineering": sum(a["requests"] for a in attempts if a["kind"] == "engineering"),
            "formal": sum(a["requests"] for a in attempts if a["kind"] == "formal"),
            "tokens": sum(a.get("actual_tokens", a["reserved_tokens"]) for a in attempts),
        }

    def reserve(self, qid, kind, count):
        totals = self.totals()
        cap = self.binding["maximum_engineering_requests" if kind == "engineering" else "maximum_formal_attempts"]
        if totals[kind] + count > cap or totals["tokens"] + count * 4097 > self.binding["maximum_incremental_tokens"]:
            raise RuntimeError("Frozen request or token budget exhausted")
        path = self.run / "request_attempts" / f"{kind}_{qid}_{uuid.uuid4().hex}.json"
        attempt = {
            "kind": kind,
            "qid": qid,
            "requests": count,
            "reserved_tokens": count * 4097,
            "status": "reserved",
            "started_at": legacy.utc_now(),
        }
        legacy.atomic_json(path, attempt)
        return path, attempt

    def commit(self, path, attempt, records):
        legacy.atomic_json(
            path,
            {
                **attempt,
                "status": "returned",
                "finished_at": legacy.utc_now(),
                "actual_tokens": sum(r["prompt_tokens"] + r["output_tokens"] for r in records),
            },
        )


def one_query(row, binding, binding_hash, frozen_frames, engine, sampling, tokenizer, ids, ledger, kind):
    from backend import make_request, score_requests
    from candidate_frames import EvidenceUnavailable, prepare_candidates

    started = time.monotonic()
    baseline, _ = parent_records(binding, row)
    original = baseline["prediction"]["pred_saliency_scores"]
    selected, selection = dispersed_candidates(original)
    output = {
        "binding_sha256": binding_hash,
        **{k: row[k] for k in ("qid", "vid", "query")},
        "candidate_ids": selected,
        "selection": selection,
        "requests": [],
        "candidate_frames": {},
        "cost": {
            "logical_requests": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "model_frames": 0,
            "inference_seconds": 0,
        },
    }
    try:
        targets, traces, seconds = prepare_candidates(row, baseline, selected)
    except EvidenceUnavailable as error:
        return {**output, "status": "fallback", "error": str(error), "scores": original}
    if traces != frozen_frames:
        raise ValueError("C1 input pixels differ from preflight")
    output.update(candidate_frames=traces, decode_preprocess_seconds=seconds)
    pairs = ordered_pairs(selected)
    requests = [make_request(tokenizer, row["query"], "P_pair", targets[a], targets[b]) for a, b in pairs]
    path, attempt = ledger.reserve(row["qid"], kind, len(requests))
    records, elapsed = score_requests(engine, sampling, requests, ids)
    ledger.commit(path, attempt, records)
    for (a, b), request, record in zip(pairs, requests, records, strict=True):
        record.update(
            a=a,
            b=b,
            prompt=request["prompt"],
            prompt_sha256=legacy.digest(request["prompt"]),
            evidence_sha256=legacy.digest([traces[str(a)], traces[str(b)]]),
            model_frames=len(targets[a]["video"]) + len(targets[b]["video"]),
            native_temporal_padding_frames=len(targets[a]["video"]) % 2 + len(targets[b]["video"]) % 2,
        )
    output.update(
        requests=records,
        completed_at=legacy.utc_now(),
        query_seconds=time.monotonic() - started,
        cost={
            "logical_requests": len(records),
            "input_tokens": sum(r["prompt_tokens"] for r in records),
            "output_tokens": sum(r["output_tokens"] for r in records),
            "model_frames": sum(r["model_frames"] for r in records),
            "inference_seconds": elapsed,
        },
    )
    try:
        for record in records:
            if record.get("error"):
                raise ValueError(record["error"])
            legacy.verify_logprob(record, strict=True)
        scores, ranking = permutation_scores(original, selected, records)
        output.update(status="complete", scores=scores, ranking=ranking)
    except ValueError as error:
        output.update(status="fallback", scores=original, error=str(error))
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    with (run / "worker.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        binding, rows, binding_hash = load_run(run)
        import torch
        from backend import make_engine
        from reze_backend import MODEL_PATH

        torch.set_num_threads(2)
        actual = "GPU-" + str(torch.cuda.get_device_properties(0).uuid).removeprefix("GPU-")
        if torch.cuda.device_count() != 1 or actual != os.environ["REZE_EXPECTED_GPU_UUID"]:
            raise ValueError("Worker GPU UUID differs from allocated device")
        model = json.loads((run / "model_verification.json").read_text())
        if MODEL_PATH != model["audit"]["model_path"]:
            raise ValueError("Backend model path differs from verified weights")
        for name, expected in model["stats"].items():
            stat = (Path(MODEL_PATH) / name).stat()
            if stat.st_size != expected["size"] or stat.st_mtime_ns != expected["mtime_ns"]:
                raise ValueError("Verified model weights changed")
        frames = {r["qid"]: r["candidate_frames"] for r in legacy.read_jsonl(run / "candidate_inputs.jsonl")}
        ledger = Ledger(run, binding)

        def progress(status, done=0, **extra):
            value = {
                "status": status,
                "updated_at": legacy.utc_now(),
                "completed_queries": done,
                "expected_queries": len(rows),
                "gpu_uuid": actual,
                "usage": ledger.totals(),
                **extra,
            }
            legacy.atomic_json(run / "status.json", value)
            print(json.dumps(value), flush=True)

        completed = []
        for row in rows:
            path = run / "records" / f"{row['qid']}.json"
            if path.exists():
                validate_record(json.loads(path.read_text()), row, binding, binding_hash)
                completed.append(row["qid"])
        progress("loading_model", len(completed))
        engine, sampling, tokenizer, ids = make_engine()
        for qid in binding["engineering_qids"]:
            row = next(r for r in rows if r["qid"] == qid)
            path = run / "engineering" / f"{qid}.json"
            if path.exists():
                record = json.loads(path.read_text())
                validate_record(record, row, binding, binding_hash)
            else:
                progress("engineering", len(completed), active_qid=qid)
                record = one_query(
                    row, binding, binding_hash, frames[qid], engine, sampling, tokenizer, ids, ledger, "engineering"
                )
                validate_record(record, row, binding, binding_hash)
                legacy.atomic_json(path, record)
            if record["status"] != "complete":
                legacy.atomic_json(
                    run / "stopped.json",
                    {
                        "status": "engineering_failed",
                        "qid": qid,
                        "error": record.get("error"),
                        "updated_at": legacy.utc_now(),
                    },
                )
                raise ValueError("Engineering failed; no semantic retries")
        legacy.atomic_json(
            run / "gpu_smoke.json",
            {"status": "passed", "binding_sha256": binding_hash, "requests": 100, "accuracy_evaluated": False},
        )
        started = time.monotonic()
        initial = len(completed)
        for row in rows:
            if row["qid"] in completed:
                continue
            progress("running", len(completed), active_qid=row["qid"])
            record = one_query(
                row, binding, binding_hash, frames[row["qid"]], engine, sampling, tokenizer, ids, ledger, "formal"
            )
            validate_record(record, row, binding, binding_hash)
            legacy.atomic_json(run / "records" / f"{row['qid']}.json", record)
            completed.append(row["qid"])
            progress(
                "running",
                len(completed),
                elapsed_seconds=time.monotonic() - started,
                rough_remaining_seconds=(time.monotonic() - started)
                / (len(completed) - initial)
                * (100 - len(completed)),
            )
        receipt = {str(r["qid"]): legacy.sha256(run / "records" / f"{r['qid']}.json") for r in rows}
        legacy.atomic_json(run / "record_receipt.json", receipt)
        legacy.atomic_json(
            run / "inference_complete.json",
            {
                "status": "complete",
                "binding_sha256": binding_hash,
                "queries": len(rows),
                "record_receipt_sha256": legacy.sha256(run / "record_receipt.json"),
            },
        )
        progress("inference_complete", len(completed))


if __name__ == "__main__":
    main()
