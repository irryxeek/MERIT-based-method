"""Score full videos with frozen REZE, bounded memory and per-batch recovery."""

import argparse
import json
import os
import time
import uuid
from collections import Counter
from pathlib import Path

from shared import (
    atomic_json,
    load_query_chunks,
    load_run,
    qfvs_scores,
    select_summary,
    sha256,
    utc_now,
    validate_chunk,
)

# isort: split
from adaptation import ACTION_MODE, make_qfvs_request
from media import StreamVideo
from pairing import check_paired_trace, paired_receipt
from reze_backend import make_engine, score_requests


def reserve(run, binding_hash, qid, start, count, gpu_uuid, purpose):
    path = run / "request_attempts" / f"{qid}_{start:06d}_{uuid.uuid4().hex}.json"
    record = {
        "binding_sha256": binding_hash,
        "qid": qid,
        "start": start,
        "requests": count,
        "gpu_uuid": gpu_uuid,
        "purpose": purpose,
        "status": "reserved",
        "started_at": utc_now(),
    }
    atomic_json(path, record)
    return path, record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    binding, rows, binding_hash = load_run(run)
    parent = paired_receipt(run, binding)
    prompt_mode = binding.get("prompt_mode", ACTION_MODE)
    if sha256(run / "model_verification.json") != binding["model_verification_sha256"]:
        raise ValueError("Model verification receipt changed")
    model = json.loads((run / "model_verification.json").read_text())
    for name, expected in model["stats"].items():
        stat = (Path(binding["model_path"]) / name).stat()
        if stat.st_size != expected["size"] or stat.st_mtime_ns != expected["mtime_ns"]:
            raise ValueError(f"Model file changed: {name}")
    import torch

    torch.set_num_threads(2)
    if not os.environ["CUDA_VISIBLE_DEVICES"].isdigit() or torch.cuda.device_count() != 1:
        raise ValueError("vLLM 0.12 requires one numeric CUDA device index")
    actual_uuid = "GPU-" + str(torch.cuda.get_device_properties(0).uuid).removeprefix("GPU-")
    if actual_uuid != os.environ["REZE_EXPECTED_GPU_UUID"]:
        raise ValueError(f"Numeric CUDA device resolves to wrong UUID: {actual_uuid}")
    batch_size = binding["batch_size"]
    charged = sum(json.loads(p.read_text())["requests"] for p in (run / "request_attempts").glob("*.json"))
    committed = sum(len(json.loads(p.read_text())["records"]) for p in (run / "chunks").glob("*/*.json"))
    completed = {p.stem for p in (run / "predictions").glob("*.json")}
    initial_committed = committed
    started = time.monotonic()
    atomic_json(
        run / "status.json",
        {
            "status": "loading_model",
            "updated_at": utc_now(),
            "gpu_uuid": actual_uuid,
            "gpu_index": int(os.environ["CUDA_VISIBLE_DEVICES"]),
            "committed_requests": committed,
            "completed_queries": len(completed),
            "expected_queries": len(rows),
        },
    )
    engine, sampling, tokenizer, ids = make_engine(model_path=binding["model_path"])
    if ids != {"Yes": 9454, "No": 2753}:
        raise ValueError("Frozen Yes/No token IDs changed")
    if not (run / "gpu_smoke.json").exists():
        probe_records, seen = [], set()
        for row in rows:
            if row["vid"] in seen:
                continue
            seen.add(row["vid"])
            video = StreamVideo(row)
            for start in (0, len(video.segments) // 2, len(video.segments) - 1):
                if charged + 1 > binding["maximum_charged_requests"]:
                    raise RuntimeError("Cumulative request budget exhausted during GPU smoke")
                clips, trace = video.read(start, 1)
                check_paired_trace(parent, row, start, video.segments[start : start + 1], trace)
                path, attempt = reserve(run, binding_hash, row["qid"], start, 1, actual_uuid, "gpu_smoke")
                charged += 1
                records, elapsed = score_requests(
                    engine, sampling, [make_qfvs_request(tokenizer, row["query"], clips[0], prompt_mode)], ids
                )
                chunk = {
                    "binding_sha256": binding_hash,
                    "qid": row["qid"],
                    "start": start,
                    "segments": video.segments[start : start + 1],
                    "trace": trace,
                    "records": records,
                }
                validate_chunk(
                    json.loads(json.dumps(chunk)), binding_hash, row, start, video.segments[start : start + 1]
                )
                if records[0]["generated_text"].strip() not in {"Yes", "No"} or records[0]["prompt_tokens"] + 1 > 4096:
                    raise ValueError("GPU smoke produced an invalid answer or exceeded context")
                atomic_json(
                    path,
                    {
                        **attempt,
                        "status": "committed",
                        "finished_at": utc_now(),
                        "input_tokens": records[0]["prompt_tokens"],
                        "inference_seconds": elapsed,
                    },
                )
                probe_records.append({**chunk, "inference_seconds": elapsed})
                atomic_json(
                    run / "status.json",
                    {
                        "status": "gpu_smoke_running",
                        "updated_at": utc_now(),
                        "gpu_uuid": actual_uuid,
                        "probes_complete": len(probe_records),
                        "probes_expected": 12,
                    },
                )
            del video
        atomic_json(
            run / "gpu_smoke.json",
            {
                "status": "passed",
                "binding_sha256": binding_hash,
                "finished_at": utc_now(),
                "gpu_uuid": actual_uuid,
                "model_requests": len(probe_records),
                "probes": probe_records,
                "metric_or_label_values_used": False,
            },
        )
    else:
        smoke = json.loads((run / "gpu_smoke.json").read_text())
        if smoke["status"] != "passed" or smoke["binding_sha256"] != binding_hash:
            raise ValueError("GPU smoke belongs to another protocol")
    for row in rows:
        qid = row["qid"]
        video = StreamVideo(row)
        description = video.describe()
        description_path = run / "media" / f"{row['vid']}.json"
        if description_path.exists():
            if json.loads(description_path.read_text()) != json.loads(json.dumps(description)):
                raise ValueError("Decoded timeline changed since checkpoint")
        else:
            atomic_json(description_path, description)
        for start in range(0, len(video.segments), batch_size):
            path = run / "chunks" / qid / f"{start:06d}.json"
            segments = video.segments[start : start + batch_size]
            if path.exists():
                validate_chunk(json.loads(path.read_text()), binding_hash, row, start, segments)
                continue
            if charged + len(segments) > binding["maximum_charged_requests"]:
                raise RuntimeError("Frozen cumulative request budget exhausted")
            clips, trace = video.read(start, batch_size)
            check_paired_trace(parent, row, start, segments, trace, batch_size)
            requests = [make_qfvs_request(tokenizer, row["query"], clip, prompt_mode) for clip in clips]
            attempt_path, attempt = reserve(run, binding_hash, qid, start, len(clips), actual_uuid, "inference")
            charged += len(clips)
            records, elapsed = score_requests(engine, sampling, requests, ids)
            chunk = {
                "binding_sha256": binding_hash,
                "qid": qid,
                "start": start,
                "segments": segments,
                "trace": trace,
                "records": records,
                "inference_seconds": elapsed,
                "attempt_id": attempt_path.stem,
            }
            validate_chunk(json.loads(json.dumps(chunk)), binding_hash, row, start, segments)
            atomic_json(path, chunk)
            atomic_json(
                attempt_path,
                {
                    **attempt,
                    "status": "committed",
                    "finished_at": utc_now(),
                    "input_tokens": sum(r["prompt_tokens"] for r in records),
                    "inference_seconds": elapsed,
                },
            )
            committed += len(clips)
            elapsed_total = time.monotonic() - started
            new_done = committed - initial_committed
            status = {
                "status": "running",
                "updated_at": utc_now(),
                "gpu_uuid": actual_uuid,
                "active_qid": qid,
                "completed_queries": len(completed),
                "expected_queries": len(rows),
                "query_windows_complete": start + len(clips),
                "query_windows_total": len(video.segments),
                "committed_requests": committed,
                "charged_requests": charged,
                "planned_requests": binding["expected_model_requests"],
                "session_seconds": elapsed_total,
                "rough_remaining_seconds": elapsed_total / new_done * (binding["expected_model_requests"] - committed),
            }
            atomic_json(run / "status.json", status)
            print(json.dumps(status), flush=True)
            del clips, requests, records
        records, chunks = load_query_chunks(run, row, binding_hash, video.segments, batch_size)
        scores = qfvs_scores([r["pyes"] for r in records], video.segments, row["duration"], row["n_shots"])
        selected = select_summary(scores, binding["summary_fraction"])
        atomic_json(
            run / "predictions" / f"{qid}.json",
            {
                "qid": qid,
                "vid": row["vid"],
                "query_name": row["query_name"],
                "shot_scores": scores,
                "selected_shot_indices_zero_based": selected,
                "selected_shot_ids_one_based": sorted(i + 1 for i in selected),
            },
        )
        atomic_json(
            run / "query_summaries" / f"{qid}.json",
            {
                "binding_sha256": binding_hash,
                "qid": qid,
                "status": "complete",
                "finished_at": utc_now(),
                "model_requests": len(records),
                "output_tokens": len(records),
                "input_tokens": sum(r["prompt_tokens"] for r in records),
                "frames": sum(c["trace"]["model_input_frames"] for c in chunks),
                "score_modes": dict(Counter(r["score_mode"] for r in records)),
                "generated_answers": dict(Counter(r["generated_text"] for r in records)),
                "decode_preprocess_seconds": sum(c["trace"]["decode_preprocess_seconds"] for c in chunks),
                "inference_seconds": sum(c["inference_seconds"] for c in chunks),
                "n_shot_scores": len(scores),
                "selected_shots": len(selected),
                "zero_score_fraction": sum(s == 0 for s in scores) / len(scores),
                "unique_score_count": len(set(scores)),
            },
        )
        completed.add(qid)
        del video, records, chunks
    atomic_json(
        run / "inference_complete.json",
        {
            "status": "complete",
            "binding_sha256": binding_hash,
            "finished_at": utc_now(),
            "queries": len(completed),
        },
    )


if __name__ == "__main__":
    main()
