"""Run the frozen original REZE scorer, checkpointing every sixteen windows."""

import argparse
import json
import os
import time
import traceback
import uuid
from collections import Counter
from pathlib import Path

from shared import atomic_json, load_run, load_video_chunks, sha256, svh_scores, utc_now, validate_chunk

# isort: split
# shared loads the pinned vendor directory.
from media import StreamVideo
from reze_backend import make_engine, make_request, score_requests


def gpu_smoke(run, rows, binding, binding_hash, engine, sampling, tokenizer, ids, charged, gpu_uuid):
    """Verify all 21 media probes on the real engine before formal scoring."""
    path = run / "gpu_smoke.json"
    if path.exists():
        result = json.loads(path.read_text())
        if result["binding_sha256"] != binding_hash or result["status"] != "passed":
            raise ValueError("GPU smoke receipt changed")
        return charged
    preflight = json.loads((run / "media_preflight.json").read_text())
    probes_by_vid = {r["vid"]: r["probes"] for r in preflight["videos"]}
    results = []
    for row in rows:
        video = StreamVideo(row)
        for probe in probes_by_vid[row["vid"]]:
            index = probe["index"]
            clips, trace = video.read(index, 1)
            if trace["model_rgb_sha256"] != probe["trace"]["model_rgb_sha256"]:
                raise ValueError("GPU smoke pixels differ from CPU preflight")
            if charged + 1 > binding["maximum_charged_requests"]:
                raise RuntimeError("Frozen cumulative request budget exhausted during smoke")
            attempt_path = run / "request_attempts" / f"smoke_{row['vid']}_{index}_{uuid.uuid4().hex}.json"
            attempt = {"binding_sha256": binding_hash, "requests": 1, "status": "reserved",
                       "kind": "engineering_smoke", "vid": row["vid"], "start": index,
                       "started_at": utc_now(), "gpu_uuid": gpu_uuid}
            atomic_json(attempt_path, attempt)
            charged += 1
            records, elapsed = score_requests(
                engine, sampling, [make_request(tokenizer, row["query"], clips[0])], ids
            )
            chunk = {"binding_sha256": binding_hash, "vid": row["vid"], "start": index,
                     "segments": [list(video.segments[index])], "records": records, "trace": trace}
            validate_chunk(chunk, binding_hash, row, index, video.segments[index:index + 1])
            results.append({**chunk, "inference_seconds": elapsed})
            atomic_json(attempt_path, {**attempt, "status": "committed", "finished_at": utc_now(),
                                      "input_tokens": sum(r["prompt_tokens"] for r in records)})
        del video
    if len(results) != 21:
        raise ValueError("Expected 21 GPU smoke probes")
    atomic_json(path, {"status": "passed", "binding_sha256": binding_hash, "checked_at": utc_now(),
                       "requests": len(results), "probes": results, "accuracy_evaluated": False})
    return charged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run
    binding, rows, binding_hash = load_run(run)
    if sha256(run / "model_verification.json") != binding["model_verification_sha256"]:
        raise ValueError("Model verification receipt changed")
    model = json.loads((run / "model_verification.json").read_text())
    for name, expected in model["stats"].items():
        stat = (Path(binding["model_path"]) / name).stat()
        if stat.st_size != expected["size"] or stat.st_mtime_ns != expected["mtime_ns"]:
            raise ValueError(f"Verified model file changed: {name}")
    import torch

    torch.set_num_threads(2)
    if torch.cuda.device_count() != 1:
        raise ValueError("Worker requires exactly one visible GPU")
    actual_uuid = "GPU-" + str(torch.cuda.get_device_properties(0).uuid).removeprefix("GPU-")
    if actual_uuid != os.environ["REZE_EXPECTED_GPU_UUID"]:
        raise ValueError(f"GPU UUID mismatch: {actual_uuid}")
    batch_size = binding["batch_size"]
    charged = sum(json.loads(p.read_text())["requests"] for p in (run / "request_attempts").glob("*.json"))
    started = time.monotonic()
    complete = {p.stem for p in (run / "predictions").glob("*.json")}
    committed = sum(len(json.loads(p.read_text())["records"]) for p in (run / "chunks").glob("*/*.json"))
    initial_committed = committed
    engine = sampling = tokenizer = ids = None
    atomic_json(
        run / "status.json",
        {
            "status": "loading_model",
            "updated_at": utc_now(),
            "completed_videos": len(complete),
            "committed_requests": committed,
            "gpu_uuid": actual_uuid,
        },
    )
    engine, sampling, tokenizer, ids = make_engine(model_path=binding["model_path"])
    if ids != {"Yes": 9454, "No": 2753}:
        raise ValueError("Answer token identities changed")
    if binding.get("require_gpu_smoke"):
        charged = gpu_smoke(run, rows, binding, binding_hash, engine, sampling, tokenizer, ids, charged, actual_uuid)
    for row in rows:
        vid = row["vid"]
        video = StreamVideo(row)
        description = video.describe()
        description_path = run / "media" / f"{vid}.json"
        if description_path.exists():
            if json.loads(description_path.read_text()) != json.loads(json.dumps(description)):
                raise ValueError("Decoded timeline changed since checkpoint")
        else:
            atomic_json(description_path, description)
        for start in range(0, len(video.segments), batch_size):
            path = run / "chunks" / vid / f"{start:06d}.json"
            segments = video.segments[start : start + batch_size]
            if path.exists():
                validate_chunk(json.loads(path.read_text()), binding_hash, row, start, segments)
                continue
            if charged + len(segments) > binding["maximum_charged_requests"]:
                raise RuntimeError("Frozen cumulative request budget exhausted")
            clips, trace = video.read(start, batch_size)
            requests = [make_request(tokenizer, row["query"], clip) for clip in clips]
            attempt_id = f"{vid}_{start:06d}_{uuid.uuid4().hex}"
            attempt_path = run / "request_attempts" / f"{attempt_id}.json"
            attempt = {
                "binding_sha256": binding_hash,
                "vid": vid,
                "start": start,
                "requests": len(clips),
                "started_at": utc_now(),
                "status": "reserved",
                "gpu_uuid": actual_uuid,
            }
            atomic_json(attempt_path, attempt)
            charged += len(clips)
            records, elapsed = score_requests(engine, sampling, requests, ids)
            chunk = {
                "binding_sha256": binding_hash,
                "vid": vid,
                "start": start,
                "segments": segments,
                "trace": trace,
                "records": records,
                "inference_seconds": elapsed,
                "attempt_id": attempt_id,
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
                },
            )
            committed += len(clips)
            elapsed_total = time.monotonic() - started
            new_done = committed - initial_committed
            remaining = binding["expected_model_requests_upper_bound"] - committed
            status = {
                "status": "running",
                "updated_at": utc_now(),
                "gpu_uuid": actual_uuid,
                "active_video": vid,
                "completed_videos": len(complete),
                "expected_videos": len(rows),
                "video_windows_complete": min(start + batch_size, len(video.segments)),
                "video_windows_total": len(video.segments),
                "committed_requests": committed,
                "charged_requests": charged,
                "planned_requests_upper_bound": binding["expected_model_requests_upper_bound"],
                "session_seconds": elapsed_total,
                "rough_remaining_seconds": elapsed_total / new_done * remaining if new_done else None,
            }
            atomic_json(run / "status.json", status)
            print(json.dumps(status), flush=True)
            del clips, requests, records
        records, chunks = load_video_chunks(run, row, binding_hash, video.segments, batch_size)
        scores = svh_scores([r["pyes"] for r in records], video.segments, row["duration"], row["n_clips"])
        prediction = {"vid": vid, "pred_saliency_scores": scores}
        atomic_json(run / "predictions" / f"{vid}.json", prediction)
        atomic_json(
            run / "video_summaries" / f"{vid}.json",
            {
                "binding_sha256": binding_hash,
                "vid": vid,
                "status": "complete",
                "model_requests": len(records),
                "frames": sum(c["trace"]["model_input_frames"] for c in chunks),
                "input_tokens": sum(r["prompt_tokens"] for r in records),
                "output_tokens": len(records),
                "score_modes": dict(Counter(r["score_mode"] for r in records)),
                "generated_answers": dict(Counter(r["generated_text"] for r in records)),
                "decode_preprocess_seconds": sum(c["trace"]["decode_preprocess_seconds"] for c in chunks),
                "inference_seconds": sum(c["inference_seconds"] for c in chunks),
                "n_scores": len(scores),
                "finished_at": utc_now(),
            },
        )
        complete.add(vid)
        del video, records, chunks
    atomic_json(
        run / "inference_complete.json",
        {"status": "complete", "binding_sha256": binding_hash, "videos": len(complete), "finished_at": utc_now()},
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
