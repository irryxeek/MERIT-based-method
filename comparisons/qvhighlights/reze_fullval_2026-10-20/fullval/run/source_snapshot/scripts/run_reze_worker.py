"""One persistent native-video Qwen worker with atomic per-query checkpoints."""

import argparse
import fcntl
import json
import os
import platform
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from pathlib import Path

from common import atomic_json, sha256, utc_now
from reze_backend import make_engine, make_request, score_requests
from reze_core import highlight_scores
from reze_media import prepare_video
from reze_run_io import load_bound_run, read_record, validate_record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--worker", type=int, required=True)
    parser.add_argument("--attempt", type=int, required=True)
    args = parser.parse_args()
    import torch

    torch.set_num_threads(2)
    run = args.run.resolve()
    folder = run / "workers" / str(args.worker)
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder / "worker.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    binding, all_rows = load_bound_run(run)
    binding_hash = sha256(run / "binding.json")
    rows = all_rows[args.worker :: len(binding["gpus"])]
    started = time.time()
    state = {
        "worker": args.worker,
        "attempt": args.attempt,
        "pid": os.getpid(),
        "started_at": utc_now(),
        "status": "loading",
        "expected_queries": len(rows),
        "completed_queries": 0,
        "clips": 0,
        "frames": 0,
        "prompt_tokens": 0,
        "generate_seconds_sum": 0,
        "decode_preprocess_seconds_sum": 0,
        "score_modes": {},
        "generated_tokens": {},
    }

    def publish():
        state.update(updated_at=utc_now(), elapsed_seconds=time.time() - started)
        atomic_json(folder / "progress.json", state)

    def count_record(record):
        state["completed_queries"] += 1
        state["clips"] += len(record["clips"])
        state["frames"] += record["media"]["model_input_frames"]
        state["prompt_tokens"] += sum(c["prompt_tokens"] for c in record["clips"])
        state["generate_seconds_sum"] += record["generate_seconds"]
        state["decode_preprocess_seconds_sum"] += record["media"]["decode_preprocess_seconds"]
        state["score_modes"] = dict(Counter(state["score_modes"]) + Counter(c["score_mode"] for c in record["clips"]))
        state["generated_tokens"] = dict(
            Counter(state["generated_tokens"]) + Counter(c["generated_text"] for c in record["clips"])
        )

    try:
        pending = []
        for row in rows:
            if (run / "records" / f"{row['qid']}.json").exists():
                count_record(read_record(run, row, binding, binding_hash))
            else:
                pending.append(row)
        publish()
        if pending:
            engine, sampling, tokenizer, ids = make_engine()
            properties = torch.cuda.get_device_properties(0)
            observed_uuid = "GPU-" + str(properties.uuid).removeprefix("GPU-")
            expected_uuid = binding["gpus"][args.worker]["uuid"]
            if observed_uuid != expected_uuid:
                raise ValueError(f"GPU mapping mismatch: {observed_uuid} != {expected_uuid}")
            atomic_json(
                folder / f"runtime_attempt_{args.attempt}.json",
                {
                    "python": platform.python_version(),
                    "packages": {p: version(p) for p in ["torch", "transformers", "vllm", "qwen-vl-utils", "decord"]},
                    "gpu_name": properties.name,
                    "gpu_uuid": observed_uuid,
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "answer_token_ids": ids,
                    "model_loading_seconds": time.time() - started,
                    "sampling_params": str(sampling),
                },
            )
            # Exactly one future video is prefetched; frames are never shared between prompts.
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(prepare_video, pending[0])
                for index, row in enumerate(pending):
                    query_start = time.time()
                    state.update(status="decoding", current_qid=row["qid"])
                    publish()
                    clips, media = future.result()
                    if index + 1 < len(pending):
                        future = pool.submit(prepare_video, pending[index + 1])
                    if media["source_sha256"] != binding["video_sha256"][row["vid"]]:
                        raise ValueError("Decoded video differs from frozen baseline media")
                    requests = [make_request(tokenizer, row["query"], clip) for clip in clips]
                    state.update(status="scoring", current_clip_requests=len(requests))
                    publish()
                    atomic_json(
                        folder / "inflight.json",
                        {
                            "qid": row["qid"],
                            "attempt": args.attempt,
                            "submitted_at": utc_now(),
                            "requests": len(requests),
                        },
                    )
                    records, elapsed = score_requests(engine, sampling, requests, ids)
                    scores = highlight_scores([c["pyes"] for c in records], media["segments"], row["duration"])
                    record = {
                        "status": "complete",
                        "completed_at": utc_now(),
                        "binding_sha256": binding_hash,
                        "worker": args.worker,
                        "attempt": args.attempt,
                        "prediction": {**{k: row[k] for k in ["qid", "vid", "query"]}, "pred_saliency_scores": scores},
                        "media": media,
                        "clips": records,
                        "generate_seconds": elapsed,
                        "query_wall_seconds": time.time() - query_start,
                    }
                    validate_record(record, row, binding, binding_hash)
                    atomic_json(run / "records" / f"{row['qid']}.json", record)
                    count_record(record)
                    state.update(status="running", last_completed_qid=row["qid"])
                    publish()
                    print(
                        json.dumps({k: state[k] for k in ["worker", "completed_queries", "clips", "elapsed_seconds"]}),
                        flush=True,
                    )
                    del clips, media, requests, records, record
        state.update(status="complete", completed_at=utc_now())
        publish()
        atomic_json(folder / "completion.json", state)
    except BaseException as exc:
        state.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
        publish()
        atomic_json(folder / f"failure_attempt_{args.attempt}.json", state)
        raise


if __name__ == "__main__":
    main()
