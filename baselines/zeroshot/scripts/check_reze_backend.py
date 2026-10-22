"""Real-video engineering smoke: one batched pass and two single-request checks."""

import json
import os
import time

import numpy as np
from common import ROOT, STORE, atomic_json, read_jsonl, utc_now
from reze_backend import make_engine, make_request, score_requests
from reze_media import prepare_video


def main():
    import torch

    torch.set_num_threads(2)
    started = time.time()
    run = STORE / "runs" / os.environ.get("REZE_BACKEND_RUN", "reze_backend_smoke_v1")
    run.mkdir(exist_ok=True, parents=True)
    row = read_jsonl(ROOT / "manifests/reze_smoke_inference.jsonl")[0]
    clips, trace = prepare_video(row, limit_segments=12)
    atomic_json(run / "media.json", trace)
    print(
        json.dumps({"stage": "loading_model", "prepared_clips": len(clips), "resized": trace["resized_height_width"]}),
        flush=True,
    )
    engine, sampling, tokenizer, ids = make_engine()
    requests = [make_request(tokenizer, row["query"], clip) for clip in clips]
    records, elapsed = score_requests(engine, sampling, requests, ids)
    differences = []
    for index in [0, len(clips) - 1]:
        single, _ = score_requests(engine, sampling, [requests[index]], ids)
        differences.append(abs(single[0]["pyes"] - records[index]["pyes"]))
        np.testing.assert_allclose(single[0]["pyes"], records[index]["pyes"], atol=0.02, rtol=0.02)
    report = {
        "time": utc_now(),
        "status": "complete",
        "qid": row["qid"],
        "clips": len(clips),
        "answer_ids": ids,
        "records": records,
        "batch_seconds": elapsed,
        "total_seconds": time.time() - started,
        "batch_single_abs_difference": differences,
        "gpu": torch.cuda.get_device_name(0),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "rendered_prompt": requests[0]["prompt"],
    }
    atomic_json(run / "completion.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in ["records", "rendered_prompt"]}), flush=True)


if __name__ == "__main__":
    main()
