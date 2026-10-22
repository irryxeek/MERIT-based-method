"""Sustained-memory regression: exceed the old cache OOM boundary on real videos."""

import json
import os
import subprocess
import time

import numpy as np
from common import ROOT, STORE, atomic_json, read_jsonl, sha256, utc_now
from reze_backend import make_engine, make_request, score_requests
from reze_media import prepare_video


def main():
    import torch

    torch.set_num_threads(2)
    run = STORE / "runs/reze_sustained_v3"
    run.mkdir(parents=True, exist_ok=False)
    atomic_json(
        run / "source_binding.json",
        {
            "sha256": {
                name: sha256(ROOT / name)
                for name in (
                    "scripts/reze_backend.py",
                    "scripts/reze_media.py",
                    "scripts/reze_core.py",
                )
            }
        },
    )
    rows = read_jsonl(ROOT / "manifests/reze_val_inference.jsonl")[:20]
    rows.append(read_jsonl(ROOT / "manifests/reze_smoke_inference.jsonl")[-1])
    started = time.time()
    engine, sampling, tokenizer, ids = make_engine()
    differences, memory, segments = [], [], 0
    for row in rows:
        clips, media = prepare_video(row)
        records, elapsed = score_requests(
            engine, sampling, [make_request(tokenizer, row["query"], c) for c in clips], ids
        )
        original = STORE / "runs/reze_val_v1/records" / f"{row['qid']}.json"
        if not original.exists():
            original = STORE / "runs/reze_smoke_v1/records" / f"{row['qid']}.json"
        if original.exists():
            before = json.loads(original.read_text())
            assert before["media"]["model_rgb_sha256"] == media["model_rgb_sha256"]
            assert [c["prompt_tokens"] for c in before["clips"]] == [c["prompt_tokens"] for c in records]
            old = np.array([c["pyes"] for c in before["clips"]])
            new = np.array([c["pyes"] for c in records])
            differences.append(
                {"qid": row["qid"], "max_absolute_probability_difference": float(np.max(abs(old - new)))}
            )
            # BF16 execution batches can change when encoder eviction changes.
            # Measure numeric drift; do not assert bitwise equality or select scores.
            differences[-1]["strict_allclose_passed"] = bool(np.allclose(old, new, atol=1e-6, rtol=1e-5))
        memory.append(
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "-i",
                    os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0],
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            ).strip()
        )
        segments += len(records)
        atomic_json(
            run / "records" / f"{row['qid']}.json", {"media": media, "clips": records, "generate_seconds": elapsed}
        )
        atomic_json(
            run / "progress.json",
            {
                "queries": len(memory),
                "segments": segments,
                "memory_mib": memory,
                "elapsed_seconds": time.time() - started,
            },
        )
        print(json.dumps({"qid": row["qid"], "queries": len(memory), "memory_mib": memory[-1]}), flush=True)
    atomic_json(
        run / "completion.json",
        {
            "status": "complete",
            "completed_at": utc_now(),
            "queries": len(rows),
            "segments": segments,
            "memory_mib": memory,
            "comparison": differences,
            "total_seconds": time.time() - started,
        },
    )


if __name__ == "__main__":
    main()
