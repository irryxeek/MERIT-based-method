"""Check real QFVS beginning/middle/end frames and lengths before GPU dispatch."""

import argparse
import time
from pathlib import Path

import numpy as np
from media import StreamVideo
from shared import atomic_json, load_run, qfvs_scores, utc_now


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    binding, rows, binding_hash = load_run(args.run)
    import torch
    from reze_backend import make_request
    from transformers import AutoTokenizer
    from vllm.config import ModelConfig
    from vllm.multimodal import MULTIMODAL_REGISTRY

    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(binding["model_path"], local_files_only=True)
    config = ModelConfig(
        model=binding["model_path"],
        dtype="bfloat16",
        max_model_len=4096,
        limit_mm_per_prompt={"image": 0, "video": {"count": 1, "num_frames": 6, "width": 1024, "height": 768}},
        mm_processor_kwargs={
            "do_sample_frames": False,
            "do_resize": False,
            "size": {"shortest_edge": 131072, "longest_edge": 6 * 786432},
        },
        mm_processor_cache_gb=0,
    )
    processor = MULTIMODAL_REGISTRY.create_processor(config, tokenizer=tokenizer)
    result = []
    seen = set()
    for row in sorted(rows, key=lambda r: -len(r["query"])):
        if row["vid"] in seen:
            continue
        seen.add(row["vid"])
        then = time.monotonic()
        video = StreamVideo(row)
        probes = []
        for index in sorted({0, len(video.segments) // 2, len(video.segments) - 1}):
            clips, trace = video.read(index, 1)
            if len(clips) != 1 or clips[0]["video"].dtype != np.uint8:
                raise ValueError("Invalid video tensor")
            if not (2 <= len(clips[0]["video"]) <= 6):
                raise ValueError("Unexpected frame count")
            request = make_request(tokenizer, row["query"], clips[0])
            processed = processor.apply(request["prompt"], request["multi_modal_data"], request["mm_processor_kwargs"])
            placeholders = processed["mm_placeholders"]["video"]
            prompt_tokens = len(processed["prompt_token_ids"])
            if len(placeholders) != 1 or prompt_tokens + 1 > 4096:
                raise ValueError("Native video processing exceeds the frozen model context")
            probes.append(
                {
                    "index": index,
                    "segment": video.segments[index],
                    "trace": trace,
                    "prompt_tokens": prompt_tokens,
                    "video_tokens": int(placeholders[0].is_embed.sum()),
                }
            )
        zeros = qfvs_scores([0.0] * len(video.segments), video.segments, row["duration"], row["n_shots"])
        result.append(
            {
                "vid": row["vid"],
                "windows": len(video.segments),
                "media": video.describe(),
                "probe_seconds": time.monotonic() - then,
                "checked_score_length": len(zeros),
                "probes": probes,
            }
        )
        print(f"PASS {row['vid']}: {len(video.segments)} windows, {len(zeros)} native shots", flush=True)
        del video
    atomic_json(
        args.run / "media_preflight.json",
        {
            "status": "passed",
            "binding_sha256": binding_hash,
            "updated_at": utc_now(),
            "gpu_used": False,
            "model_called": False,
            "label_values_read": False,
            "videos": result,
        },
    )


if __name__ == "__main__":
    main()
