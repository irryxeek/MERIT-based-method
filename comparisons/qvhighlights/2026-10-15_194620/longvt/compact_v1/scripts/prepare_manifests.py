#!/usr/bin/env python3
"""Create label-free candidate manifests and separate QVHighlights ground truth."""

import argparse
import json
import random
from pathlib import Path


def split_vid(vid: str) -> tuple[str, float, float]:
    youtube_id, start, end = vid.rsplit("_", 2)
    return youtube_id, float(start), float(end)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    rows = [json.loads(line) for line in Path(args.annotations).read_text().splitlines() if line.strip()]
    rng = random.Random(config["seed"])
    rng.shuffle(rows)

    # One query per video prevents repeated-video weighting and makes smoke/pilot
    # video isolation explicit. Selection uses no label fields.
    selected = []
    seen_vids = set()
    target = config["smoke_size"] + config["pilot_size"] + config["download_candidate_reserve"]
    for row in rows:
        if row["vid"] in seen_vids:
            continue
        seen_vids.add(row["vid"])
        youtube_id, source_start, source_end = split_vid(row["vid"])
        duration = float(row["duration"])
        selected.append({
            "candidate_rank": len(selected),
            "qid": row["qid"],
            "vid": row["vid"],
            "query": row["query"],
            "duration": duration,
            "num_clips": int(duration / config["clip_seconds"]),
            "youtube_id": youtube_id,
            "source_start_seconds": source_start,
            "source_end_seconds": source_end,
        })
        if len(selected) == target:
            break

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "candidate_queue.json").write_text(json.dumps(selected, indent=2) + "\n")
    gt = [{
        "qid": row["qid"], "vid": row["vid"], "query": row["query"],
        "duration": row["duration"], "relevant_windows": row["relevant_windows"],
        "relevant_clip_ids": row["relevant_clip_ids"], "saliency_scores": row["saliency_scores"],
    } for row in rows]
    with (output_dir / "validation_ground_truth.jsonl").open("w") as handle:
        for row in gt:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps({"candidates": len(selected), "ground_truth": len(gt), "seed": config["seed"]}))


if __name__ == "__main__":
    main()
