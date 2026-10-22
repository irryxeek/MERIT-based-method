"""Label-free capacity/throughput probes; never used as accuracy predictions."""

import argparse
import json
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from merit_qvh import prompts
from merit_qvh.client import text_part
from merit_qvh.core import digest, read_jsonl, validate_decision, write_json
from merit_qvh.media import prepare_frames
from merit_qvh.pipeline import visual_content

from .client import Client
from .pipeline import SharedEmbedder, dense_frames, hd_content, validate_hd
from .prompts import SYSTEM


def gpu_sample():
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return {
        int(parts[0]): {"memory_mib": int(parts[1]), "utilization": int(parts[2])}
        for line in output.splitlines()
        if (parts := line.split(",")) and int(parts[0]) in (4, 5)
    }


def benchmark(config, directory, endpoint):
    directory.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(Path(config["hd"]["data_root"]) / "manifests/explore.jsonl")[:10]
    embedder = SharedEmbedder(config["embedding"], config["artifact_root"])
    embedder.encode([rows[0]["query"]])  # Resident during every capacity measurement.
    jobs, dimensions = [], []
    from PIL import Image
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        config["vlm"]["weight_path"],
        local_files_only=True,
        size={"shortest_edge": 65536, "longest_edge": 1048576},
    )
    for row in rows:
        info, frames = dense_frames(row, directory)
        ids = list(range(min(24, len(frames))))
        jobs.append(
            (
                f"hd-{row['qid']}",
                "hd_initialize",
                hd_content(row, info, frames, ids),
                2048,
                lambda value, expected=ids: validate_hd(value, expected),
                SYSTEM,
            )
        )
        with Image.open(frames[0]["path"]) as image:
            grid = processor.image_processor(images=[image], return_tensors="pt")
            dimensions.append(
                {
                    "qid": row["qid"],
                    "jpeg_size": list(image.size),
                    "image_grid_thw": grid["image_grid_thw"].tolist(),
                }
            )
    # Controlled-size image tests verify the server-side cap does not reduce 672px to 448px.
    synthetic = Image.new("RGB", (672, 448))
    grid_audit = {}
    for cap in (200704, 1048576):
        audit_processor = AutoProcessor.from_pretrained(
            config["vlm"]["weight_path"],
            local_files_only=True,
            size={"shortest_edge": 65536, "longest_edge": cap},
        )
        g = audit_processor.image_processor(images=[synthetic], return_tensors="pt")
        grid_audit[str(cap)] = g["image_grid_thw"].tolist()
    for row in rows[:2]:
        media = prepare_frames(row, config["memory"], config["artifact_root"])
        clips = media["clips"][:9]
        jobs.append(
            (
                f"capacity-{row['qid']}",
                "capacity_90_images",
                [
                    text_part(
                        prompts.DECIDE
                        + "\nORIGINAL QUERY: "
                        + row["query"]
                        + "\nRETRIEVED CAPTIONS: []\nFILTERED EVIDENCE: []"
                    )
                ]
                + visual_content(clips),
                256,
                validate_decision,
                None,
            )
        )
    # Warm every payload so prefix/image-cache reuse is comparable at each concurrency.
    warm = Client(endpoint, config["vlm"], directory / "warmup", "hd-v2-warmup")
    for _, stage, content, tokens, validator, system in jobs:
        warm.complete(stage, content, tokens, validator, system=system)
    report = {
        "qids": [r["qid"] for r in rows],
        "requests_per_concurrency": len(jobs),
        "scope": "10 real initial HD batches + 2 capacity-only MERIT image requests; not full-query speedup",
        "cache_condition": "same payloads warmed once; request cache distinct for each concurrency",
        "dimension_audit": dimensions,
        "synthetic_grid_audit": grid_audit,
        "embedding_resident": True,
        "runs": {},
    }
    raw_outputs = {}
    for workers in (1, 2, 4):
        samples, stop = [], threading.Event()

        def monitor(stop=stop, samples=samples):
            while not stop.is_set():
                samples.append(gpu_sample())
                stop.wait(1)

        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        started = time.monotonic()

        def request(job, workers=workers):
            name, stage, content, tokens, validator, system = job
            client = Client(
                endpoint, config["vlm"], directory / f"requests_c{workers}", f"hd-bench-c{workers}"
            )
            tick = time.monotonic()
            try:
                result = client.complete(stage, content, tokens, validator, system=system)
                return {
                    "name": name,
                    "status": "ok",
                    "seconds": time.monotonic() - tick,
                    "parsed": result,
                    "digest": digest(result),
                    "events": client.events,
                }
            except Exception as error:
                return {"name": name, "status": "failed", "error": str(error), "events": client.events}

        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(request, jobs))
        elapsed = time.monotonic() - started
        stop.set()
        thread.join()
        raw_outputs[workers] = {r["name"]: r.get("parsed") for r in results}
        report["runs"][str(workers)] = {
            "seconds": elapsed,
            "logical_requests_per_second": len(jobs) / elapsed,
            "failures": sum(r["status"] != "ok" for r in results),
            "results": results,
            "gpu_peak_mib": {str(g): max(s[g]["memory_mib"] for s in samples) for g in (4, 5)},
        }
        write_json(directory / "benchmark.json", report)
        print(
            json.dumps(
                {"workers": workers, "seconds": elapsed, "failures": report["runs"][str(workers)]["failures"]}
            ),
            flush=True,
        )
    eligible = [w for w in (1, 2, 4) if report["runs"][str(w)]["failures"] == 0]
    if not eligible:
        raise RuntimeError("No concurrency level passed capacity/format checks")
    # Use the smallest concurrency within 5% of best observed throughput.
    best = max(report["runs"][str(w)]["logical_requests_per_second"] for w in eligible)
    report["selected_workers"] = min(
        w for w in eligible if report["runs"][str(w)]["logical_requests_per_second"] >= 0.95 * best
    )
    report["selection_rule"] = "zero failures; lowest concurrency within 5% of best requests/s; no GT"
    report["parsed_output_changes_vs_serial"] = {
        str(w): sum(raw_outputs[w][name] != value for name, value in raw_outputs[1].items()) for w in (2, 4)
    }
    report["hd_score_vector_changes_vs_serial"] = {
        str(w): sum(
            raw_outputs[w][name].get("clip_scores") != value.get("clip_scores")
            for name, value in raw_outputs[1].items()
            if name.startswith("hd-") and value is not None and raw_outputs[w][name] is not None
        )
        for w in (2, 4)
    }
    write_json(directory / "benchmark.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8037/v1")
    args = parser.parse_args()
    benchmark(json.loads(args.config.read_text()), args.directory, args.endpoint)
