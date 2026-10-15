"""Validate GPU co-residency, run the fixed smoke gate, then the full validation split."""

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

from merit_qvh.cli import frozen_run
from merit_qvh.client import CachedClient, text_part
from merit_qvh.core import read_jsonl, write_json
from merit_qvh.embedding import Embedder
from merit_qvh.pipeline import visual_content


def capacity_check(config, protocol_id, root, endpoint):
    client = CachedClient(endpoint, config["vlm"], root / "requests", protocol_id)
    client.check_model()
    embedder = Embedder(config["embedding"], config["artifact_root"])
    vectors = embedder.encode(["A person opens a box."])
    # A cache hit must not bypass the simultaneous-residency check.
    if embedder.model is None:
        vectors = embedder.encode([f"GPU capacity check {time.time_ns()}"])
    row = read_jsonl(root / "manifests/smoke.jsonl")[0]
    memory = json.loads((root / "memories" / (row["vid"] + ".json")).read_text())
    clips = memory["clips"][:9]
    image_count = sum(len(clip["frames"]) for clip in clips)
    if image_count != 90:
        raise ValueError("The fixed capacity check requires exactly 90 frames")
    answer = client.complete(
        "capacity_preflight",
        [text_part("Reply with only OK. This checks video frame input capacity.")] + visual_content(clips),
        8,
    )
    write_json(
        root / "checks/gpu_capacity.json",
        {
            "status": "passed",
            "image_count": image_count,
            "embedding_shape": list(vectors.shape),
            "embedding_device": config["embedding"]["device"],
            "answer": answer,
            "requests": client.events,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--capacity-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    protocol_id, root = frozen_run(config)
    if args.capacity_only:
        capacity_check(config, protocol_id, root, args.endpoint)
        return
    state = {
        "protocol_id": protocol_id,
        "pid": os.getpid(),
        "started_at": time.time(),
        "endpoint": args.endpoint,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }

    def status(stage, **fields):
        state.update(stage=stage, updated_at=time.time(), **fields)
        write_json(root / "execution_status.json", state)
        print(json.dumps(state), flush=True)

    with (root / "orchestrator.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = root / "execution_status.json"
        if previous.exists():
            write_json(
                root / "runtime" / f"execution_previous_{time.time_ns()}.json",
                json.loads(previous.read_text()),
            )
        try:
            status("capacity_preflight")
            common = ["--config", str(args.config.resolve()), "--endpoint", args.endpoint]
            # Separate processes release all CUDA contexts and model reference cycles
            # between stages, including allocations made by the capacity check.
            subprocess.run(
                [sys.executable, "-u", str(Path(__file__).resolve()), *common, "--capacity-only"], check=True
            )
            status("smoke")
            command = [sys.executable, "-u", "-m", "merit_qvh.cli", "run", *common]
            subprocess.run([*command, "--split", "smoke"], check=True)
            smoke = json.loads((root / "smoke_report.json").read_text())
            if smoke["failed_qids"]:
                raise RuntimeError("Smoke gate failed; validation inference was not started")
            status("validation", smoke_queries=smoke["queries"], smoke_failures=0)
            subprocess.run([*command, "--split", "val"], check=True)
            report = json.loads((root / "val_report.json").read_text())
            status(
                "complete",
                validation_queries=report["queries"],
                validation_failures=len(report["failed_qids"]),
                metrics=report["official_metrics"]["brief"],
            )
        except Exception as error:
            status("failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
            raise


if __name__ == "__main__":
    main()
