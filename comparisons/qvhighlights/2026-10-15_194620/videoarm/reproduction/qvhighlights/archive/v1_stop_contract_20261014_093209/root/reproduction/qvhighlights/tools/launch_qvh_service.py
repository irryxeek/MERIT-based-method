"""Launch one owned BF16/TP2 service only on two process-free GPUs."""

import _bootstrap  # noqa: F401 -- establish the isolated implementation import
import argparse
import csv
import io
import json
import os
import socket
import subprocess
import time
from pathlib import Path

from videoarm.qvhighlights.common import ARTIFACTS, atomic_json, file_hash

SERVER_ENV = Path(
    "/data/linyifei/videoarm-repro/svhighlights/tf_selector/qwen35_9b/env-vllm017"
)
UNIT = "videoarm-qvh-model.service"


def inventory():
    result = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.used",
            "--format=csv,nounits",
        ],
        text=True,
    )
    gpu = list(csv.DictReader(io.StringIO(result), skipinitialspace=True))
    result = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,nounits",
        ],
        text=True,
    )
    processes = list(csv.DictReader(io.StringIO(result), skipinitialspace=True))
    busy = {p["gpu_uuid"] for p in processes}
    free = [
        g for g in gpu if g["uuid"] not in busy and int(g["memory.used [MiB]"]) <= 200
    ]
    return {
        "time": time.time(),
        "gpus": gpu,
        "compute_processes": processes,
        "free_indices": [int(g["index"]) for g in free],
        "blocked": len(free) < 2,
        "policy": "requires two GPUs with <=200 MiB used AND no compute processes; never terminate others",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--launch", action="store_true")
    a = ap.parse_args()
    audit = inventory()
    atomic_json(ARTIFACTS / "checks/gpu_readiness.json", audit)
    if audit["blocked"]:
        print(
            json.dumps(
                {
                    "status": "blocked_gpu_tp2",
                    "free_indices": audit["free_indices"],
                    "needed": 2,
                }
            )
        )
        return
    if not a.launch:
        print(json.dumps({"status": "ready", "free_indices": audit["free_indices"]}))
        return
    config = json.loads((ARTIFACTS / "configs/default_v1.json").read_text())
    config_path = Path(config["model_path"])
    for name, data in json.loads(
        (ARTIFACTS / "checks/model_hash_verification.json").read_text()
    ).items():
        if file_hash(config_path / name) != data["sha256"]:
            raise ValueError("local model weight drift")
    revisions = {
        p.read_text().splitlines()[0]
        for p in (config_path / ".cache/huggingface/download").glob("*.metadata")
    }
    if revisions != {config["model_revision"]}:
        raise ValueError("model metadata revision does not match pinned revision")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 8026))
    chosen = audit["free_indices"][:2]
    now = inventory()
    if not set(chosen) <= set(now["free_indices"]):
        raise RuntimeError("GPU occupancy changed immediately before launch")
    # Independent service name, port, and cache. No SVHighlights service mutation.
    cache = (
        Path(os.getenv("QVH_DATA_ROOT", str(ARTIFACTS.parents[1] / "qvh_data")))
        / "vllm_cache"
    )
    command = [
        "systemd-run",
        "--user",
        "--collect",
        "--unit=" + UNIT,
        "--property=Restart=no",
        "--property=WorkingDirectory=" + str(ARTIFACTS.parents[1]),
        "--setenv=CUDA_VISIBLE_DEVICES=" + ",".join(map(str, chosen)),
        "--setenv=CUDA_DEVICE_ORDER=PCI_BUS_ID",
        "--setenv=HF_HUB_OFFLINE=1",
        "--setenv=TRANSFORMERS_OFFLINE=1",
        "--setenv=VLLM_NO_USAGE_STATS=1",
        "--setenv=DO_NOT_TRACK=1",
        "--setenv=VLLM_CACHE_ROOT=" + str(cache),
        str(SERVER_ENV / "bin/vllm"),
        "serve",
        str(config_path),
        "--host",
        "127.0.0.1",
        "--port",
        "8026",
        "--served-model-name",
        config["model"],
        "--dtype",
        "bfloat16",
        "--tensor-parallel-size",
        "2",
        "--max-model-len",
        "32768",
        "--max-num-seqs",
        "1",
        "--limit-mm-per-prompt",
        '{"image":24,"video":0}',
        "--mm-processor-kwargs",
        '{"max_pixels":1048576}',
        "--gpu-memory-utilization",
        "0.8",
        "--enforce-eager",
        "--skip-mm-profiling",
        "--no-enable-log-requests",
    ]
    subprocess.run(command, check=True)
    atomic_json(
        ARTIFACTS / "checks/service_launch.json",
        {
            "command": command,
            "gpu_indices": chosen,
            "gpu_uuids": [
                g["uuid"] for g in audit["gpus"] if int(g["index"]) in chosen
            ],
            "time": time.time(),
        },
    )
    # Readiness validation is a separate invocation to avoid a long blocking tool call.
    print(
        json.dumps(
            {"status": "starting", "unit": UNIT, "next": "verify_qvh_service.py"}
        )
    )


if __name__ == "__main__":
    main()
