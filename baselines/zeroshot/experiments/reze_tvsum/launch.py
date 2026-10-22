"""Create a new immutable run plan and persistent user-systemd queue."""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from dataset import UMT_DATA
from prepare import selection, source_snapshot
from shared import atomic_json, segments_for, sha256, utc_now

GPU_CANDIDATES = [uuid.strip() for uuid in os.environ.get("REPRO_GPU_UUIDS", "").split(",") if uuid.strip()]


def service(root, prefix, suffix):
    complete = [root / "complete.json"]
    if suffix.startswith("shard"):
        index = int(suffix[-1])
        complete += [root / f"shard_{index}/complete.json", root / f"shard_{index}/stopped.json"]
        command = f"{root}/source_snapshot/run_shard.py --run {root} --shard-index {index}"
    else:
        command = f"{root}/source_snapshot/aggregate_shards.py --run {root}"
    conditions = "\n".join(f"ConditionPathExists=!{p}" for p in complete)
    return f"""[Unit]
Description=REZE TVSum full50 {suffix}
ConditionPathExists={root}/plan.json
{conditions}

[Service]
Type=oneshot
WorkingDirectory={root}
ExecStart={sys.executable} -u {command}
Environment=CUDA_VISIBLE_DEVICES=
EnvironmentFile={root}/publication.env
Environment=HF_HUB_OFFLINE=1
Environment=TRANSFORMERS_OFFLINE=1
Environment=HF_HUB_DISABLE_TELEMETRY=1
Environment=VLLM_NO_USAGE_STATS=1
Environment=WANDB_MODE=disabled
Environment=OMP_NUM_THREADS=2
Environment=MKL_NUM_THREADS=2
Environment=TOKENIZERS_PARALLELISM=false
Environment=PYTHONUNBUFFERED=1
TimeoutStartSec=0
TimeoutStopSec=30
KillMode=control-group
Nice=5
StandardOutput=append:{root}/orchestrator_{suffix}.log
StandardError=append:{root}/orchestrator_{suffix}.log
"""


def timer(name):
    return f"""[Unit]
Description=Resume {name} automatically

[Timer]
OnBootSec=30
OnUnitInactiveSec=60
AccuracySec=1
Unit={name}.service

[Install]
WantedBy=timers.target
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--unit-prefix", default="reze-tvsum-full50-v1")
    args = parser.parse_args()
    if not GPU_CANDIDATES or len(set(GPU_CANDIDATES)) != len(GPU_CANDIDATES):
        parser.error("Set REPRO_GPU_UUIDS to unique allocated NVIDIA GPU UUIDs")
    root = args.run.resolve()
    prefix = args.unit_prefix
    if root.exists():
        raise FileExistsError("Preserve old runs; select a new directory")
    if not re.fullmatch(r"[a-z0-9-]+", prefix) or not re.fullmatch(r"/[A-Za-z0-9_./-]+", str(root)):
        raise ValueError("Use a plain filesystem path and unit prefix")
    unit_dir = Path.home() / ".config/systemd/user"
    names = [f"{prefix}-{suffix}" for suffix in ("shard0", "shard1", "aggregate")]
    if any((unit_dir / (name + extension)).exists() for name in names for extension in (".service", ".timer")):
        raise FileExistsError("Preserve existing systemd units")
    audit = json.loads((UMT_DATA / "dataset_audit.json").read_text())
    if audit["status"] != "passed":
        raise ValueError("Dataset audit required before launching")
    root.mkdir(parents=True)
    # A systemd user manager does not inherit the terminal's exported paths.
    environment = {
        key: value
        for key, value in os.environ.items()
        if key
        in {
            "MERIT_ROOT",
            "ZEROSHOT_ROOT",
            "REPRO_DATA_ROOT",
            "MERIT_ARTIFACT_ROOT",
            "ZEROSHOT_ARTIFACT_ROOT",
            "REZE_MODEL_PATH",
            "QVH_EVALUATOR_ROOT",
            "MRHISUM_SOURCE_ROOT",
            "QFVS_DATA_ROOT",
            "REPRO_GPU_UUIDS",
            "PYTHONPATH",
        }
    }
    (root / "publication.env").write_text(
        "".join(key + "=" + json.dumps(value) + "\n" for key, value in sorted(environment.items()))
    )
    sources = source_snapshot(root / "source_snapshot")
    ids = selection(audit, 0, 1)
    shards = []
    for index in (0, 1):
        selected = selection(audit, index, 2)
        shards.append(
            {
                "index": index,
                "directory": f"shard_{index}",
                "video_ids": selected,
                "gpu_candidates": GPU_CANDIDATES,
                "planned_requests": sum(len(segments_for(audit["videos"][v]["duration"])) for v in selected),
            }
        )
    plan = {
        "dataset": "TVSum",
        "created_at": utc_now(),
        "selected_ids": ids,
        "umt_val_ids": audit["umt_val_ids"],
        "shards": shards,
        "source_sha256": sources,
        "input_sha256": {
            str(UMT_DATA / n): sha256(UMT_DATA / n)
            for n in ("dataset_audit.json", "tvsum_anno.json", "download_receipt.json")
        },
        "max_concurrent_gpus": 2,
        "primary_scope": "umt_val10",
        "secondary_scope": "all50_zero_shot",
        "selection_uses_labels": False,
        "training": False,
        "unit_prefix": prefix,
    }
    atomic_json(root / "plan.json", plan)
    atomic_json(
        root / "status.json",
        {"status": "queued_for_preparation", "updated_at": utc_now(), "expected_queries": 50, "completed_queries": 0},
    )
    unit_dir.mkdir(parents=True, exist_ok=True)
    automation = root / "automation"
    automation.mkdir()
    for suffix in ("shard0", "shard1", "aggregate"):
        name = f"{prefix}-{suffix}"
        for extension, contents in ((".service", service(root, prefix, suffix)), (".timer", timer(name))):
            path = unit_dir / (name + extension)
            with path.open("x") as output:
                output.write(contents)
            (automation / path.name).write_text(contents)
    subprocess.run(
        ["systemd-analyze", "--user", "verify", *[str(unit_dir / (n + ".service")) for n in names]], check=True
    )
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "--user", "enable", "--now", *[n + ".timer" for n in names]], check=True)
    subprocess.run(["systemctl", "--user", "start", "--no-block", *[n + ".service" for n in names]], check=True)
    atomic_json(
        root / "launch_receipt.json",
        {"launched_at": utc_now(), "units": names, "plan_sha256": sha256(root / "plan.json")},
    )
    print(
        json.dumps(
            {
                "run": str(root),
                "videos": 50,
                "requests": sum(s["planned_requests"] for s in shards),
                "status": "preparation_and_idle_gpu_queue_launched",
            }
        )
    )


if __name__ == "__main__":
    main()
