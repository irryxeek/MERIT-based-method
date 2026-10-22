"""Persistent preparation, native input checks and bounded dispatch for one shard."""

import argparse
import fcntl
import json
import subprocess
import sys
from pathlib import Path

from shared import HERE, atomic_json, sha256, utc_now


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    args = parser.parse_args()
    root = args.run.resolve()
    plan = json.loads((root / "plan.json").read_text())
    shard = plan["shards"][args.shard_index]
    if shard["index"] != args.shard_index:
        raise ValueError("Shard identity mismatch")
    for name, digest in plan["source_sha256"].items():
        if sha256(HERE / name) != digest:
            raise ValueError(f"Control source changed: {name}")
    run = root / shard["directory"]
    with (root / f"shard_{args.shard_index}.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        if (run / "complete.json").exists() or (run / "stopped.json").exists():
            return
        if not (run / "binding.json").exists():
            record_path = root / f"preparation_shard_{args.shard_index}.json"
            attempts = json.loads(record_path.read_text()) if record_path.exists() else []
            if len(attempts) >= 3:
                state = {"status": "preparation_attempt_limit", "updated_at": utc_now(), "attempts": attempts}
                atomic_json(run / "status.json", state)
                atomic_json(run / "stopped.json", state)
                return
            # Keep an incomplete preparation after a reboot; never overwrite it.
            if run.exists():
                archive = root / "preparation_attempts" / f"shard_{args.shard_index}_attempt_{len(attempts)}"
                archive.parent.mkdir(exist_ok=True)
                run.rename(archive)
            attempt = {"started_at": utc_now(), "status": "running", "number": len(attempts) + 1}
            attempts.append(attempt)
            atomic_json(record_path, attempts)
            command = [
                sys.executable,
                "-u",
                str(HERE / "prepare.py"),
                "--run",
                str(run),
                "--all-verified",
                "--shard-count",
                str(len(plan["shards"])),
                "--shard-index",
                str(args.shard_index),
                "--gpu-uuids",
                *shard["gpu_candidates"],
            ]
            with (root / f"prepare_shard_{args.shard_index}_attempt_{len(attempts)}.log").open("a") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=3600)
            attempt.update(
                status="complete" if result.returncode == 0 else "failed",
                returncode=result.returncode,
                finished_at=utc_now(),
            )
            atomic_json(record_path, attempts)
            if result.returncode:
                state = {"status": "preparation_failed", "updated_at": utc_now(), "returncode": result.returncode}
                atomic_json(run / "status.json", state)
                atomic_json(run / "stopped.json", state)
                return
        binding = json.loads((run / "binding.json").read_text())
        if binding["selected_qids"] != shard["video_ids"] or binding["source_sha256"] != plan["source_sha256"]:
            raise ValueError("Prepared shard differs from the frozen full-run plan")
        if not (run / "media_preflight.json").exists():
            with (run / "preflight.log").open("a") as log:
                result = subprocess.run(
                    [sys.executable, "-u", str(run / "source_snapshot/preflight_media.py"), "--run", str(run)],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=7200,
                )
            if result.returncode:
                state = {"status": "preflight_failed", "updated_at": utc_now(), "returncode": result.returncode}
                atomic_json(run / "status.json", state)
                atomic_json(run / "stopped.json", state)
                return
        subprocess.run(
            [sys.executable, "-u", str(run / "source_snapshot/dispatch.py"), "--run", str(run)],
            check=True,
        )


if __name__ == "__main__":
    main()
