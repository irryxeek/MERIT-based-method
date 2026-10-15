"""Freeze once after real development and budget measurement, before full val."""

import _bootstrap  # noqa: F401
import json
import sqlite3
import os
import time
from pathlib import Path

from videoarm.qvhighlights.common import (
    ARTIFACTS,
    atomic_json,
    code_identity,
    digest,
    file_hash,
)


def main():
    existing = ARTIFACTS / "checks/formal_freeze.json"
    if existing.exists():
        if json.loads(existing.read_text())["code"] != code_identity():
            raise ValueError(
                "existing formal freeze drift: preserve prior experiment; register a documented revision"
            )
        print("Existing formal freeze preserved")
        return
    amendment = json.loads(
        (ARTIFACTS / "checks/development_amendment_v2.json").read_text()
    )
    if amendment.get("validation_status") != "offline_and_live_v2_verified":
        raise ValueError(
            "v2 stop contract must pass historical reparse, regression and live autonomous checks before val"
        )
    for filename in ("live_image_binding.json", "live_forced_interfaces.json"):
        data = json.loads((ARTIFACTS / "checks" / filename).read_text())
        if not data["passed"]:
            raise ValueError("real development gate failed")
    media = json.loads((ARTIFACTS / "checks/media_all_progress.json").read_text())
    if media["completed"] != media["planned"]:
        raise ValueError("complete media audit required before formal freeze")
    root = (
        Path(os.getenv("QVH_DATA_ROOT", str(ARTIFACTS.parents[1] / "qvh_data")))
        / "runs"
    )
    development = {}
    for task in ("mr", "hd"):
        for split in ("smoke8", "dev50"):
            for method in ("B0", "B1", "M1", "A1"):
                run = root / f"dev_v2_{task}_{split}_{method}"
                db = sqlite3.connect(run / "state.sqlite")
                statuses = dict(
                    db.execute("SELECT status,COUNT(*) FROM items GROUP BY status")
                )
                n = sum(statuses.values())
                db.close()
                if (
                    n != (8 if split == "smoke8" else 50)
                    or statuses.get("pending", 0)
                    or statuses.get("running", 0)
                    or statuses.get("paused", 0)
                ):
                    raise ValueError(
                        "all planned development attempts must be terminal, with failures retained"
                    )
                if statuses.get("succeeded", 0) == 0:
                    raise ValueError(
                        "no valid development predictions in a planned condition"
                    )
                development[run.name] = statuses
        for method in ("B1", "M1"):
            run = root / f"dev_low_v2_{task}_dev50_{method}"
            db = sqlite3.connect(run / "state.sqlite")
            statuses = dict(
                db.execute("SELECT status,COUNT(*) FROM items GROUP BY status")
            )
            db.close()
            if sum(statuses.values()) != 50 or any(
                statuses.get(s, 0) for s in ("pending", "running", "paused")
            ):
                raise ValueError("low/high development budget study incomplete")
            development[run.name] = statuses
    estimate = ARTIFACTS / "checks/full_run_estimate.json"
    if not estimate.exists():
        raise ValueError("measured full-run cost estimate must be saved first")
    code = code_identity()
    tag = digest(code)[:12]
    run_ids = {
        task: {
            split: {
                method: f"formal_{tag}_{task}_{split}_{method}" for method in methods
            }
            for split, methods in (("val", ("B0", "B1", "M1")), ("test", ("B1", "M1")))
        }
        for task in ("mr", "hd")
    }
    snapshot = {
        "created": time.time(),
        "code": code,
        "tag": tag,
        "run_ids": run_ids,
        "config_sha256": file_hash(ARTIFACTS / "configs/default_v2.json"),
        "manifests": {
            s: file_hash(ARTIFACTS / f"manifests/{s}.jsonl")
            for s in ("train", "val", "test", "dev50", "smoke8")
        },
        "media_audit_sha256": file_hash(ARTIFACTS / "checks/media_all.jsonl"),
        "model": json.loads(
            (ARTIFACTS / "checks/model_hash_verification.json").read_text()
        ),
        "sources": json.loads((ARTIFACTS / "checks/source_lock.json").read_text()),
        "client_environment_sha256": file_hash(
            ARTIFACTS / "checks/client_environment_20261013.txt"
        ),
        "server_environment_sha256": file_hash(
            ARTIFACTS / "checks/server_environment_20261013.txt"
        ),
        "development_statuses": development,
        "quality_tuning_after_val": False,
        "development_amendment_sha256": file_hash(
            ARTIFACTS / "checks/development_amendment_v2.json"
        ),
    }
    atomic_json(existing, snapshot)
    print(json.dumps({"frozen": str(existing), "tag": tag}))


if __name__ == "__main__":
    main()
