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
    scope = json.loads((ARTIFACTS / "checks/m1_scope_revision.json").read_text())
    if scope["methods"] != ["M1"]:
        raise ValueError("current task is M1-only")
    efficiency = json.loads((ARTIFACTS / "checks/concurrency_benchmark_m1.json").read_text())
    execution = json.loads((ARTIFACTS / "configs/execution_m1.json").read_text())
    if execution["query_workers"] != efficiency["selected_query_workers"]:
        raise ValueError("execution must use measured stable concurrency")
    for task in ("mr", "hd"):
        for split in ("smoke8", "dev50"):
            run = root / f"dev_v2_{task}_{split}_M1"
            db = sqlite3.connect(f"file:{run / 'state.sqlite'}?mode=ro", uri=True)
            statuses = dict(db.execute("SELECT status,COUNT(*) FROM items GROUP BY status"))
            db.close()
            if sum(statuses.values()) != (8 if split == "smoke8" else 50) or any(statuses.get(s, 0) for s in ("pending", "running", "paused")):
                raise ValueError("all planned M1 development attempts must be terminal; failures retained")
            if statuses.get("succeeded", 0) == 0:
                raise ValueError("no valid M1 development predictions")
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
            for split, methods in (("val", ("M1",)), ("test", ("M1",)))
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
        "methods": ["M1"],
        "execution": execution,
        "execution_config_sha256": file_hash(ARTIFACTS / "configs/execution_m1.json"),
        "scope_revision_sha256": file_hash(ARTIFACTS / "checks/m1_scope_revision.json"),
        "concurrency_benchmark_sha256": file_hash(ARTIFACTS / "checks/concurrency_benchmark_m1.json"),
        "quality_tuning_after_val": False,
        "development_amendment_sha256": file_hash(
            ARTIFACTS / "checks/development_amendment_v2.json"
        ),
    }
    atomic_json(existing, snapshot)
    print(json.dumps({"frozen": str(existing), "tag": tag}))


if __name__ == "__main__":
    main()
