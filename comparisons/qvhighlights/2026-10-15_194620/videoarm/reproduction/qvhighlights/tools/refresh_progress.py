"""Reconstruct progress from artifact files and independent run databases."""

import _bootstrap  # noqa: F401 -- establish the isolated implementation import
import json
import os
import sqlite3
import time
from pathlib import Path

from videoarm.qvhighlights.common import ARTIFACTS, atomic_json


def refresh(data_root=None):
    data_root = Path(
        data_root or os.getenv("QVH_DATA_ROOT", "/path/to/storage/videoarm-repro/qvh_data")
    )
    totals = {}
    for db_path in sorted((data_root / "runs").glob("*/state.sqlite")):
        db = sqlite3.connect(db_path)
        totals[db_path.parent.name] = dict(
            db.execute("SELECT status,COUNT(*) FROM items GROUP BY status")
        )
        db.close()
    media = ARTIFACTS / "checks/media_all_progress.json"
    gpu = ARTIFACTS / "checks/gpu_readiness.json"
    fixtures = ARTIFACTS / "checks/official_fixture_check.json"
    offline = ARTIFACTS / "checks/offline_tests.json"
    worker_file = ARTIFACTS / "checks/worker_status.json"
    worker = json.loads(worker_file.read_text()) if worker_file.exists() else {}
    historical = {
        k: v for k, v in totals.items() if k.startswith(("dev_v1_", "dev_low_v1_"))
    }
    retained = {k: v for k, v in totals.items() if k not in historical and not k.endswith("_M1")}
    current = {k: v for k, v in totals.items() if k not in historical and k not in retained}
    scored = [
        json.loads(p.read_text())
        for p in (data_root / "runs").glob("*/scores.json")
        if p.parent.name.startswith("formal_")
    ]
    result = {
        "project": "VideoARM 在 QVHighlights 上的开源模型迁移评测",
        "updated_unix": time.time(),
        "data_root": str(data_root),
        "svhighlights_inference": "paused_by_user_unchanged",
        "current_methods": ["M1"],
        "paper_comparison": "PAPER_COMPARISON.md",
        "execution_config": "configs/execution_m1.json",
        "stages": {
            "P0": {
                "status": "audit_saved",
                "inventory": "checks/data_inventory.json",
                "dev_timeline": "checks/data_audit_dev50_mapping.json",
                "full_timeline": json.loads(media.read_text())
                if media.exists()
                else {"planned": 10148, "completed": 0, "failed": 0},
            },
            "P1": {
                "status": "offline_checked"
                if offline.exists()
                else "implementation_in_progress",
                "tests": json.loads(offline.read_text()) if offline.exists() else None,
                "official_fixture": json.loads(fixtures.read_text())
                if fixtures.exists()
                else None,
            },
            "P2": {
                "status": worker.get("status", "resource_blocked")
                if not totals
                else "inspect_run_statuses",
                "runs": current,
                "historical_v1_runs": historical,
                "retained_deferred_other_methods": retained,
                "development_revision": "v2_stop_contract_m1_parallel_v1",
                "scope_revision": "checks/m1_scope_revision.json",
                "planned_condition_queries": 116,
                "smoke8": {"queries_per_condition": 8, "conditions": 2},
                "dev50": {
                    "queries_per_condition": 50,
                    "default_conditions": 2,
                    "low_budget_conditions": 0,
                },
                "gpu": json.loads(gpu.read_text()) if gpu.exists() else None,
            },
            "P3": {
                "status": "pending_P2_and_freeze",
                "planned_queries_per_task_method": 1550,
                "tasks": ["mr", "hd"],
                "methods": ["M1"],
            },
            "P4": {
                "status": "pending_P3_same_freeze",
                "planned_queries_per_task_method": 1542,
                "tasks": ["mr", "hd"],
                "methods": ["M1"],
            },
            "P5": {"status": "partial_handoff_report", "report": "FINAL_REPORT.md"},
        },
        "latest_checkpoints": [
            str(media.relative_to(ARTIFACTS))
            if media.exists()
            else "checks/data_audit_dev50_mapping.json"
        ],
        "worker": worker,
        "development_amendment": "checks/development_amendment_v2.json",
        "completed_full_model_evaluations": sum(
            r["status"] == "complete_official_submission" for r in scored
        ),
        "performance_claim": None,
    }
    if worker.get("phase") in {"P3", "P4", "P5"}:
        result["stages"]["P2"]["status"] = "real_development_attempts_completed"
        result["stages"]["P3"]["status"] = (
            "running"
            if worker["phase"] == "P3"
            else "attempts_completed_inspect_coverage"
        )
    if worker.get("phase") in {"P4", "P5"}:
        result["stages"]["P4"]["status"] = (
            "running"
            if worker["phase"] == "P4"
            else "attempts_completed_inspect_coverage"
        )
    if worker.get("status") == "completed":
        result["stages"]["P5"]["status"] = "generated_after_formal_attempts"
    atomic_json(ARTIFACTS / "progress.json", result)
    return result


if __name__ == "__main__":
    p = refresh()
    print(
        json.dumps({"updated": "progress.json", "runs": len(p["stages"]["P2"]["runs"])})
    )
