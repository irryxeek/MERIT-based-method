"""Resume only unfinished queries using the existing frozen inference implementation."""

import argparse
import fcntl
import json
import os
import sys
import time
import traceback
from pathlib import Path

from merit_qvh.cli import check_revisions, evaluate, frozen_run, infer_query
from merit_qvh.client import CachedClient
from merit_qvh.core import digest, read_jsonl, validate_manifest, validate_scores, write_json
from merit_qvh.embedding import Embedder
from merit_qvh.pipeline import Pipeline


def completed_results(root, rows, protocol_id):
    completed = {}
    for row in rows:
        path = root / "queries" / f"{row['qid']}.json"
        if not path.exists():
            continue
        result = json.loads(path.read_text())
        if (
            result["qid"] != row["qid"]
            or result["vid"] != row["vid"]
            or result["query"] != row["query"]
            or result["protocol_id"] != protocol_id
            or result["status"] not in {"ok", "failed"}
        ):
            raise ValueError(f"Incompatible completed result: {path}")
        if result.get("failure_stage") in {"memory", "index"}:
            identity = digest(
                {"protocol": protocol_id, "input": row, "source_sha256": result["source_sha256"]}
            )
        else:
            memory = json.loads((root / "memories" / f"{row['vid']}.json").read_text())
            identity = digest({"protocol": protocol_id, "input": row, "memory": memory["identity"]})
        if identity != result["identity"]:
            raise ValueError(f"Completed result identity mismatch: {path}")
        scores = validate_scores({"scores": result["pred_saliency_scores"]}, int(row["duration"] / 2))
        if result["status"] == "failed" and any(scores):
            raise ValueError("A failed query must retain its whole-query zero fallback")
        completed[row["qid"]] = result
    return completed


def resume_rows(pipeline, rows, completed, progress, health_check):
    """Historical failures do not re-trigger the fresh-query failure circuit breaker."""
    failures = 0
    progress(completed)
    for row in rows:
        if row["qid"] in completed:
            continue
        # Check infrastructure before beginning a new observation.
        health_check()
        print(json.dumps({"event": "start_video", "qid": row["qid"], "vid": row["vid"]}), flush=True)
        result = infer_query(pipeline, row)
        completed[row["qid"]] = result
        progress(completed)
        print(json.dumps({"event": "query_done", "qid": row["qid"], "status": result["status"]}), flush=True)
        failures = failures + 1 if result["status"] != "ok" else 0
        if failures >= 3:
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate frozen inputs and completed results without inference",
    )
    args = parser.parse_args()
    root = args.run_root.resolve()
    protocol = json.loads((root / "protocol.json").read_text())
    config = protocol["config"]
    protocol_id, expected_root = frozen_run(config)
    if expected_root.resolve() != root or digest(protocol) != protocol_id:
        raise ValueError("Live source differs from the frozen run; automatic migration is forbidden")
    check_revisions(config)
    audited = json.loads((root / "model_audit.json").read_text())
    for name in ("vlm", "embedding"):
        if audited[name]["revision"] != config[name]["revision"]:
            raise ValueError("Model audit revision differs")
    smoke_rows = read_jsonl(root / "manifests/smoke.jsonl")
    smoke = completed_results(root, smoke_rows, protocol_id)
    if len(smoke) != 5 or any(result["status"] != "ok" for result in smoke.values()):
        raise ValueError("The five-video smoke gate has not passed")
    rows = validate_manifest(read_jsonl(root / "manifests/val.jsonl"))
    if len(rows) != config["full"]["expected_queries"]:
        raise ValueError("Validation manifest is incomplete")
    if args.check_only:
        completed = completed_results(root, rows, protocol_id)
        print(
            json.dumps(
                {
                    "protocol_id": protocol_id,
                    "verified_completed": len(completed),
                    "pending": len(rows) - len(completed),
                }
            )
        )
        return 0
    state = {
        "protocol_id": protocol_id,
        "pid": os.getpid(),
        "started_at": time.time(),
        "endpoint": args.endpoint,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "mode": "resume_unfinished_only",
    }

    def status(stage, **fields):
        state.update(stage=stage, updated_at=time.time(), **fields)
        write_json(root / "execution_status.json", state)

    def progress(completed):
        queries = [
            {"qid": row["qid"], "status": completed[row["qid"]]["status"]}
            for row in rows
            if row["qid"] in completed
        ]
        write_json(
            root / "val_progress.json", {"completed": len(queries), "total": len(rows), "queries": queries}
        )
        status("validation", completed=len(queries), total=len(rows))

    with (root / "orchestrator.lock").open("a") as outer, (root / "run.lock").open("a") as inner:
        try:
            for lock in (outer, inner):
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Another runner holds the run lock; no duplicate worker was started", flush=True)
            return 0
        previous = root / "execution_status.json"
        if previous.exists():
            write_json(
                root / "runtime" / f"execution_previous_{time.time_ns()}.json",
                json.loads(previous.read_text()),
            )
        try:
            completed = completed_results(root, rows, protocol_id)
            client = CachedClient(args.endpoint, config["vlm"], root / "requests", protocol_id)
            pipeline = Pipeline(
                config, protocol_id, root, client, Embedder(config["embedding"], config["artifact_root"])
            )
            if not resume_rows(pipeline, rows, completed, progress, client.check_model):
                status(
                    "awaiting_recovery", reason="Three new consecutive failures; supervisor must check health"
                )
                return 75
            status("evaluating", completed=len(rows), total=len(rows))
            report = evaluate(config, root, "val")
            status(
                "complete",
                validation_queries=report["queries"],
                validation_failures=len(report["failed_qids"]),
                metrics=report["official_metrics"]["brief"],
            )
            return 0
        except Exception as error:
            status("failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
            raise


if __name__ == "__main__":
    sys.exit(main())
