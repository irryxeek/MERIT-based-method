import _bootstrap  # noqa: F401 -- establish the isolated implementation import
import argparse
import fcntl
import json
import os
import signal
import time
from pathlib import Path

import videoarm
from videoarm.qvhighlights.backend import LocalBackend
from videoarm.qvhighlights.common import (
    ARTIFACTS,
    atomic_json,
    code_identity,
    digest,
    file_hash,
    load_manifest,
    read_jsonl,
)
from videoarm.qvhighlights.contracts import validate_predictions
from videoarm.qvhighlights.media import FrameReader, probe
from videoarm.qvhighlights.pipeline import Engine
from videoarm.qvhighlights.state import BudgetExceeded, Store


class PauseRequested(BaseException):
    """Interrupt an in-flight request without turning pause into model failure."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["mr", "hd"], required=True)
    ap.add_argument("--method", choices=["B0", "B1", "M1", "A1"], required=True)
    ap.add_argument(
        "--split", choices=["smoke8", "dev50", "val", "test"], required=True
    )
    ap.add_argument("--config", default=str(ARTIFACTS / "configs/default_v1.json"))
    ap.add_argument(
        "--data-root",
        default=os.getenv("QVH_DATA_ROOT", "/path/to/storage/videoarm-repro/qvh_data"),
    )
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    if Path(a.run_id).name != a.run_id:
        raise ValueError("run_id must be a simple directory name")
    if (a.method == "A1" and a.split not in {"dev50", "smoke8"}) or (
        a.split == "test" and a.method == "B0"
    ):
        raise ValueError("method outside authorized experimental scope")
    config = json.loads(Path(a.config).read_text())
    config.update(task=a.task, method=a.method, split=a.split)
    config["limits"]["images"] = 640 if a.task == "hd" else 512
    manifest_path = ARTIFACTS / "manifests" / f"{a.split}.jsonl"
    rows = load_manifest(manifest_path)
    frozen_media = None
    identity = {
        "code": code_identity(),
        "config": config,
        "manifest_sha256": file_hash(manifest_path),
        "videoarm_import": videoarm.__file__,
    }
    directory = Path(a.data_root) / "runs" / a.run_id
    if directory.exists() and not (a.resume or a.dry_run):
        raise ValueError("run exists: use --resume; budgets cannot be reset")
    if a.split in {"val", "test"}:
        frozen = ARTIFACTS / "checks/formal_freeze.json"
        if not frozen.exists():
            raise ValueError(
                "P2 real development gates and formal freeze are required before val/test"
            )
        freeze = json.loads(frozen.read_text())
        if (
            identity["code"] != freeze["code"]
            or file_hash(a.config) != freeze["config_sha256"]
        ):
            raise ValueError("formal freeze drift")
        if identity["manifest_sha256"] != freeze["manifests"][a.split]:
            raise ValueError("formal manifest drift")
        if a.run_id != freeze["run_ids"][a.task][a.split][a.method]:
            raise ValueError(
                "formal run_id differs from predeclared registry; cannot reset an attempt with a new ID"
            )
        media_audit = ARTIFACTS / "checks/media_all.jsonl"
        if file_hash(media_audit) != freeze["media_audit_sha256"]:
            raise ValueError("formal media audit drift")
        frozen_media = {r["vid"]: r for r in read_jsonl(media_audit)}
        if a.limit:
            raise ValueError(
                "formal runs retain complete denominators; --limit is development only"
            )
    if a.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run_no_model_requests",
                    "planned_queries": len(rows),
                    "task": a.task,
                    "method": a.method,
                    "videoarm_import": videoarm.__file__,
                    "missing_media": sum(
                        not Path(r["media_path"]).is_file() for r in rows
                    ),
                    "identity_sha256": digest(identity),
                }
            )
        )
        return
    backend = LocalBackend(config)
    verified_path = ARTIFACTS / "checks/backend_verified.json"
    if not verified_path.exists():
        raise RuntimeError(
            "no verified QVH-owned TP2 backend; run launch_qvh_service.py after two GPUs are free"
        )
    verified = json.loads(verified_path.read_text())
    if (
        verified["model_revision"] != config["model_revision"]
        or verified["base_url"] != config["base_url"]
    ):
        raise ValueError("backend verification does not match model/config")
    os.kill(verified["pid"], 0)
    catalog = backend.verify()
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / "runner.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    store = Store(directory, identity, rows)
    store.recover()
    atomic_json(directory / "identity.json", identity)
    atomic_json(directory / "backend_catalog.json", catalog)
    stop = {"requested": False}

    def pause(signum, frame):
        stop["requested"] = True
        raise PauseRequested()

    signal.signal(signal.SIGTERM, pause)
    signal.signal(signal.SIGINT, pause)
    completed = 0
    for row in rows:
        if a.limit and completed >= a.limit:
            break
        item = store.item(row["qid"])
        if item["status"] in {"succeeded", "failed", "budget_exhausted"}:
            continue
        if stop["requested"]:
            store.checkpoint(row["qid"], status="paused", error="user_or_service_pause")
            break
        reader = None
        store.checkpoint(row["qid"], status="running", error=None)
        t = time.monotonic()
        try:
            if frozen_media and frozen_media[row["vid"]]["status"] != "passed":
                raise ValueError(
                    "predeclared full-media audit failure; retained in planned denominator"
                )
            info = probe(
                row["media_path"], row["duration"], Path(a.data_root) / "timelines"
            )
            if frozen_media and any(
                info[k] != frozen_media[row["vid"]][k]
                for k in ("media_sha256", "timeline_sha256")
            ):
                raise ValueError("frozen media/PTS identity changed")
            item = store.item(row["qid"])
            if (
                item["snapshot"].get("media_sha256", info["media_sha256"])
                != info["media_sha256"]
            ):
                raise ValueError("media changed across resume")
            item["snapshot"]["media_sha256"] = info["media_sha256"]
            item["snapshot"]["timeline_sha256"] = info["timeline_sha256"]
            store.checkpoint(row["qid"], snapshot=item["snapshot"])
            reader = FrameReader(row["media_path"], info)
            item = store.item(row["qid"])
            item["budget"]["active_seconds"] += time.monotonic() - t
            item["snapshot"].setdefault("executions", []).append(
                {"start_unix": time.time(), "end_unix": None}
            )
            store.checkpoint(
                row["qid"], snapshot=item["snapshot"], budget=item["budget"]
            )
            engine = Engine(row, config, store, backend, reader, info)
            try:
                prediction = engine.run()
            finally:
                engine.shared.close()
            validate_predictions([prediction], [row], a.task)
            store.checkpoint(row["qid"], status="succeeded", prediction=prediction)
        except PauseRequested:
            store.checkpoint(
                row["qid"],
                status="paused",
                error="user_or_service_pause; in-flight reservations retained",
            )
        except Exception as e:
            store.checkpoint(
                row["qid"],
                status="budget_exhausted"
                if isinstance(e, BudgetExceeded)
                else "failed",
                error=f"{type(e).__name__}: {e}",
            )
        finally:
            if reader:
                reader.close()
            item = store.item(row["qid"])
            if item["snapshot"].get("executions"):
                item["snapshot"]["executions"][-1]["end_unix"] = time.time()
                store.checkpoint(row["qid"], snapshot=item["snapshot"])
        completed += 1
        statuses = store.export()
        atomic_json(
            directory / "progress.json",
            {
                "run_id": a.run_id,
                "task": a.task,
                "method": a.method,
                "planned": len(rows),
                "statuses": statuses,
                "latest_qid": row["qid"],
                "updated": time.time(),
            },
        )
        print(
            json.dumps(
                {
                    "qid": row["qid"],
                    "status": store.item(row["qid"])["status"],
                    "elapsed": time.monotonic() - t,
                    "counts": statuses,
                }
            ),
            flush=True,
        )
        if stop["requested"]:
            break
    store.export()


if __name__ == "__main__":
    main()
