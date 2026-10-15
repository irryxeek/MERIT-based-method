import importlib.util
import os
import signal
import sys
import time
from pathlib import Path

import pytest

from merit_qvh.core import digest, write_json


def script_module(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


supervision = script_module("experiment_supervisor")
resumption = script_module("resume_validation")


def test_resume_skips_historical_failures_and_continues(tmp_path, monkeypatch):
    rows = [{"qid": number, "vid": str(number)} for number in range(5)]
    completed = {number: {"qid": number, "status": "failed"} for number in range(3)}
    calls, snapshots = [], []

    def infer(pipeline, row):
        calls.append(row["qid"])
        return {"qid": row["qid"], "status": "ok"}

    monkeypatch.setattr(resumption, "infer_query", infer)
    assert resumption.resume_rows(
        None, rows, completed, lambda values: snapshots.append(len(values)), lambda: None
    )
    assert calls == [3, 4]
    assert snapshots == [3, 4, 5]


def test_three_new_failures_checkpoint_before_recovery(monkeypatch):
    rows = [{"qid": number, "vid": str(number)} for number in range(5)]
    completed, snapshots = {}, []
    monkeypatch.setattr(resumption, "infer_query", lambda _, row: {"qid": row["qid"], "status": "failed"})
    assert not resumption.resume_rows(
        None, rows, completed, lambda values: snapshots.append(len(values)), lambda: None
    )
    assert list(completed) == [0, 1, 2]
    assert snapshots[-1] == 3


def test_endpoint_failure_does_not_create_failed_observation(monkeypatch):
    completed = {}
    monkeypatch.setattr(resumption, "infer_query", lambda *_: pytest.fail("inference must not start"))

    def unavailable():
        raise ConnectionError("service is offline")

    with pytest.raises(ConnectionError):
        resumption.resume_rows(None, [{"qid": 1}], completed, lambda _: None, unavailable)
    assert not completed


def test_resume_rejects_tampered_result_identity(tmp_path):
    row = {"qid": 1, "vid": "v", "query": "box", "duration": 4, "video_path": "/v.mp4"}
    result = {
        **{key: row[key] for key in ("qid", "vid", "query")},
        "protocol_id": "p",
        "status": "failed",
        "failure_stage": "memory",
        "source_sha256": "fixed",
        "identity": digest({"protocol": "p", "input": row, "source_sha256": "fixed"}),
        "pred_saliency_scores": [0.0, 0.0],
    }
    write_json(tmp_path / "queries/1.json", result)
    assert resumption.completed_results(tmp_path, [row], "p")[1] == result
    with pytest.raises(ValueError, match="identity mismatch"):
        resumption.completed_results(tmp_path, [{**row, "duration": 6}], "p")


def test_pid_reuse_cannot_target_an_unrelated_process(monkeypatch):
    identity = supervision.process_identity(os.getpid())
    assert supervision.same_process(identity)
    stale = {**identity, "start_ticks": identity["start_ticks"] - 1}
    assert not supervision.same_process(stale)
    calls = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: calls.append((pid, sig)))
    supervision.stop_owned({"process": stale, "members": [stale]})
    assert not calls


@pytest.fixture
def supervisor(tmp_path, monkeypatch):
    (tmp_path / "runtime").mkdir()
    (tmp_path / "logs").mkdir()
    config = {
        "run_root": str(tmp_path),
        "worker_command": [sys.executable, "-u", "-c", "import time; time.sleep(60)"],
        "worker_environment": {},
        "stall_seconds": 1800,
        "service_startup_seconds": 600,
        "stop_service_on_complete": False,
        "endpoint": "http://unused/v1",
    }
    instance = supervision.Supervisor(config)
    monkeypatch.setattr(instance, "verify_protocol", lambda: {"vlm": {}, "full": {"expected_queries": 5}})
    monkeypatch.setattr(instance, "health", lambda _: (True, None))
    monkeypatch.setattr(instance, "gpu_ready", lambda *_, **__: (True, None))
    monkeypatch.setattr(instance, "completed_report", lambda: False)
    original_active = instance.active
    service = {"pid": 999999, "started_at": 0}
    monkeypatch.setattr(
        instance, "active", lambda kind: (service, True) if kind == "service" else original_active(kind)
    )
    yield instance
    record = supervision.load_json(instance.record_path("worker"), {})
    supervision.stop_owned(record)
    for child in instance.children:
        child.wait(timeout=5)


def test_worker_crash_is_restarted_without_duplicate_worker(supervisor):
    supervisor.tick()
    first = supervision.load_json(supervisor.record_path("worker"))
    assert supervision.same_process(first["process"])
    supervisor.tick()
    assert supervision.load_json(supervisor.record_path("worker"))["pid"] == first["pid"]
    os.kill(first["pid"], signal.SIGKILL)
    supervisor.children[0].wait(timeout=5)
    supervisor.state["worker_next_attempt_at"] = 0
    supervisor.tick()
    second = supervision.load_json(supervisor.record_path("worker"))
    assert second["pid"] != first["pid"]
    assert supervision.same_process(second["process"])


def test_gpu_owned_by_other_job_is_not_taken(supervisor, monkeypatch):
    monkeypatch.setattr(supervisor, "gpu_ready", lambda *_, **__: (False, "unmanaged GPU process"))
    supervisor.tick()
    assert supervisor.state["phase"] == "waiting_for_gpu"
    assert not supervisor.record_path("worker").exists()


def test_stalled_worker_triggers_recovery(supervisor):
    supervisor.tick()
    record = supervision.load_json(supervisor.record_path("worker"))
    supervisor.state["last_progress_at"] = time.time() - 1801
    supervisor.tick()
    assert supervisor.state["phase"] == "recovering_stall"
    assert not supervision.same_process(record["process"])


def test_transient_service_probe_failure_does_not_restart(supervisor, monkeypatch):
    monkeypatch.setattr(supervisor, "health", lambda _: (False, "timeout"))
    supervisor.tick()
    supervisor.tick()
    assert supervisor.state["phase"] == "checking_service"
    assert not supervisor.record_path("worker").exists()


def test_three_bad_health_checks_restart_service(supervisor, monkeypatch):
    launched = []
    monkeypatch.setattr(supervisor, "health", lambda _: (False, "engine unavailable"))
    monkeypatch.setattr(supervisor, "spawn", launched.append)
    for _ in range(3):
        supervisor.tick()
    assert launched == ["service"]
    assert supervisor.state["phase"] == "starting_service"


def test_gpu_inventory_rejects_foreign_process_and_uuid_change(supervisor, monkeypatch):
    supervisor.config["gpu_uuids"] = {"4": "assigned-4", "5": "assigned-5"}
    applications = "assigned-4, 987654321\n"

    def inventory(command, **kwargs):
        if "--query-gpu=index,uuid,memory.free" in command:
            return "4, assigned-4, 11000\n5, assigned-5, 11000\n"
        return applications

    monkeypatch.setattr(supervision.subprocess, "check_output", inventory)
    # Exercise the real parser; the fixture's normal method is a stub.
    ready, reason = supervision.Supervisor.gpu_ready(supervisor, [])
    assert not ready and "unmanaged" in reason
    applications = ""
    assert supervision.Supervisor.gpu_ready(supervisor, [], require_embedding_space=True) == (True, None)
    supervisor.config["gpu_uuids"]["4"] = "different-device"
    ready, reason = supervision.Supervisor.gpu_ready(supervisor, [])
    assert not ready and "UUID" in reason


def test_ten_failed_observations_require_attention(supervisor):
    write_json(
        supervisor.root / "val_progress.json",
        {"completed": 10, "queries": [{"qid": number, "status": "failed"} for number in range(10)]},
    )
    supervisor.tick()
    assert supervisor.state["phase"] == "attention_required"
    assert not supervisor.record_path("worker").exists()


def test_restart_backoff_is_bounded():
    assert [supervision.backoff(n) for n in range(1, 8)] == [60, 120, 240, 480, 960, 1800, 1800]
