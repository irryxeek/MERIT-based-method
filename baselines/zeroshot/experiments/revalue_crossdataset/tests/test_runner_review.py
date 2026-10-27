"""CPU regressions for transfer provenance and supervisor interruption cleanup."""

import importlib.util
import signal
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
SPEC = importlib.util.spec_from_file_location("reviewed_crossdataset_runner", HERE / "run.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_verify_binds_common_and_per_video_parent_lists(tmp_path, monkeypatch):
    common, video = tmp_path / "common.json", tmp_path / "video.json"
    common.write_text("common receipt")
    video.write_text("video receipt")
    def receipts(path):
        return [{"path": str(path), "sha256": runner.sha256(path), "bytes": path.stat().st_size}]

    row = {"qid": "one", "parent_receipts": receipts(video)}
    runner.write_json(tmp_path / "manifest.json", {"source_sha256": {}, "input_sha256": {}})
    runner.write_json(tmp_path / "model_verification.json", {})
    runner.write_json(tmp_path / "inference.json", [row])
    runner.write_json(tmp_path / "inputs/adapter_manifest.json", {"parent_receipts": receipts(common)})
    monkeypatch.setattr(runner, "model_check", lambda _: None)
    assert runner.verify(tmp_path)[1] == [row]
    common.write_text("changed common receipt")
    with pytest.raises(ValueError, match="Parent cache drift"):
        runner.verify(tmp_path)
    common.write_text("common receipt")
    video.write_text("changed per-video receipt")
    with pytest.raises(ValueError, match="Parent cache drift"):
        runner.verify(tmp_path)


def test_changed_completed_record_is_rejected_before_evaluation(tmp_path, monkeypatch):
    runner.write_json(tmp_path / "manifest.json", {"frozen": True})
    manifest = {"shards": [["one"], ["two"]]}
    monkeypatch.setattr(runner, "verify", lambda _: (manifest, []))
    record_path = tmp_path / "shard_0/records/one.json"
    runner.write_json(record_path, {"qid": "one", "scores": [0.1]})
    runner.write_json(tmp_path / "shard_0/inference_complete.json", {
        "qids": ["one"], "manifest_sha256": runner.sha256(tmp_path / "manifest.json"),
        "records_sha256": {"one": runner.sha256(record_path)},
    })
    runner.write_json(record_path, {"qid": "one", "scores": [0.9]})
    with pytest.raises(ValueError, match="Completed prediction record changed"):
        runner.analyze(SimpleNamespace(run=tmp_path))
    assert not (tmp_path / "predictions.json").exists()


@pytest.mark.parametrize("interrupt_signal", [signal.SIGTERM, signal.SIGHUP, signal.SIGINT])
def test_supervisor_interrupt_cleans_both_workers_and_records_cost(tmp_path, monkeypatch, interrupt_signal):
    handlers, processes, cleaned = {}, [], []
    gpus = [{"uuid": "GPU-a", "index": 0}, {"uuid": "GPU-b", "index": 1}]
    inventory = {g["uuid"]: g for g in gpus}

    class Process:
        def __init__(self, *_args, **_kwargs):
            self.pid, self.returncode = 90000 + len(processes), None
            processes.append(self)

        def poll(self):
            return self.returncode

    def idle(candidates):
        return inventory[next(iter(inventory)) if candidates == "all" else candidates[0]], inventory

    def cleanup(process):
        cleaned.append(process.pid)
        process.returncode = -int(interrupt_signal)
        return []

    def interrupt(_seconds):
        handlers[interrupt_signal](interrupt_signal, None)

    monkeypatch.setattr(runner, "verify", lambda _: ({}, []))
    monkeypatch.setattr(runner, "idle_gpu", idle)
    monkeypatch.setattr(runner, "environment", lambda _: {})
    monkeypatch.setattr(runner.subprocess, "Popen", Process)
    monkeypatch.setattr(runner.signal, "signal", lambda signum, handler: handlers.setdefault(signum, handler))
    monkeypatch.setattr(runner.time, "sleep", interrupt)
    monkeypatch.setattr(runner, "cleanup", cleanup)
    monkeypatch.setattr(runner, "process_group_members", lambda _: [])
    with pytest.raises(InterruptedError, match="Supervisor signal"):
        runner.supervise(SimpleNamespace(run=tmp_path))
    assert len(processes) == 2
    assert cleaned == [p.pid for p in processes]
    for slot in (0, 1):
        receipt = runner.read_json(tmp_path / f"shard_{slot}/resource_receipt.json")
        assert receipt["interrupted"] is True
        assert receipt["exit_code"] == -int(interrupt_signal)
        assert receipt["remaining_owned_pids"] == []
        assert receipt["seconds"] >= 0
