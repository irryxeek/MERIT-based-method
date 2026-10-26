"""GPU selection must skip occupied cards while allowing any idle device."""

import subprocess

import dispatch
import pytest


def inventory(monkeypatch, rows, processes=""):
    def check_output(command, **kwargs):
        if "--query-compute-apps=gpu_uuid" in command:
            return processes
        assert "--query-gpu=index,uuid,memory.used,utilization.gpu" in command
        return rows

    monkeypatch.setattr(dispatch.subprocess, "check_output", check_output)


def test_any_gpu_can_use_idle_card_outside_previous_allowlist(monkeypatch):
    inventory(monkeypatch, "4, GPU-4, 22000, 90\n5, GPU-5, 15, 0\n7, GPU-7, 15, 0\n", "GPU-4\n GPU-5 \n")
    selected, _ = dispatch.idle_gpu("all")
    assert selected["index"] == 7
    assert dispatch.idle_gpu(["GPU-4", "GPU-5"])[0] is None


def test_all_policy_includes_lower_gpu_indices(monkeypatch):
    inventory(monkeypatch, "7, GPU-7, 15, 0\n0, GPU-0, 15, 0\n")
    assert dispatch.idle_gpu("all")[0]["index"] == 0
    assert dispatch.idle_gpu(["GPU-7", "GPU-0"])[0]["index"] == 7


@pytest.mark.parametrize("memory, utilization", [(512, 0), (15, 5)])
def test_memory_and_utilization_thresholds_reject_busy_devices(monkeypatch, memory, utilization):
    inventory(monkeypatch, f"6, GPU-6, {memory}, {utilization}\n")
    assert dispatch.idle_gpu("all")[0] is None


def test_second_check_rejects_a_device_that_became_occupied(monkeypatch):
    inventory(monkeypatch, "7, GPU-7, 15, 0\n")
    selected, _ = dispatch.idle_gpu("all")
    inventory(monkeypatch, "7, GPU-7, 15, 0\n", "GPU-7\n")
    assert dispatch.idle_gpu([selected["uuid"]])[0] is None


def test_missing_occupancy_data_fails_closed(monkeypatch):
    def unavailable(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "nvidia-smi")

    monkeypatch.setattr(dispatch.subprocess, "check_output", unavailable)
    with pytest.raises(subprocess.CalledProcessError):
        dispatch.idle_gpu("all")
