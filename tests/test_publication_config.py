"""Portable configuration must preserve experiment settings and existing files."""

import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from hd_adapter.service import guard

SPEC = importlib.util.spec_from_file_location(
    "materialize_config", Path(__file__).resolve().parents[1] / "scripts/materialize_config.py"
)
CONFIG = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONFIG)


def test_relocation_preserves_protocol_and_does_not_overwrite(tmp_path):
    template = tmp_path / "template.json"
    output = tmp_path / "machine.json"
    template.write_text(json.dumps({"path": "${MERIT_ROOT}/frames", "temperature": 0, "clip_seconds": 2}))
    CONFIG.materialize(template, output, {"MERIT_ROOT": "/new project"})
    assert json.loads(output.read_text()) == {
        "path": "/new project/frames",
        "temperature": 0,
        "clip_seconds": 2,
    }
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        CONFIG.materialize(template, output, {"MERIT_ROOT": "/another"})
    assert output.read_bytes() == before


def test_missing_path_is_rejected_before_writing(tmp_path):
    template = tmp_path / "template.json"
    output = tmp_path / "machine.json"
    template.write_text('{"path": "${REPRO_DATA_ROOT}/video.mp4"}')
    with pytest.raises(ValueError, match="REPRO_DATA_ROOT"):
        CONFIG.materialize(template, output, {})
    assert not output.exists()


def test_gpu_guard_resolves_allocated_uuid_instead_of_fixed_slot():
    config = {"hd": {"server": {"tensor_parallel_size": 2, "gpu_uuids": ["GPU-b", "GPU-a"]}}}
    inventory = "2, GPU-a, 10\n7, GPU-b, 20\n"
    with patch("hd_adapter.service.subprocess.check_output", return_value=inventory):
        assert guard(config) == [7, 2]
    with patch("hd_adapter.service.subprocess.check_output", return_value="2, GPU-a, 8000\n7, GPU-b, 20\n"):
        with pytest.raises(RuntimeError, match="occupied"):
            guard(config)
