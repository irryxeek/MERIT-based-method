"""Exercise the standalone CPU entry point without preloaded worker imports."""

import os
import subprocess
import sys
from pathlib import Path


def test_evaluator_starts_from_unrelated_directory_without_pythonpath(tmp_path):
    script = Path(__file__).resolve().parents[1] / "evaluate.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": "", "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "--run" in result.stdout
