"""Initialize the same explicit import roots as the command-line entry points."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import io_utils  # noqa: E402,F401
