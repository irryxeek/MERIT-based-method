import _bootstrap  # noqa: F401 -- establish the isolated implementation import
import csv
import io
import json
import subprocess
from pathlib import Path
from videoarm.qvhighlights.backend import LocalBackend
from videoarm.qvhighlights.common import ARTIFACTS, atomic_json

config = json.loads((ARTIFACTS / "configs/default_v2.json").read_text())
pid = int(
    subprocess.check_output(
        [
            "systemctl",
            "--user",
            "show",
            "videoarm-qvh-model.service",
            "-p",
            "MainPID",
            "--value",
        ],
        text=True,
    )
)
if pid <= 0:
    raise RuntimeError("owned QVH model service is not running")
command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
if (
    config["model_path"] not in command
    or "8026" not in command
    or "--tensor-parallel-size 2" not in command
):
    raise ValueError("owned service process identity mismatch")
catalog = LocalBackend(config).verify()
cgroup = subprocess.check_output(
    [
        "systemctl",
        "--user",
        "show",
        "videoarm-qvh-model.service",
        "-p",
        "ControlGroup",
        "--value",
    ],
    text=True,
).strip()
members = set()
for f in (Path("/sys/fs/cgroup") / cgroup.lstrip("/")).rglob("cgroup.procs"):
    members.update(int(x) for x in f.read_text().split())
processes = list(
    csv.DictReader(
        io.StringIO(
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-compute-apps=gpu_uuid,pid",
                    "--format=csv,nounits",
                ],
                text=True,
            )
        ),
        skipinitialspace=True,
    )
)
actual = {p["gpu_uuid"] for p in processes if int(p["pid"]) in members}
expected = set(
    json.loads((ARTIFACTS / "checks/service_launch.json").read_text())["gpu_uuids"]
)
if actual != expected:
    raise ValueError(
        f"CUDA/NVML UUID allocation mismatch: expected {expected}, actual {actual}"
    )
atomic_json(
    ARTIFACTS / "checks/backend_verified.json",
    {
        "model_revision": config["model_revision"],
        "base_url": config["base_url"],
        "pid": pid,
        "catalog": catalog,
        "command": command,
        "actual_gpu_uuids": sorted(actual),
    },
)
print(
    "QVH-owned model catalog and TP2 process verified; real image/context probes still required"
)
