import _bootstrap  # noqa: F401
import json
import subprocess
import time

from videoarm.qvhighlights.common import ARTIFACTS, atomic_json

launch = json.loads((ARTIFACTS / "checks/service_launch.json").read_text())
uuids = set(launch["gpu_uuids"])
start = launch["time"]
peak = {u: 0 for u in uuids}
path = ARTIFACTS / "logs/qvh_gpu_telemetry.jsonl"
while True:
    state = subprocess.run(
        ["systemctl", "--user", "is-active", "videoarm-qvh-model.service"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    if state not in {"active", "activating"}:
        break
    values = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=uuid,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    GPUs = []
    for line in values.splitlines():
        u, used, utilization = [v.strip() for v in line.split(",")]
        if u in uuids:
            peak[u] = max(peak[u], int(used))
            GPUs.append(
                {
                    "uuid": u,
                    "memory_used_mib": int(used),
                    "utilization_percent": int(utilization),
                }
            )
    with path.open("a") as f:
        f.write(json.dumps({"time": time.time(), "gpus": GPUs}) + "\n")
    atomic_json(
        ARTIFACTS / "checks/gpu_telemetry_summary.json",
        {
            "start_unix": start,
            "observed_until_unix": time.time(),
            "peak_memory_per_gpu_mib": peak,
            "allocated_gpu_hours": len(uuids) * (time.time() - start) / 3600,
            "scope": "QVH-owned service lifetime, including startup and idle time; GPU memory includes server allocation",
        },
    )
    time.sleep(5)
