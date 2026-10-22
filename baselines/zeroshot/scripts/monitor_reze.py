"""Read-only GPU telemetry until the owning experiment service finishes."""

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--unit", required=True)
    args = parser.parse_args()
    binding = json.loads((args.run / "binding.json").read_text())
    indices = ",".join(str(g["index"]) for g in binding["gpus"])
    path = args.run / "gpu_telemetry.csv"
    with path.open("a", newline="") as stream:
        writer = csv.writer(stream)
        if path.stat().st_size == 0:
            writer.writerow(["time", "gpu_index", "gpu_uuid", "memory_mib", "gpu_utilization_pct", "power_w"])
        while True:
            state = subprocess.check_output(
                ["systemctl", "--user", "show", args.unit, "--property=ActiveState", "--value"], text=True
            ).strip()
            if state not in {"active", "activating"} or (args.run / "completion.json").exists():
                break
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "-i",
                    indices,
                    "--query-gpu=timestamp,index,uuid,memory.used,utilization.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            )
            writer.writerows([[cell.strip() for cell in row] for row in csv.reader(output.splitlines())])
            stream.flush()
            time.sleep(10)


if __name__ == "__main__":
    main()
