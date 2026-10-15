#!/usr/bin/env bash
set -euo pipefail

ROOT=/path/to/workspace/LongVT
GPU_IDS=(6 7)

while true; do
    busy=0
    for gpu in "${GPU_IDS[@]}"; do
        uuid=$(nvidia-smi -i "${gpu}" --query-gpu=uuid --format=csv,noheader,nounits)
        if nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits \
            | awk -F', ' -v uuid="${uuid}" '$1 == uuid {found=1} END {exit !found}'; then
            busy=1
            break
        fi
    done
    if [[ "${busy}" -eq 0 ]]; then
        cd "${ROOT}"
        exec bash reproduction/qvhighlights/scripts/start_longvt_server.sh
    fi
    sleep 60
done
