#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-18006}"
GPU_IDS=(6 7)
ENV_ROOT=/path/to/storage/longvt-repro/env/inference
HF_HOME=/path/to/storage/longvt-repro/cache/huggingface
MODEL_ROOT=/path/to/storage/longvt-repro/cache/huggingface/hub/models--longvideotool--LongVT-RFT
MODEL="${MODEL_ROOT}/snapshots/16ed9314e4ea600dc4682b172a614252f2f383ad"
TEMPLATE=/path/to/workspace/LongVT/examples/eval/tool_call_qwen2_5_vl.jinja
OUTPUT_DIR=/path/to/workspace/LongVT/reproduction/qvhighlights/results/server
LOG_PATH="${OUTPUT_DIR}/vllm_tp2_64k_port${PORT}.log"
PID_PATH="${OUTPUT_DIR}/vllm_tp2_64k_port${PORT}.pid"

mkdir -p "${OUTPUT_DIR}"

for gpu in "${GPU_IDS[@]}"; do
    uuid=$(nvidia-smi -i "${gpu}" --query-gpu=uuid --format=csv,noheader,nounits)
    processes=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
        --format=csv,noheader,nounits | awk -F', ' -v uuid="${uuid}" '$1 == uuid {print}')
    if [[ -n "${processes}" ]]; then
        printf 'Refusing to start: GPU %s has compute processes:\n%s\n' "${gpu}" "${processes}" >&2
        exit 1
    fi
done

if ss -ltn "sport = :${PORT}" | tail -n +2 | grep -q .; then
    printf 'Refusing to start: TCP port %s is already listening\n' "${PORT}" >&2
    exit 1
fi

export HF_HOME
export CUDA_VISIBLE_DEVICES=6,7
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost

setsid nohup "${ENV_ROOT}/bin/vllm" serve "${MODEL}" \
    --served-model-name longvideotool/LongVT-RFT \
    --chat-template "${TEMPLATE}" \
    --tool-call-parser hermes \
    --enable-auto-tool-choice \
    --trust-remote-code \
    --tensor-parallel-size 2 \
    --max-model-len 65536 \
    --gpu-memory-utilization 0.85 \
    --port "${PORT}" \
    > "${LOG_PATH}" 2>&1 < /dev/null &

pid=$!
printf '%s\n' "${pid}" > "${PID_PATH}"
printf 'Started LongVT vLLM parent PID %s\nLog: %s\n' "${pid}" "${LOG_PATH}"
