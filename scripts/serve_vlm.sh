#!/usr/bin/env bash
# Start only after GPUs have been assigned to this experiment. Existing services are not stopped.
set -euo pipefail
: "${MERIT_GPU_IDS:?Set MERIT_GPU_IDS to the GPUs assigned to this experiment}"
MERIT_PY=${MERIT_PY:-python}
: "${MERIT_VLM_MODEL:?Set MERIT_VLM_MODEL to the pinned local Qwen3-VL-8B-Instruct directory}"
MERIT_PORT=${MERIT_PORT:-8036}
MERIT_TP_SIZE=${MERIT_TP_SIZE:-2}
MERIT_VLM_GPU_MEMORY=${MERIT_VLM_GPU_MEMORY:-0.8}
export CUDA_VISIBLE_DEVICES="$MERIT_GPU_IDS"
exec "$MERIT_PY" -m vllm.entrypoints.openai.api_server \
  --model "$MERIT_VLM_MODEL" \
  --served-model-name Qwen/Qwen3-VL-8B-Instruct \
  --host 127.0.0.1 --port "$MERIT_PORT" \
  --tensor-parallel-size "$MERIT_TP_SIZE" --dtype bfloat16 \
  --max-model-len 32768 --max-num-seqs 4 --gpu-memory-utilization "$MERIT_VLM_GPU_MEMORY" \
  --limit-mm-per-prompt '{"image":90,"video":0}' \
  --mm-processor-kwargs '{"max_pixels":200704}'
