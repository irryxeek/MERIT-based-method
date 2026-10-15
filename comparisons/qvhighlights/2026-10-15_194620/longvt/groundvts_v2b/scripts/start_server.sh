#!/usr/bin/env bash
set -euo pipefail

ROOT=/path/to/workspace/LongVT
OUT="$ROOT/reproduction/qvhighlights/groundvts_inference_v1/results/server"
ENV_ROOT=/path/to/storage/longvt-repro/env/inference
HF_HOME=/path/to/storage/longvt-repro/cache/huggingface
MODEL=/path/to/storage/longvt-repro/cache/huggingface/hub/models--longvideotool--LongVT-RFT/snapshots/16ed9314e4ea600dc4682b172a614252f2f383ad
TEMPLATE="$ROOT/examples/eval/tool_call_qwen2_5_vl.jinja"
PORT=18007
LOG="$OUT/vllm_tp2_64k_port${PORT}.log"
PID_FILE="$OUT/vllm_tp2_64k_port${PORT}.pid"

mkdir -p "$OUT"
if [[ -e "$LOG" || -e "$PID_FILE" ]]; then
    echo "Refusing to overwrite an existing experiment server log or PID file" >&2
    exit 1
fi

python3 - <<'PY'
import subprocess

expected = {
    6: "GPU-REDACTED-01",
    7: "GPU-REDACTED-02",
}
for index, wanted_uuid in expected.items():
    actual = subprocess.check_output(
        ["nvidia-smi", "-i", str(index), "--query-gpu=uuid", "--format=csv,noheader,nounits"], text=True
    ).strip()
    if actual != wanted_uuid:
        raise SystemExit(f"GPU {index} UUID mismatch: {actual}")
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader,nounits"],
        text=True,
    ).splitlines()
    occupied = [row for row in rows if row.split(",", 1)[0].strip() == actual]
    if occupied:
        raise SystemExit(f"GPU {index} is occupied: {occupied}")
PY

if ss -ltn "sport = :$PORT" | tail -n +2 | grep -q .; then
    echo "Port $PORT is already listening" >&2
    exit 1
fi
[[ -d "$MODEL" && -f "$TEMPLATE" ]]

export HF_HOME
export CUDA_VISIBLE_DEVICES=6,7
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost

setsid nohup "$ENV_ROOT/bin/vllm" serve "$MODEL" \
    --served-model-name longvideotool/LongVT-RFT \
    --dtype bfloat16 \
    --chat-template "$TEMPLATE" \
    --tool-call-parser hermes \
    --enable-auto-tool-choice \
    --trust-remote-code \
    --tensor-parallel-size 2 \
    --max-model-len 65536 \
    --gpu-memory-utilization 0.85 \
    --port "$PORT" \
    > "$LOG" 2>&1 < /dev/null &

pid=$!
printf '%s\n' "$pid" > "$PID_FILE"
printf 'Started isolated LongVT-RFT vLLM server PID=%s port=%s GPUs=6,7\nLog=%s\n' "$pid" "$PORT" "$LOG"
