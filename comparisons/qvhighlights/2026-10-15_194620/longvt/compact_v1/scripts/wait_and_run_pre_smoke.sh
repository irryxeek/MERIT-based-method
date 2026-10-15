#!/usr/bin/env bash
set -euo pipefail

ROOT=/path/to/workspace/LongVT
PY=/path/to/storage/longvt-repro/env/inference/bin/python
API=http://127.0.0.1:18006
RUN_DIR="${ROOT}/reproduction/qvhighlights/results/pre_smoke_1_v4"

for _ in $(seq 1 240); do
    if curl --noproxy '*' -fsS "${API}/v1/models" >/dev/null 2>&1; then
        break
    fi
    sleep 30
done

if ! curl --noproxy '*' -fsS "${API}/v1/models" >/dev/null 2>&1; then
    printf 'LongVT service did not become ready within two hours\n' >&2
    exit 1
fi

cd "${ROOT}"
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

for mode in global tool; do
    "${PY}" reproduction/qvhighlights/scripts/run_inference.py \
        --manifest reproduction/qvhighlights/manifests/pre_smoke_1.json \
        --config reproduction/qvhighlights/configs/pilot.json \
        --mode "${mode}" --api-base "${API}/v1" \
        --output-dir "${RUN_DIR}/longvt_${mode}"
    "${PY}" reproduction/qvhighlights/scripts/evaluate_highlight.py \
        --predictions "${RUN_DIR}/longvt_${mode}/predictions.jsonl" \
        --ground-truth reproduction/qvhighlights/manifests/validation_ground_truth.jsonl \
        --official-repo /path/to/storage/longvt-repro/benchmark_sources/moment_detr \
        --output "${RUN_DIR}/longvt_${mode}/metrics.json"
done

"${PY}" reproduction/qvhighlights/scripts/summarize_experiment.py \
    --global-dir "${RUN_DIR}/longvt_global" \
    --tool-dir "${RUN_DIR}/longvt_tool" \
    --ground-truth reproduction/qvhighlights/manifests/validation_ground_truth.jsonl \
    --official-repo /path/to/storage/longvt-repro/benchmark_sources/moment_detr \
    --output "${RUN_DIR}/summary.json"
