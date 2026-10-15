#!/usr/bin/env bash
set -euo pipefail

ROOT=/path/to/workspace/LongVT
PY=/path/to/storage/longvt-repro/env/inference/bin/python
API=http://127.0.0.1:18006
MANIFEST="${ROOT}/reproduction/qvhighlights/manifests/smoke.json"
RUN_DIR="${ROOT}/reproduction/qvhighlights/results/smoke_v1"

for _ in $(seq 1 480); do
    if [[ -f "${MANIFEST}" ]] && curl --noproxy '*' -fsS "${API}/v1/models" >/dev/null 2>&1; then
        break
    fi
    sleep 30
done

if [[ ! -f "${MANIFEST}" ]]; then
    printf 'Smoke manifest did not become available within four hours\n' >&2
    exit 1
fi
if ! curl --noproxy '*' -fsS "${API}/v1/models" >/dev/null 2>&1; then
    printf 'LongVT service is not ready\n' >&2
    exit 1
fi

sample_count=$("${PY}" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "${MANIFEST}")
if [[ "${sample_count}" -ne 8 ]]; then
    printf 'Expected 8 smoke samples, found %s\n' "${sample_count}" >&2
    exit 1
fi

cd "${ROOT}"
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

for mode in global tool; do
    "${PY}" reproduction/qvhighlights/scripts/run_inference.py \
        --manifest reproduction/qvhighlights/manifests/smoke.json \
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
