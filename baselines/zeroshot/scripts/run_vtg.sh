#!/usr/bin/env bash
set -euo pipefail
cd "${ZEROSHOT_ROOT:?Source configs/paths.example.sh first}"
PY="${REPRO_PYTHON:-python}"
ARTIFACTS="${ZEROSHOT_ARTIFACT_ROOT}/runs"
if [[ ! -f "$ARTIFACTS/vtg_gpt_smoke/completion.json" ]]; then
  "$PY" scripts/run_vtg_gpt.py --smoke
fi
if [[ ! -f "$ARTIFACTS/vtg_gpt_smoke/metrics.json" ]]; then
  "$PY" scripts/evaluate.py --run "$ARTIFACTS/vtg_gpt_smoke" --smoke
fi
if [[ ! -f "$ARTIFACTS/vtg_gpt_val/completion.json" ]]; then
  "$PY" scripts/run_vtg_gpt.py
fi
"$PY" scripts/evaluate.py --run "$ARTIFACTS/vtg_gpt_val"
