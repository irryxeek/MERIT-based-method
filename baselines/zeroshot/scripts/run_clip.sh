#!/usr/bin/env bash
set -euo pipefail
cd "${ZEROSHOT_ROOT:?Source configs/paths.example.sh first}"
PY="${REPRO_PYTHON:-python}"
ARTIFACTS="${ZEROSHOT_ARTIFACT_ROOT}/runs"
if [[ ! -f "$ARTIFACTS/clip_smoke/completion.json" ]]; then
  "$PY" scripts/run_clip.py --smoke
fi
if [[ ! -f "$ARTIFACTS/clip_smoke/metrics.json" ]]; then
  "$PY" scripts/evaluate.py --run "$ARTIFACTS/clip_smoke" --smoke
fi
if [[ ! -f "$ARTIFACTS/clip_val/completion.json" ]]; then
  "$PY" scripts/run_clip.py
fi
"$PY" scripts/evaluate.py --run "$ARTIFACTS/clip_val"
