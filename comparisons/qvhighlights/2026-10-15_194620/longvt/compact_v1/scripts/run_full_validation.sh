#!/usr/bin/env bash
set -euo pipefail

ROOT=/path/to/workspace/LongVT
PY=/path/to/storage/longvt-repro/env/inference/bin/python
OFFICIAL_REPO=/path/to/storage/longvt-repro/benchmark_sources/moment_detr
MANIFEST="$ROOT/reproduction/qvhighlights/manifests/full_validation.json"
MEDIA_CHECKS="$ROOT/reproduction/qvhighlights/manifests/full_validation_media_checks.json"
GROUND_TRUTH="$ROOT/reproduction/qvhighlights/manifests/validation_ground_truth.jsonl"
CONFIG="$ROOT/reproduction/qvhighlights/configs/full_validation_global64_tool.json"
RUN_ID="${RUN_ID:-full_validation_v1}"
RUN_DIR="$ROOT/reproduction/qvhighlights/results/$RUN_ID"
API_BASE="${API_BASE:-http://127.0.0.1:18006/v1}"

cd "$ROOT"
mkdir -p "$RUN_DIR"
exec > >(tee -a "$RUN_DIR/run.log") 2>&1

"$PY" - "$MANIFEST" "$MEDIA_CHECKS" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
checks = json.loads(Path(sys.argv[2]).read_text())
if len(manifest) != 1550:
    raise SystemExit(f"expected 1550 manifest rows, found {len(manifest)}")
if len({row["qid"] for row in manifest}) != 1550:
    raise SystemExit("full-validation qids are not unique")
bad = [row for row in checks if row["status"] != "available"]
if bad:
    raise SystemExit(f"media checks contain {len(bad)} unavailable videos")
PY

sha256sum "$CONFIG" "$MANIFEST" "$MEDIA_CHECKS" "$GROUND_TRUTH" \
  reproduction/qvhighlights/scripts/run_inference.py \
  reproduction/qvhighlights/scripts/inference_common.py \
  reproduction/qvhighlights/scripts/evaluate_highlight.py \
  reproduction/qvhighlights/scripts/summarize_experiment.py \
  reproduction/qvhighlights/scripts/audit_paired_experiment.py > "$RUN_DIR/frozen_sha256.txt"
cp "$CONFIG" "$RUN_DIR/config_frozen.json"
cp "$MANIFEST" "$RUN_DIR/manifest_frozen.json"

for MODE in global tool; do
  "$PY" reproduction/qvhighlights/scripts/run_inference.py \
    --manifest "$MANIFEST" \
    --config "$CONFIG" \
    --mode "$MODE" \
    --api-base "$API_BASE" \
    --output-dir "$RUN_DIR/longvt_$MODE"
  "$PY" reproduction/qvhighlights/scripts/evaluate_highlight.py \
    --predictions "$RUN_DIR/longvt_$MODE/predictions.jsonl" \
    --ground-truth "$GROUND_TRUTH" \
    --official-repo "$OFFICIAL_REPO" \
    --output "$RUN_DIR/longvt_$MODE/metrics.json"
done

"$PY" reproduction/qvhighlights/scripts/summarize_experiment.py \
  --global-dir "$RUN_DIR/longvt_global" \
  --tool-dir "$RUN_DIR/longvt_tool" \
  --ground-truth "$GROUND_TRUTH" \
  --official-repo "$OFFICIAL_REPO" \
  --output "$RUN_DIR/summary.json"

"$PY" reproduction/qvhighlights/scripts/audit_paired_experiment.py \
  --summary "$RUN_DIR/summary.json" \
  --global-dir "$RUN_DIR/longvt_global" \
  --tool-dir "$RUN_DIR/longvt_tool" \
  --ground-truth "$GROUND_TRUTH" \
  --output "$RUN_DIR/paired_audit.json" \
  --csv-output "$RUN_DIR/per_question_differences.csv"

date --iso-8601=seconds > "$RUN_DIR/COMPLETED"
