# Source this file from Bash, or copy it to paths.local.sh and edit the overrides.
# The GPU list must contain only devices allocated to this experiment.
MERIT_ROOT="${MERIT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
export MERIT_ROOT
export ZEROSHOT_ROOT="${ZEROSHOT_ROOT:-$MERIT_ROOT/baselines/zeroshot}"
export REPRO_DATA_ROOT="${REPRO_DATA_ROOT:-$MERIT_ROOT/data}"
export MERIT_ARTIFACT_ROOT="${MERIT_ARTIFACT_ROOT:-$MERIT_ROOT/artifacts/merit}"
export ZEROSHOT_ARTIFACT_ROOT="${ZEROSHOT_ARTIFACT_ROOT:-$MERIT_ROOT/artifacts/zeroshot}"
export REZE_MODEL_PATH="${REZE_MODEL_PATH:-$MERIT_ROOT/models/Qwen3-VL-8B-Instruct}"
export QVH_EVALUATOR_ROOT="${QVH_EVALUATOR_ROOT:-$MERIT_ROOT/third_party/moment_detr}"
export MRHISUM_SOURCE_ROOT="${MRHISUM_SOURCE_ROOT:-$MERIT_ROOT/third_party/MR.HiSum}"
export QFVS_DATA_ROOT="${QFVS_DATA_ROOT:-$REPRO_DATA_ROOT/QFVS}"
export SVH_INFERENCE_MANIFEST="${SVH_INFERENCE_MANIFEST:-$REPRO_DATA_ROOT/SVHighlights/inference_manifest.jsonl}"
export SVH_PAIRED_RUN="${SVH_PAIRED_RUN:-$REPRO_DATA_ROOT/SVHighlights/paired_videoarm_run}"
export VIDEOARM_ARTIFACT_ROOT="${VIDEOARM_ARTIFACT_ROOT:-$MERIT_ROOT/artifacts/videoarm}"
export LONGVT_ROOT="${LONGVT_ROOT:-$MERIT_ROOT/third_party/LongVT}"
export REPRO_PYTHON="${REPRO_PYTHON:-$(command -v python)}"
export REPRO_GPU_UUIDS="${REPRO_GPU_UUIDS:-}"
export PYTHONPATH="$MERIT_ROOT/src:$MERIT_ROOT/experiments${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_DISABLE_TELEMETRY=1
export VLLM_NO_USAGE_STATS=1
export WANDB_MODE=disabled
