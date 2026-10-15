# LongVT-RFT on QVHighlights

This directory contains a training-free QVHighlights highlight-detection adaptation. It freezes the released LongVT-RFT checkpoint and emits one saliency score for every complete, non-overlapping two-second clip. This scoring procedure is an adaptation for this study, not an evaluation protocol from the LongVT paper.

## Fixed Sources

- LongVT commit: `08d755b973e4ad990ac5cdd64fb992c804d840db`
- Model: `longvideotool/LongVT-RFT`, revision `16ed9314e4ea600dc4682b172a614252f2f383ad`
- QVHighlights evaluator: `jayleicn/moment_detr`, commit `b7e553ac3b0c898ee6b85e03ee507c064eab89ca`
- QVHighlights archive: `https://nlp.cs.unc.edu/data/jielei/qvh/qvhilights_videos.tar.gz`
- Python environment: `/path/to/storage/longvt-repro/env/inference`

The model is Qwen2.5-VL, so the service uses `IS_QWEN3_VL=False`, the repository's `tool_call_qwen2_5_vl.jinja`, the Hermes parser, and automatic tool choice.

## Protocol

`configs/pilot.json` is frozen before pilot inference. Seed 42 selects unique videos without inspecting labels or model outputs. The first eight validated videos form smoke/dev; the next 50 form pilot, so the sets are video-disjoint. Ground truth is stored separately from label-free inference manifests.

`longvt_global` uses at most 64 uniformly sampled global frames and no tools. `longvt_tool` receives the identical global sample and may autonomously make at most two `crop_video` observations, each capped at 32 frames. Both predict a model-selected background score plus at most five local-time overrides on an integer 0-4 scale. The strictly validated ranges are deterministically expanded to the complete official vector using the two-second clip midpoint. Crop intervals are observations and are never converted directly into predictions. Repeated local frames count toward tool-mode cost.

One deterministic format retry is allowed. A sample without a valid final vector receives an all-zero fallback and remains in the official-metric denominator. Valid-prediction completion and tool execution success are reported separately. No label field enters model messages.

Inference has two fixed stages for both baselines. First, LongVT produces a concise temporal evidence assessment in its native response format, capped at 768 output tokens; tool mode may autonomously call `crop_video` during this stage. Second, a native-format finalization turn, capped at 512 tokens, predicts one background score for all clips plus at most five local-time overrides in seconds. The adapter strictly validates the ranges and deterministically maps each range to official clips whose midpoint lies inside it; overlapping overrides use their maximum score. The background score is explicitly selected by the model for all other clips, so the adapter does not zero regions merely because they were not cropped. One JSON-Schema-constrained format retry is allowed if validation fails. Combining a response schema with `tools=auto` suppresses tool calls in vLLM 0.12, while constrained long arrays, digit strings, and unbounded segment lists caused value repetition during smoke testing. Clip-index overrides were also rejected because the model copied local seconds from its evidence analysis into index fields. Rejected smoke runs are retained in `results/`.

Two deterministic syntax cleanups are declared in the frozen config and recorded per parse attempt: removing an outer Markdown code fence and appending one missing final object brace only when delimiter counts prove that this is the sole missing delimiter. Truncated arrays, extra segments, invalid times, and semantic errors are never repaired silently.

`manifests/pre_smoke_1.json` is a one-query engineering diagnostic built from an earlier exact YouTube segment download. It is excluded from the official-archive smoke and pilot manifests and is not used for prompt selection or reported pilot performance.

## Data Preparation

The official archive is about 133.7 GiB. It is kept on `/ssd`; the extraction script scans it and extracts only seeded candidates needed for smoke/pilot plus a fixed reserve.

```bash
cd /path/to/workspace/LongVT
PY=/path/to/storage/longvt-repro/env/inference/bin/python

$PY reproduction/qvhighlights/scripts/prepare_manifests.py \
  --annotations /path/to/storage/longvt-repro/benchmark_sources/moment_detr/data/highlight_val_release.jsonl \
  --config reproduction/qvhighlights/configs/pilot.json \
  --output-dir reproduction/qvhighlights/manifests

$PY reproduction/qvhighlights/scripts/prepare_from_archive.py \
  --archive /path/to/storage/longvt-repro/benchmark_data/QVHighlights/archive/qvhilights_videos.tar.gz \
  --candidates reproduction/qvhighlights/manifests/candidate_queue.json \
  --config reproduction/qvhighlights/configs/pilot.json \
  --media-dir /path/to/storage/longvt-repro/benchmark_data/QVHighlights/videos_official \
  --manifest-dir reproduction/qvhighlights/manifests
```

Each extracted file is checked with ffprobe and with decord at its first, middle, and final frame. `media_checks.json` records missing files, duration mismatches, decode failures, and replacements.

For the recorded resumable download, `scripts/wait_and_prepare_media.sh` waits for aria2 to finish, verifies the exact official byte count and gzip stream, and then runs the same selective extraction. It exits without extraction if any download-integrity condition fails.

## Evaluator Validation

```bash
$PY reproduction/qvhighlights/scripts/validate_evaluator.py \
  --official-repo /path/to/storage/longvt-repro/benchmark_sources/moment_detr \
  --ground-truth reproduction/qvhighlights/manifests/validation_ground_truth.jsonl \
  --output reproduction/qvhighlights/results/evaluator_validation/metrics.json
```

The official 1,550-row sample exactly reproduces the published highlight metrics under the official evaluator. Thirteen released example rows are one score short and rely on the evaluator's silent zero-padding behavior; this adapter records that discrepancy and intentionally rejects such output from model runs. A separate ten-row label-only diagnostic confirms that the expected score direction beats its reversal. Synthetic label-derived scores are never used for inference.

## Service

Check GPU 6/7 and port 18006 before launch. Do not run this command while another user's process occupies either GPU.

```bash
bash reproduction/qvhighlights/scripts/start_longvt_server.sh
```

The launcher refuses to start if GPU 6/7 has any compute process or port 18006 is occupied. It never kills a process. TP=2 splits one model replica over GPU 6/7; it is not data parallelism and does not create two replicas. The expanded command and service PID are retained in the script and `results/server/`.

## Inference and Scoring

Use new output directories for reruns so old traces are preserved.

```bash
RUN_ID=pilot_v1
for MODE in global tool; do
  $PY reproduction/qvhighlights/scripts/run_inference.py \
    --manifest reproduction/qvhighlights/manifests/pilot.json \
    --config reproduction/qvhighlights/configs/pilot.json \
    --mode "$MODE" --api-base http://127.0.0.1:18006/v1 \
    --output-dir "reproduction/qvhighlights/results/${RUN_ID}/longvt_${MODE}"

  $PY reproduction/qvhighlights/scripts/evaluate_highlight.py \
    --predictions "reproduction/qvhighlights/results/${RUN_ID}/longvt_${MODE}/predictions.jsonl" \
    --ground-truth reproduction/qvhighlights/manifests/validation_ground_truth.jsonl \
    --official-repo /path/to/storage/longvt-repro/benchmark_sources/moment_detr \
    --output "reproduction/qvhighlights/results/${RUN_ID}/longvt_${MODE}/metrics.json"
done

$PY reproduction/qvhighlights/scripts/summarize_experiment.py \
  --global-dir "reproduction/qvhighlights/results/${RUN_ID}/longvt_global" \
  --tool-dir "reproduction/qvhighlights/results/${RUN_ID}/longvt_tool" \
  --ground-truth reproduction/qvhighlights/manifests/validation_ground_truth.jsonl \
  --official-repo /path/to/storage/longvt-repro/benchmark_sources/moment_detr \
  --output "reproduction/qvhighlights/results/${RUN_ID}/summary.json"
```

The optional budget-matched control uses 66 uniform global frames, defined as the ceiling of the tool pilot's observed mean total frame count (`ceil(65.72)`) without inspecting labels:

```bash
$PY reproduction/qvhighlights/scripts/run_inference.py \
  --manifest reproduction/qvhighlights/manifests/pilot.json \
  --config reproduction/qvhighlights/configs/pilot_global_budget_matched.json \
  --mode global --api-base http://127.0.0.1:18006/v1 \
  --output-dir reproduction/qvhighlights/results/pilot_v1/longvt_global_budget_matched
```

Replace `pilot.json` with `smoke.json` for development. Full validation uses a separate manifest and is not started by this workflow.

After deliberately extracting the complete official video archive, prepare and validate all 1,550 query-video pairs with:

```bash
mkdir -p /path/to/storage/longvt-repro/benchmark_data/QVHighlights/videos_full
tar -xzf /path/to/storage/longvt-repro/benchmark_data/QVHighlights/archive/qvhilights_videos.tar.gz \
  -C /path/to/storage/longvt-repro/benchmark_data/QVHighlights/videos_full

$PY reproduction/qvhighlights/scripts/prepare_full_validation.py \
  --annotations /path/to/storage/longvt-repro/benchmark_sources/moment_detr/data/highlight_val_release.jsonl \
  --media-dir /path/to/storage/longvt-repro/benchmark_data/QVHighlights/videos_full \
  --config reproduction/qvhighlights/configs/pilot.json \
  --output reproduction/qvhighlights/manifests/full_validation.json \
  --media-checks reproduction/qvhighlights/manifests/full_validation_media_checks.json \
  --decode-check
```

Then use `full_validation.json` in the same inference commands with a new run ID. This command is documented but not run in the pilot workflow. The complete archive extraction and 1,550-query inference are explicit prerequisites, not automatic side effects.
