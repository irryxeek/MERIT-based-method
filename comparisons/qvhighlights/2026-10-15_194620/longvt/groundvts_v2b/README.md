# GroundVTS-inspired QVHighlights interval inference

This is a frozen-base, training-free LongVT-RFT inference adaptation. It borrows GroundVTS's highlight-detection representation of temporal spans with three importance levels; it does not implement GroundVTS's trainable VTS module or progressive SFT and is not a GroundVTS reproduction.

## Completed and pending

- **Completed:** the initial continuous-seconds protocol v1 ran its 100-query development pilot. Native-valid output rates were 89% Global and 94% Tool, below the predeclared 95% gate; no v1 full validation was run.
- **Completed:** the large-enum clip-span v2 smoke exposed schema enforcement gaps under vLLM `backend=auto`; invalid output was rejected by the parser and v2 was stopped before pilot.
- **Completed:** v2b now uses a compact regex schema with vLLM XGrammar and `disable_fallback=true`. Its illegal-span server probe passed. Both 8-query smoke conditions completed with 8/8 native-valid outputs and no fallback. Initial global frame hashes match across conditions; all eight Tool crops succeeded and their frames entered subsequent requests.
- **Completed:** both 100-query v2b pilot conditions passed the engineering gate with 100% native-valid output, exact qid coverage, valid vectors, and no transient infrastructure failures. Global had 2 valid all-zero outputs; Tool had 1.
- **Pilot diagnostic only:** VeryGood Global was 15.75 mAP / 29.0 Hit@1; Tool was 16.49 / 29.0. The paired source-video bootstrap delta was +0.74 mAP (95% CI −1.05 to +2.75) and 0 Hit@1 (95% CI −9.00 to +8.42). These 100 development queries are not final validation metrics and were not used to change prompts.
- **Tool pilot behavior:** 99 tool requests, 92 successful crops and 7 local-time boundary rejections; 1 query made no call. All final predictions remained valid, and there were no media/API failures. Tool read 1,784 local frames versus 6,400 global frames in both conditions; its extra visual budget and cost must be considered when interpreting the comparison.
- **Official validation, Global completed:** `results/full_v2b/longvt_global` contains 1,550/1,550 unique validation qids, all successful and native-valid, with zero fallbacks. The official metrics are Fair 32.93/37.87, Good 26.18/36.39, and VeryGood 14.94/28.45 (mAP/Hit@1).
- **Official validation, Tool completed:** 1,550/1,550 unique qids and predictions match the validation manifest; all predictions are native-valid with zero fallbacks. There are 1,443 `success` and 107 `success_with_tool_failure` rows. The 107 rejected tool calls comprise 106 local-duration boundary rejections and one current-video path mismatch (qid 6658); the path guard prevented reading a different query’s clip.
- **Official evaluation and paired analysis completed:** Tool metrics are Fair 33.38/35.81, Good 26.52/33.81, VeryGood 15.12/27.23 (mAP/Hit@1). The 10,000-replicate source-video paired bootstrap and detailed audit are in `analysis/full_v2b/`; see [FINAL_REPORT_ZH.md](FINAL_REPORT_ZH.md) for confidence intervals, costs, evidence coverage, and interpretation.

## v2b protocol and resources

- Base: `longvideotool/LongVT-RFT`, revision `16ed9314e4ea600dc4682b172a614252f2f383ad`, BF16, TP=2, context 65,536.
- Global: 64 uniform frames, no tools. Tool: identical global frames and at most two `crop_video` observations, each up to 32 frames.
- Final JSON contains `clip_span="start:end_exclusive"` over official two-second clip indices and `level` in `very_important/important/less_important`, mapped to `4/3/2`. Other clips score zero; overlapping spans use the maximum. The per-query regex schema admits only legal, nonempty spans for that video's official clip count.
- This clip-grid encoding is a documented amendment after v1's malformed time bounds. It quantizes outputs to full clips and is distinct from the original continuous-seconds/positive-overlap plan and the old score-vector midpoint protocol. See `amendments/V2B_PROTOCOL_CHANGE.md`.
- The train-derived 100-query dev split has no source-video overlap with validation. Inference manifests strip labels. Official validation has 1,550 qids and was inspected in a prior experiment, so it is not an untouched test set.
- Evaluation uses the released `moment_detr` evaluator at commit `b7e553ac3b0c898ee6b85e03ee507c064eab89ca`; Hit@1 keeps official first-`numpy.argmax` tie behavior.
- The current run's server is port 18008, physical GPUs 6/7 only. It serves the exact base revision with the existing Qwen2.5-VL chat template and Hermes tool parser; structured outputs use XGrammar with fallback disabled. The server's verified identity/config is in `results/server/health_v2b.json`.

## Reproduce CPU checks and smoke

Run from the repository root:

```bash
cd /path/to/workspace/LongVT
PY=/path/to/storage/longvt-repro/env/inference/bin/python
QVH=reproduction/qvhighlights/groundvts_inference_v1

$PY -m pytest -q "$QVH/tests"
$PY -m py_compile "$QVH"/scripts/*.py
bash -n "$QVH/scripts/start_server_v2b.sh"
```

The existing service can be checked with:

```bash
$PY "$QVH/scripts/probe_schema_v2b.py" \
  --api-base http://127.0.0.1:18008/v1 \
  --output "$QVH/results/server/schema_v2b_xgrammar_probe_rerun.json"
$PY "$QVH/scripts/probe_schema_v2b_maxitems.py" \
  --api-base http://127.0.0.1:18008/v1 \
  --output "$QVH/results/server/schema_v2b_maxitems_probe_rerun.json"
```

The first probe deliberately requests the illegal `0:150` span under a 75-clip schema and succeeds only if the server returns one different legal span. The max-items probe asks for 76 duplicate entries under `maxItems=75` and requires the response to remain within the parser/schema limit. The first span-probe attempt had a local proxy-client error and is kept separately as `schema_v2b_xgrammar_probe.json`.

To launch a new isolated service, only if GPUs 6/7 and port 18008 are available, run `bash "$QVH/scripts/start_server_v2b.sh"`; it refuses occupied GPUs/port and does not stop processes. Do not relaunch it over the current service.

## Run v2b development pilot

Both commands are resumable with the same output directory and identical config/source hashes:

```bash
for MODE in global tool; do
  $PY "$QVH/scripts/run_interval_inference.py" \
    --manifest "$QVH/manifests/pilot_dev.json" \
    --config "$QVH/configs/protocol_v2b.json" \
    --mode "$MODE" --api-base http://127.0.0.1:18008/v1 \
    --output-dir "$QVH/results/pilot_v2b/longvt_${MODE}"
done

$PY "$QVH/scripts/check_pilot_gate.py" \
  --global-dir "$QVH/results/pilot_v2b/longvt_global" \
  --tool-dir "$QVH/results/pilot_v2b/longvt_tool" \
  --expected 100 --output "$QVH/results/pilot_v2b/engineering_gate.json"
```

The gate requires exact qid coverage, valid score vectors, no unresolved transient errors, and at least 95% native-valid predictions in each condition; parser fallbacks do not count as native-valid. Low metrics alone do not fail the engineering gate. Review all-zero collapse and diagnostics separately. Do not start full validation unless the saved gate passes.

## Full validation commands

The gate has passed and protocol v2b is frozen in `results/server/protocol_freeze_v2b.json`. These commands document how the frozen official run was launched and scored. Global and Tool validation, official evaluation, and paired analysis are completed. Re-running a command with the same output directory resumes configuration-identical terminal rows; do not edit protocol files after the first validation query.

```bash
for MODE in global tool; do
  $PY "$QVH/scripts/run_interval_inference.py" \
    --manifest "$QVH/manifests/validation.json" \
    --config "$QVH/configs/protocol_v2b.json" \
    --mode "$MODE" --api-base http://127.0.0.1:18008/v1 \
    --output-dir "$QVH/results/full_v2b/longvt_${MODE}"
done

for MODE in global tool; do
  $PY reproduction/qvhighlights/scripts/evaluate_highlight.py \
    --predictions "$QVH/results/full_v2b/longvt_${MODE}/predictions.jsonl" \
    --ground-truth "$QVH/manifests/validation_ground_truth.jsonl" \
    --official-repo /path/to/storage/longvt-repro/benchmark_sources/moment_detr \
    --output "$QVH/results/full_v2b/longvt_${MODE}/metrics.json"
done

$PY "$QVH/scripts/summarize_interval.py" \
  --global-dir "$QVH/results/full_v2b/longvt_global" \
  --tool-dir "$QVH/results/full_v2b/longvt_tool" \
  --manifest "$QVH/manifests/validation.json" \
  --ground-truth "$QVH/manifests/validation_ground_truth.jsonl" \
  --output-dir "$QVH/analysis/full_v2b" --bootstrap 10000
```

Each attempt and trace is flushed to JSONL. Infrastructure errors remain pending after the fixed retries; investigate before resuming. A full report must keep full-denominator fallback metrics and per-query paired source-video bootstrap CIs separate from common-valid diagnostics.

## Provenance

`PLAN.md` describes the frozen design and gates. `PROGRESS.md` is an append-only record of completed actions/results; future actions are labeled pending. Input and code hashes are in `input_hashes.json`, `sources.json`, run `config_snapshot.json` files, and server records. The v1 and enum-v2 versions and results remain preserved; no old runner or result was overwritten.
