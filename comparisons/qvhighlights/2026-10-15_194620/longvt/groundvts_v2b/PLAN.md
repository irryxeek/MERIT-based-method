# GroundVTS-style interval inference on QVHighlights

Status: active. Objective and protocol follow `reproduction/NEXT_QVH_GROUNDVTS_INFERENCE_PROMPT_LUNA.md`; this is a frozen-base, training-free inference adaptation, not GroundVTS reproduction or SFT.

## Frozen run design

- Base LongVT-RFT revision `16ed9314e4ea600dc4682b172a614252f2f383ad`, BF16, vLLM TP=2 on physical GPUs 6/7 only.
- Condition A: 64 uniform global frames, tools disabled. Condition B: identical global input, up to 2 `crop_video` calls, up to 32 frames/crop.
- Candidate v2b structured output: `clip_span="start:end_exclusive"` on the official 2-second clip grid, with `very_important=4`, `important=3`, `less_important=2`; empty list means valid all-zero. The regex schema enumerates exactly the legal nonempty spans for the duration, and maps spans directly to score-vector entries. This is an amendment after v1's invalid decimal bounds and differs from the original continuous-time protocol. GT is isolated from inference manifests.
- Official validation: all 1,550 qids. Development: existing train-derived, source-video-grouped 100-qid `pilot_dev`; smoke is first 8 sorted by qid. Official evaluator fixed at `moment_detr` commit `b7e553ac3b0c898ee6b85e03ee507c064eab89ca`.
- Two-stage response: evidence (768 tokens) then schema-constrained interval JSON (2,048 tokens); at most one format-only retry. Temperature 0, top_p 1, seed 42. No prompt selection by validation performance.
- Existing old-protocol results remain frozen and serve only as context, never as same-protocol controls.

## Stop and completion gates

1. CPU protocol tests and evaluator check.
2. Smoke both conditions on 8 rows; verify payloads, history, output vector, and crop path (tool call may naturally be absent).
3. Pilot both conditions on all 100 train-derived development rows. Require 100/100 unique qids, exact score vector length/range, no known leakage/history errors, >=95% first/final schema-valid native predictions excluding fallback, and no unresolved infrastructure failure before full inference. Model output collapse is reported, not relabeled as implementation failure.
4. Freeze manifest/config/prompt/code/service hashes and run both conditions over all 1,550 validation rows. Resume only configuration-identical completed rows. Infrastructure failures stay pending until resolved; no selective answer reruns.
5. Official metrics, per-query paired summary and source-video cluster bootstrap.

## Resource boundary

Never inspect, stop, or use GPU 0–5. Do not terminate/restart existing services. This run's initial server was isolated on GPUs 6/7, port 18007; v2b's strict-XGrammar server is isolated on the same GPUs at port 18008 after gracefully stopping only this run's verified port-18007 process. If either GPU is occupied or service identity cannot be verified, do offline work and pause inference.
