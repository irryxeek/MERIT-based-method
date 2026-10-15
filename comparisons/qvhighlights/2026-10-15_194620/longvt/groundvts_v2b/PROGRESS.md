# Progress log

## 2026-09-26 02:40 Asia/Shanghai — resumed and initialized

- Read the execution prompt, repository handoff, QVHighlights README/full-validation report, and existing manifests/scripts.
- Verified current repository HEAD `bd9dc4f5ad60556aa18028dee4c825676859e494`; working tree has many pre-existing untracked reproduction artifacts. They are preserved.
- Existing same-task full validation is complete only for the old score-vector protocol. It is not reused as an interval-protocol result.
- Selected existing 100-qid `lora_sft/data/pilot_dev.jsonl`; its source-video-grouped overlap audit reports zero dev/validation source-video overlap. Inference manifest will remove all label fields. Official validation manifest has 1,550 rows and media paths resolve under the existing `videos_full/videos` archive (12,562 MP4 files present).
- At setup, physical 6/7 UUIDs matched the frozen handoff and each had ~15 MiB used; no compute process was present. Physical 0–3 hosted an existing TP=4 service and were out of scope. Port 18006 was not listening. The proposed isolated service port was 18007; no process had been stopped at that point.
- Inference environment imports torch 2.9.0+cu128, vLLM 0.12.0, OpenAI client 3.14.1, qwen_vl_utils, numpy 2.2.6, scikit-learn 1.7.2, pytest 8.4.2.
- Input hashes and fixed protocol plan initialized. No new model inference has run yet.

## 2026-09-26 02:55 Asia/Shanghai — offline implementation and evaluator checks complete

- Created isolated label-free manifests: train-derived dev 100, smoke 8, validation 1,550; all media files exist, IDs are unique, and dev/validation overlap is empty under the frozen source-video audit. Inference manifests strip saliency and relevant-window labels.
- Implemented strict interval schema, semantic validation, positive-overlap mapping, retry/resume accounting, two-stage Global/Tool runner, pilot gate, and official-metric/paired-bootstrap analysis in this directory. No old runner/results were modified.
- Focused CPU checks: 18 passed; Python compilation and `bash -n` passed.
- Official evaluator's released 1,550-row sample reproduced every metric exactly; 13 sample rows rely on evaluator silent padding, so our prediction validator remains strict.
- Re-evaluated the frozen old-protocol full results and reproduced the stored metrics exactly. Running the new summary path on those old predictions also reproduced the historical metrics and zero initial-frame mismatches. These are evaluator implementation checks only, not results for the interval protocol.
- Resource preflight reconfirmed GPU 6/7 idle with fixed UUIDs and the previously observed GPU 0–3 TP=4 service out of scope. The new experiment server was isolated on port 18007; no pre-existing process was stopped.
- No interval-protocol model inference has completed yet.

## 2026-09-26 03:50 Asia/Shanghai — v2b offline checks and strict service launch

- Added an exact regex schema for every legal nonempty `start_clip:end_clip_exclusive` span for durations up to 75 official clips. Exhaustive CPU comparison against the legal span set passed for clip counts 1–75.
- CPU test suite now reports 28 passed; Python compilation, v2b launcher `bash -n`, and `git diff --check` also passed.
- Added a repeatable 75-clip schema probe that asks the service for an illegal `0:150` range and requires exactly one different, parser-valid span.
- Verified PID 27503 from this run's port-18007 PID file and listener command line, sent it SIGTERM, and observed it exit. GPU 6/7 returned to 15 MiB each. No other service was stopped.
- Started this run's v2b LongVT-RFT service PID 26273 on port 18008 using the frozen model revision, BF16, TP=2, context 65,536, Qwen2.5-VL template, Hermes parser, XGrammar, and `disable_fallback=true`. vLLM startup logs confirmed the structured-output configuration; the readiness endpoint became healthy. No v2b dataset inference had started at this timestamp.

## 2026-09-26 03:53 Asia/Shanghai — strict schema probe passed

- The first probe attempt failed in the probe client's proxy initialization (`socksio` was unavailable); it did not reach the server. The failure record is preserved separately.
- Updated the probe client to disable environment proxies, matching the runner. The retry requested illegal 75-clip span `0:150`; XGrammar returned `0:15`, and the parser accepted the one returned legal span. `passed=true` is recorded in `results/server/schema_v2b_xgrammar_probe_retry1.json`.
- The 2,884-character schema pattern was accepted by the configured server. Smoke inference is the next planned stage; no v2b dataset output exists yet.

## 2026-09-26 04:02 Asia/Shanghai — v2b smoke and schema constraints verified

- Under the strict XGrammar service, the illegal-span probe returned one parser-valid legal value (`0:15`) after being prompted for illegal `0:150`.
- The max-items stress probe requested 76 entries under the production 75-clip schema; the server returned exactly 75 parser-valid entries. Both schema probes passed.
- Global and Tool smoke each completed all 8 qids, with 8/8 native-valid outputs and no fallbacks. All score-vector lengths/values passed; exactly one Global output was all-zero and none in Tool.
- Initial global-frame hashes matched for all 8 paired qids. Tool made 8 successful crop calls; all crop frame timestamps stayed inside official local durations, all returned frames were recorded as entering the next request, and frame counts were 2–32.
- One first-pass audit artifact marked the score-vector check false because its local audit expression required all four levels to occur. That audit is preserved as `analysis/smoke_v2b_audit_attempt1.json`; corrected validation is `analysis/smoke_v2b.json` and passes. No inference output changed.
- Saved `results/server/protocol_candidate_v2b.json` with current source/config/prompt/manifest hashes before pilot. The 100-query v2b pilot has not started yet.

## 2026-09-26 04:03 Asia/Shanghai — v2b Global pilot running

- Started the 100-qid Global pilot with `protocol_v2b.json` and the frozen candidate files. At first checkpoint, qids 59, 67, and 74 had completed successfully with no tools. The run is still in progress; Tool pilot has not started.

## 2026-09-26 04:06 Asia/Shanghai — v2b Global pilot checkpoint

- 24/100 qids have durable traces; 24 are native-valid and 0 have non-success terminal status. The run remains active; Tool pilot has not started.

## 2026-09-26 04:09 Asia/Shanghai — v2b Global pilot checkpoint

- 41/100 qids have durable traces; 41 are native-valid, 0 are valid all-zero outputs, and 0 have non-success terminal status. The run remains active; Tool pilot has not started.

## 2026-09-26 04:11 Asia/Shanghai — v2b Global pilot checkpoint

- Durable trace count: 49/100; native-valid: 49; valid all-zero: 0; non-success status: 0. Tool pilot has not started.

## 2026-09-26 04:12 Asia/Shanghai — v2b Global pilot checkpoint

- Durable trace count: 62/100; native-valid: 62; valid all-zero: 1; non-success status: 0. Tool pilot has not started.

## 2026-09-26 04:14 Asia/Shanghai — v2b Global pilot checkpoint

- Durable trace count: 76/100; native-valid: 76; valid all-zero: 1; non-success status: 0. Tool pilot has not started.

## 2026-09-26 04:16 Asia/Shanghai — v2b Global pilot checkpoint

- Durable trace count: 88/100; native-valid: 88; valid all-zero: 2; non-success status: 0. Tool pilot has not started.

## 2026-09-26 04:19 Asia/Shanghai — v2b Tool pilot running

- Global pilot completed 100/100 with exact qid coverage, 100 native-valid predictions, zero fallback/vector/transient errors, and 2 legal all-zero vectors. The 100-qid Tool pilot has started; its first two qids completed successfully with one crop each.

## 2026-09-26 04:20 Asia/Shanghai — v2b Tool pilot checkpoint

- Durable traces: 7/100; native-valid: 7; terminal non-success: 0; crop calls: 7; successful crops: 7. Tool pilot remains active.

## 2026-09-26 04:22 Asia/Shanghai — v2b Tool pilot checkpoint

- Durable traces: 14/100; native-valid: 14; terminal non-success: 0; crop calls: 14; successful crops: 14. Tool pilot remains active.

## 2026-09-26 04:23 Asia/Shanghai — v2b Tool pilot checkpoint

- Durable traces: 19/100; native-valid: 19; crop calls: 19; successful crops: 18; failed crop attempts: 1. The observed failure was a model-requested start at the exact official duration and was rejected by the duration guard before decoder access; the final prediction remained valid. Tool pilot is active.

## 2026-09-26 04:25 Asia/Shanghai — v2b Tool pilot checkpoint

- Durable traces: 27/100; native-valid: 27; valid all-zero: 0; crop calls: 27; successful crops: 26; invalid/failed crop attempts: 1. Tool pilot is active.

## 2026-09-26 04:29 Asia/Shanghai — v2b Tool pilot checkpoint

- Durable traces: 43/100; native-valid: 43; valid all-zero: 0; crop calls: 43; successful crops: 42; failed crop attempts: 1. Tool pilot remains active.

## 2026-09-26 04:30 Asia/Shanghai — v2b Tool pilot checkpoint

- Durable traces: 49/100; native-valid: 49; crop calls: 49; successful crops: 47; failed crop attempts: 2. Both observed failures so far were model-requested starts at 150.0s on 150s clips and were rejected by the local guard before decode; final answers remained valid.

## 2026-09-26 04:32 Asia/Shanghai — v2b Tool pilot checkpoint

- Durable traces: 56/100; native-valid: 56; valid all-zero: 0; crop calls: 56; successful crops: 53; failed crop attempts: 3. All 3 failures to this point are the same model-requested 150s boundary start; no media decode/API failure has occurred.

## 2026-09-26 04:33 Asia/Shanghai — v2b Tool pilot checkpoint

- Durable traces: 63/100; native-valid: 63; valid all-zero: 0; crop calls: 63; successful crops: 59; boundary-rejected crop attempts: 4. The same exact-duration boundary pattern persists; each affected final answer is still valid.

## 2026-09-26 04:34 Asia/Shanghai — v2b Tool pilot checkpoint

- Durable traces: 69/100; native-valid: 69; crop calls: 69; successful crops: 64; boundary-rejected attempts: 5. Failure starts include official end (150s) and source-timeline values (e.g., 290s) outside the local 150s segment; all were rejected before decode.

## 2026-09-26 04:36 Asia/Shanghai — v2b Tool pilot checkpoint

- Durable traces: 76/100; native-valid: 76; crop calls: 76; successful crops: 71; guard-rejected crop calls: 5. Tool pilot remains active; no transient service/media failure observed.

## 2026-09-26 04:38 Asia/Shanghai — v2b Tool pilot checkpoint

- Durable traces: 83/100; native-valid: 83; valid all-zero: 0; samples with no tool call: 1; crop calls: 82; guard-rejected calls: 5. Run remains active without transient failures.

## 2026-09-26 04:39 Asia/Shanghai — v2b Tool pilot checkpoint

- Durable traces: 89/100; native-valid: 89; crop calls: 88; successful crops: 81; guard-rejected calls: 7. Both most recent tool failures repeated the 150s boundary-start pattern; final outputs were valid.

## 2026-09-26 04:41 Asia/Shanghai — v2b Tool pilot checkpoint

- Durable traces: 95/100; native-valid: 95; valid all-zero: 1; tool calls: 94; successful crops: 87; boundary-rejected calls: 7. Tool pilot remains active.

## 2026-09-26 04:45 Asia/Shanghai — v2b pilot analysis and official-run freeze

- Both 100-query conditions passed the engineering gate: exact qid coverage, 100/100 native-valid predictions, zero vector errors, and zero transient infrastructure errors. Valid all-zero counts were Global 2/100 and Tool 1/100.
- Official dev metrics: Global VeryGood 15.75 mAP / 29.0 Hit@1; Tool 16.49 / 29.0. Paired source-video bootstrap (10,000 replicates, 33 clusters, seed 42) gave Tool−Global VeryGood mAP +0.74 (95% CI −1.05 to +2.75) and Hit@1 0.00 (95% CI −9.00 to +8.42). These are pilot diagnostics, not final validation results; no prompt/config was selected from them.
- Tool diagnostics: 99 requests, 92 successful crops, 7 guard-rejected local-time boundary starts, 1 query without a tool call; all 100 final answers were native-valid. Successful crop frames were recorded in subsequent messages; paired global-frame hashes matched on all queries.
- Frozen v2b config, prompts, schema/mapping implementation, code, manifests, evaluator, and service identity in `results/server/protocol_freeze_v2b.json`. Official validation is next and was not yet started at this log entry.

## 2026-09-26 04:46 Asia/Shanghai — official validation Global started

- Started the frozen v2b Global run on all 1,550 official validation qids in `results/full_v2b/longvt_global`. The first three streamed rows (2579, 5071, 5342) completed successfully. Tool validation has not started. Per-qid traces and attempts are flushed as the run proceeds.

## 2026-09-26 04:48 Asia/Shanghai — official Global checkpoint

- Durable traces: 15/1,550; native-valid: 15; fallbacks: 0; statuses: `{"success": 15}`. Run remains active; official Tool has not started.

## 2026-09-26 04:50 Asia/Shanghai — official Global checkpoint

- Durable traces: 27/1,550; native-valid: 27; fallbacks: 0; statuses: `{"success": 27}`. Run remains active.

## 2026-09-26 04:52 Asia/Shanghai — official Global checkpoint

- Durable traces: 39/1,550; native-valid: 39; fallbacks: 0; statuses: `{"success": 39}`. Run remains active.

## 2026-09-26 04:53 Asia/Shanghai — official Global checkpoint

- Durable traces: 52/1,550; native-valid: 52; fallbacks: 0; statuses: `{"success": 52}`. Run remains active.

## 2026-09-26 04:56 Asia/Shanghai — official Global checkpoint

- Durable traces: 65/1,550; native-valid: 65; fallbacks: 0; statuses: `{"success": 65}`. Run remains active.

## 2026-09-26 04:58 Asia/Shanghai — official Global checkpoint

- Durable traces: 76/1,550; native-valid: 76; fallbacks: 0; statuses: `{"success": 76}`. Run remains active.

## 2026-09-26 04:59 Asia/Shanghai — official Global checkpoint

- Durable traces: 90/1,550; native-valid: 90; fallbacks: 0; statuses: `{"success": 90}`. Run remains active.

## 2026-09-26 05:01 Asia/Shanghai — official Global checkpoint

- Durable traces: 104/1,550; native-valid: 104; fallbacks: 0; statuses: `{"success": 104}`. Run remains active.

## 2026-09-26 05:03 Asia/Shanghai — official Global checkpoint

- Durable traces: 113/1,550; native-valid: 113; fallbacks: 0; statuses: `{"success": 113}`. A recent slow query completed successfully; no intervention was needed.

## 2026-09-26 05:05 Asia/Shanghai — official Global checkpoint

- Durable traces: 129/1,550; native-valid: 129; fallbacks: 0; statuses: `{"success": 129}`. Run remains active.

## 2026-09-26 05:08 Asia/Shanghai — official Global checkpoint

- Durable traces: 145/1,550; native-valid: 145; fallbacks: 0; statuses: `{"success": 145}`. Run remains active.

## 2026-09-26 05:10 Asia/Shanghai — official Global checkpoint

- Durable traces: 162/1,550; native-valid: 162; fallbacks: 0; statuses: `{"success": 162}`. Run remains active.

## 2026-09-26 05:13 Asia/Shanghai — official Global checkpoint

- Durable traces: 181/1,550; native-valid: 181; fallbacks: 0; statuses: `{"success": 181}`. Run remains active.

## 2026-09-26 05:15 Asia/Shanghai — official Global checkpoint

- Durable traces: 194/1,550; native-valid: 194; fallbacks: 0; statuses: `{"success": 194}`. Run remains active.

## 2026-09-26 05:18 Asia/Shanghai — official Global checkpoint

- Durable traces: 212/1,550; native-valid: 212; fallbacks: 0; statuses: `{"success": 212}`. Run remains active.

## 2026-09-26 05:20 Asia/Shanghai — official Global checkpoint

- Durable traces: 231/1,550; native-valid: 231; fallbacks: 0; statuses: `{"success": 231}`. Run remains active.

## 2026-09-26 05:23 Asia/Shanghai — official Global checkpoint

- Durable traces: 250/1,550; native-valid: 250; fallbacks: 0; statuses: `{"success": 250}`. Run remains active.

## 2026-09-26 05:26 Asia/Shanghai — official Global checkpoint

- Durable traces: 269/1,550; native-valid: 269; fallbacks: 0; statuses: `{"success": 269}`. Run remains active.

## 2026-09-26 05:28 Asia/Shanghai — official Global checkpoint

- Durable traces: 285/1,550; native-valid: 285; fallbacks: 0; statuses: `{"success": 285}`. Run remains active.

## 2026-09-26 05:31 Asia/Shanghai — official Global checkpoint

- Durable traces: 305/1,550; native-valid: 305; fallbacks: 0; statuses: `{"success": 305}`. Run remains active.

## 2026-09-26 05:34 Asia/Shanghai — official Global checkpoint

- Durable traces: 323/1,550; native-valid: 323; fallbacks: 0; statuses: `{"success": 323}`. Run remains active.

## 2026-09-26 05:36 Asia/Shanghai — official Global checkpoint

- Durable traces: 339/1,550; native-valid: 339; fallbacks: 0; statuses: `{"success": 339}`. Run remains active.

## 2026-09-26 05:39 Asia/Shanghai — official Global checkpoint

- Durable traces: 358/1,550; native-valid: 358; fallbacks: 0; statuses: `{"success": 358}`. Run remains active.

## 2026-09-26 05:41 Asia/Shanghai — official Global checkpoint

- Durable traces: 374/1,550; native-valid: 374; fallbacks: 0; statuses: `{"success": 374}`. Run remains active.

## 2026-09-26 05:44 Asia/Shanghai — official Global checkpoint

- Durable traces: 392/1,550; native-valid: 392; fallbacks: 0; statuses: `{"success": 392}`. Run remains active.

## 2026-09-26 05:46 Asia/Shanghai — official Global checkpoint

- Durable traces: 407/1,550; native-valid: 407; fallbacks: 0; statuses: `{"success": 407}`. Run remains active.

## 2026-09-26 05:49 Asia/Shanghai — official Global checkpoint

- Durable traces: 423/1,550; native-valid: 423; fallbacks: 0; statuses: `{"success": 423}`. Run remains active.

## 2026-09-26 05:51 Asia/Shanghai — official Global checkpoint

- Durable traces: 440/1,550; native-valid: 440; fallbacks: 0; statuses: `{"success": 440}`. Run remains active.

## 2026-09-26 05:54 Asia/Shanghai — official Global checkpoint

- Durable traces: 454/1,550; native-valid: 454; fallbacks: 0; statuses: `{"success": 454}`. Run remains active.

## 2026-09-26 05:56 Asia/Shanghai — official Global checkpoint

- Durable traces: 468/1,550; native-valid: 468; fallbacks: 0; statuses: `{"success": 468}`. Run remains active.

## 2026-09-26 05:58 Asia/Shanghai — official Global checkpoint

- Durable traces: 484/1,550; native-valid: 484; fallbacks: 0; statuses: `{"success": 484}`. Run remains active.

## 2026-09-26 06:01 Asia/Shanghai — official Global checkpoint

- Durable traces: 497/1,550; native-valid: 497; fallbacks: 0; statuses: `{"success": 497}`. Run remains active.

## 2026-09-26 06:03 Asia/Shanghai — official Global checkpoint

- Durable traces: 514/1,550; native-valid: 514; fallbacks: 0; statuses: `{"success": 514}`. Run remains active.

## 2026-09-26 06:05 Asia/Shanghai — official Global checkpoint

- Durable traces: 532/1,550; native-valid: 532; fallbacks: 0; statuses: `{"success": 532}`. Run remains active.

## 2026-09-26 06:08 Asia/Shanghai — official Global checkpoint

- Durable traces: 545/1,550; native-valid: 545; fallbacks: 0; statuses: `{"success": 545}`. Run remains active.

## 2026-09-26 06:10 Asia/Shanghai — official Global checkpoint

- Durable traces: 560/1,550; native-valid: 560; fallbacks: 0; statuses: `{"success": 560}`. Run remains active.

## 2026-09-26 06:13 Asia/Shanghai — official Global checkpoint

- Durable traces: 576/1,550; native-valid: 576; fallbacks: 0; statuses: `{"success": 576}`. Run remains active.

## 2026-09-26 06:15 Asia/Shanghai — official Global checkpoint

- Durable traces: 594/1,550; native-valid: 594; fallbacks: 0; statuses: `{"success": 594}`. Run remains active.

## 2026-09-26 06:17 Asia/Shanghai — official Global checkpoint

- Durable traces: 612/1,550; native-valid: 612; fallbacks: 0; statuses: `{"success": 612}`. Run remains active.

## 2026-09-26 06:20 Asia/Shanghai — official Global checkpoint

- Durable traces: 627/1,550; native-valid: 627; fallbacks: 0; statuses: `{"success": 627}`. Run remains active.

## 2026-09-26 06:22 Asia/Shanghai — official Global checkpoint

- Durable traces: 642/1,550; native-valid: 642; fallbacks: 0; statuses: `{"success": 642}`. Run remains active.

## 2026-09-26 06:25 Asia/Shanghai — official Global checkpoint

- Durable traces: 652/1,550; native-valid: 652; fallbacks: 0; statuses: `{"success": 652}`. Run remains active.

## 2026-09-26 06:27 Asia/Shanghai — official Global checkpoint

- Durable traces: 666/1,550; native-valid: 666; fallbacks: 0; statuses: `{"success": 666}`. Run remains active.

## 2026-09-26 06:30 Asia/Shanghai — official Global checkpoint

- Durable traces: 682/1,550; native-valid: 682; fallbacks: 0; statuses: `{"success": 682}`. Run remains active.

## 2026-09-26 06:32 Asia/Shanghai — official Global checkpoint

- Durable traces: 694/1,550; native-valid: 694; fallbacks: 0; statuses: `{"success": 694}`. Run remains active.

## 2026-09-26 06:34 Asia/Shanghai — official Global checkpoint

- Durable traces: 708/1,550; native-valid: 708; fallbacks: 0; statuses: `{"success": 708}`. Run remains active.

## 2026-09-26 06:37 Asia/Shanghai — official Global checkpoint

- Durable traces: 725/1,550; native-valid: 725; fallbacks: 0; statuses: `{"success": 725}`. Run remains active.

## 2026-09-26 06:39 Asia/Shanghai — official Global checkpoint

- Durable traces: 741/1,550; native-valid: 741; fallbacks: 0; statuses: `{"success": 741}`. Run remains active.

## 2026-09-26 06:42 Asia/Shanghai — official Global checkpoint

- Durable traces: 754/1,550; native-valid: 754; fallbacks: 0; statuses: `{"success": 754}`. Run remains active.

## 2026-09-26 06:45 Asia/Shanghai — official Global checkpoint

- Durable traces: 769/1,550; native-valid: 769; fallbacks: 0; statuses: `{"success": 769}`. Run remains active.

## 2026-09-26 06:47 Asia/Shanghai — official Global checkpoint

- Durable traces: 785/1,550; native-valid: 785; fallbacks: 0; statuses: `{"success": 785}`. Run remains active.

## 2026-09-26 06:49 Asia/Shanghai — official Global checkpoint

- Durable traces: 800/1,550; native-valid: 800; fallbacks: 0; statuses: `{"success": 800}`. Run remains active.

## 2026-09-27 14:26 Asia/Shanghai — official Global checkpoint (resumed after user steering)

- Durable traces: 1550/1,550; native-valid: 1550; fallbacks: 0; statuses: `{"success": 1550}`. Run remains active.

## 2026-09-27 14:28 Asia/Shanghai — official Global validation completed

- Global traces/predictions cover all 1,550 unique validation qids; all 1,550 statuses are `success`, all are native-valid, and there are zero fallbacks.
- Official metrics (mAP / Hit@1): Fair 32.93 / 37.87; Good 26.18 / 36.39; VeryGood 14.94 / 28.45.
- The official evaluator wrote `results/full_v2b/longvt_global/metrics.json`. Current serving model and GPU 6/7 UUIDs were rechecked before Tool inference. Tool validation and paired analysis remain pending.

## 2026-09-27 14:29 Asia/Shanghai — official Tool checkpoint (active)

- Durable traces: 3/1,550; native-valid: 3; fallbacks: 0; statuses: `{"success": 3}`. Tool inference remains active; no full-set conclusion is available yet.

## 2026-09-27 14:30 Asia/Shanghai — official Tool checkpoint (active)

- Durable traces: 10/1,550; native-valid: 10; fallbacks: 0; statuses: `{"success": 10}`. Latest completed qids: `[5979, 6223, 8737, 8783]`.
- The Tool process remains active; metrics and full-set conclusions are pending.

## 2026-09-27 14:32 Asia/Shanghai — official Tool checkpoint (active)

- Durable traces: 15/1,550; native-valid: 15; fallbacks: 0; statuses: `{"success": 15}`.
- Run remains active. Paired audit command and required inputs were checked; no final metrics are available until Tool coverage is complete.

## 2026-09-27 14:33 Asia/Shanghai — official Tool checkpoint (active)

- Durable unique traces: 21/1,550; native-valid: 21; fallbacks: 0; statuses: `{"success": 21}`; recorded tool calls: 22.
- Tool runner remains active. Predictions are assembled after the run; current inference artifacts are the append-only attempts and traces.

## 2026-09-27 14:35 Asia/Shanghai — official Tool checkpoint (active)

- Durable unique traces: 27/1,550; native-valid: 27; fallbacks: 0; statuses: `{"success": 26, "success_with_tool_failure": 1}`; failed crop actions: 1; transient infrastructure statuses: 0.
- Inspected qid 8014: requested crop begins at 150.0 s, outside its official 150 s local duration; guard rejected the crop, while the final prediction remained native-valid with no fallback. This is a tool-action boundary failure, not an infrastructure error.

## 2026-09-27 14:36 Asia/Shanghai — official Tool checkpoint (active)

- Durable unique traces: 36/1,550; native-valid: 36; fallbacks: 0; statuses: `{"success": 35, "success_with_tool_failure": 1}`; failed crop actions: 1; transient infrastructure statuses: 0.
- Inference remains active; final metrics and the full cost/coverage summary are pending.

## 2026-09-27 14:38 Asia/Shanghai — official Tool checkpoint (active)

- Durable traces: 42/1,550 unique qids; native-valid: 42; fallbacks: 0; statuses: `{"success": 41, "success_with_tool_failure": 1}`; failed crop actions: 1; transient infrastructure statuses: 0.

## 2026-09-27 14:40 Asia/Shanghai — official Tool checkpoint (active)

- Durable traces: 49/1,550 unique qids; native-valid: 49; fallbacks: 0; statuses: `{"success": 48, "success_with_tool_failure": 1}`; failed crop actions: 1; transient infrastructure statuses: 0. Run remains active.

## 2026-09-27 14:41 Asia/Shanghai — official Tool checkpoint (active)

- Durable unique traces: 53/1,550; native-valid: 53; fallbacks: 0; statuses: `{"success": 51, "success_with_tool_failure": 2}`; failed crop actions: 2; transient infrastructure statuses: 0.
- Inspected crop boundary rejections are qids 8014 and 7735; each produced a native-valid final prediction without fallback. Both requests used source-video absolute starts equal to the official local duration and were rejected by the local-time guard.

## 2026-09-27 14:42 Asia/Shanghai — official Tool checkpoint (active)

- Durable unique traces: 59/1,550; native-valid: 59; fallbacks: 0; statuses: `{"success": 57, "success_with_tool_failure": 2}`; tool requests: 60; failed crop actions: 2; transient infrastructure statuses: 0. Run remains active.

## 2026-09-27 14:44 Asia/Shanghai — official Tool checkpoint (active)

- Durable unique traces: 67/1,550; native-valid: 67; fallbacks: 0; statuses: `{"success": 65, "success_with_tool_failure": 2}`; tool requests: 68; failed crop actions: 2; transient infrastructure statuses: 0.
- Run remains active; only append-only traces and attempts are durable until final prediction aggregation.

## 2026-09-27 14:46 Asia/Shanghai — official Tool checkpoint (active)

- Durable unique traces: 76/1,550; native-valid: 76; fallbacks: 0; statuses: `{"success": 73, "success_with_tool_failure": 3}`; tool requests: 77; failed crop actions: 3; transient infrastructure statuses: 0.
- Run remains active.

## 2026-09-27 14:46 Asia/Shanghai — Tool boundary rejection detail

- qid 1136 requested crop `[210,360]`; the start is outside the official local interval, so the crop guard rejected it. The sample still has a native-valid prediction with no fallback. This is the third recorded local-time boundary rejection in the active Tool run.

## 2026-09-27 14:47 Asia/Shanghai — Tool boundary rejection detail

- qid 5510 requested start/end at 150 s on a clip whose official local duration is 150 s. The crop-start guard rejected it; the final prediction is native-valid with no fallback. This is the fourth recorded local-time boundary rejection.

## 2026-09-27 14:48 Asia/Shanghai — official Tool checkpoint (active)

- Durable unique traces: 87/1,550; native-valid: 87; fallbacks: 0; statuses: `{"success": 83, "success_with_tool_failure": 4}`; tool requests: 87; failed crop actions: 4; transient infrastructure statuses: 0. Run remains active.

## 2026-09-27 14:50 Asia/Shanghai — official Tool checkpoint (active)

- Durable unique traces: 94/1,550; native-valid: 94; fallbacks: 0; statuses: `{"success": 90, "success_with_tool_failure": 4}`; tool requests: 94; failed crop actions: 4; transient infrastructure statuses: 0. Run remains active.

## 2026-09-27 14:52 Asia/Shanghai — official Tool checkpoint (active)

- Durable unique traces: 102/1,550; native-valid: 102; fallbacks: 0; statuses: `{"success": 98, "success_with_tool_failure": 4}`; tool requests: 102; failed crop actions: 4; transient infrastructure statuses: 0. Run remains active.

## 2026-09-27 14:53 Asia/Shanghai — official Tool checkpoint (active)

- Durable unique traces: 110/1,550; native-valid: 110; fallbacks: 0; statuses: `{"success": 106, "success_with_tool_failure": 4}`; tool requests: 110; failed crop actions: 4; transient infrastructure statuses: 0. Run remains active.

## 2026-09-27 14:55 Asia/Shanghai — Tool boundary rejection detail

- qid 3834 requested crop `[150,170]` in local clip `x4kKIm8WWOA_60.0_210.0.mp4`; start 150 s exceeds the clip local duration, so the guard rejected it. Final prediction is native-valid with no fallback. This is the fifth recorded crop boundary rejection.

## 2026-09-27 14:57 Asia/Shanghai — official Tool checkpoint (active)

- Durable unique traces: 127/1,550; native-valid: 127; fallbacks: 0; statuses: `{"success": 122, "success_with_tool_failure": 5}`; tool requests: 127; failed crop actions: 5; transient infrastructure statuses: 0.
- Final evaluation and paired diagnostics remain planned after complete coverage.

## 2026-09-27 14:58 Asia/Shanghai — Tool boundary rejection detail

- qid 3793 requested `[210,360]` against a clip whose official local start is 210 s; the local-time guard rejected the out-of-range crop start. Prediction remained native-valid with no fallback. This is the sixth recorded crop boundary rejection.

## 2026-09-27 14:59 Asia/Shanghai — Tool boundary rejection detail

- qid 2253 requested `[150,155]` on a 150 s local clip. The crop guard rejected the out-of-range start; the final prediction is native-valid without fallback. This is the seventh recorded boundary rejection.

## 2026-09-27 15:00 Asia/Shanghai — official Tool checkpoint (active)

- Durable unique traces: 139/1,550; native-valid: 139; fallbacks: 0; statuses: `{"success": 132, "success_with_tool_failure": 7}`; tool requests: 139; failed crop actions: 7; transient infrastructure statuses: 0.
- Run remains active; final prediction materialization, official metrics and paired analysis have not yet run for Tool.

## 2026-09-27 15:02 Asia/Shanghai — official Tool checkpoint (active)

- Durable unique traces: 147/1,550; native-valid: 147; fallbacks: 0; statuses: `{"success": 139, "success_with_tool_failure": 8}`; tool requests: 147; failed crop actions: 8; transient infrastructure statuses: 0.
- Run remains active.

## 2026-09-27 15:02 Asia/Shanghai — Tool boundary rejection detail

- qid 9562 requested a crop beginning at 150 s on a clip with 150 s official local duration. The local-time guard rejected it; its final prediction remains native-valid with no fallback. This is the eighth recorded crop boundary rejection.

## 2026-09-27 15:04 Asia/Shanghai — official Tool checkpoint (active)

- Durable unique traces: 155/1,550; native-valid: 155; fallbacks: 0; statuses: `{"success": 147, "success_with_tool_failure": 8}`; tool requests: 155; failed crop actions: 8; transient infrastructure statuses: 0. Run remains active.

## 2026-09-27 15:06 Asia/Shanghai — Tool boundary rejection detail

- qid 7604 requested `[150,160]` on a clip with 150 s official local duration. The guard rejected the out-of-range start; the answer is native-valid without fallback. This is the ninth recorded crop boundary rejection.

## 2026-09-27 15:07 Asia/Shanghai — official Tool checkpoint (active)

- Durable unique traces: 169/1,550; native-valid: 169; fallbacks: 0; statuses: `{"success": 160, "success_with_tool_failure": 9}`; tool requests: 169; failed crop actions: 9; transient infrastructure statuses: 0. Run remains active.

## 2026-09-27 15:09 Asia/Shanghai — official Tool checkpoint (active)

- Durable unique traces: 177/1,550; native-valid: 177; fallbacks: 0; statuses: `{"success": 168, "success_with_tool_failure": 9}`; tool requests: 177; failed crop actions: 9; transient infrastructure statuses: 0.
- Run remains active.

## 2026-09-27 15:11 Asia/Shanghai — official Tool checkpoint (active)

- Durable unique traces: 186/1,550; native-valid: 186; fallbacks: 0; statuses: `{"success": 177, "success_with_tool_failure": 9}`; tool requests: 186; failed crop actions: 9; transient infrastructure statuses: 0. Run remains active.

## 2026-09-27 15:12 Asia/Shanghai — Tool boundary rejection detail

- qid 525 requested crop `[160,170]` on `XpVt6Z1Gjjo_360.0_510.0.mp4`; start 160 s exceeds the official 150 s local duration. Guard rejected the crop; prediction remained native-valid with no fallback. This is the tenth recorded crop boundary rejection.

## 2026-09-27 15:14 Asia/Shanghai — official Tool checkpoint (active)

- Durable unique traces: 200/1,550; native-valid: 200; fallbacks: 0; statuses: `{"success": 189, "success_with_tool_failure": 11}`; tool requests: 200; failed crop actions: 11; transient infrastructure statuses: 0.
- qid 5703 requested `[350,360]`, beyond the 150 s local duration; the crop guard rejected it and the prediction remained valid. This is the eleventh recorded boundary rejection.

## 2026-09-27 15:15 Asia/Shanghai — Tool boundary rejection detail

- qid 5690 requested `[480,510]` on a clip whose official local duration is 150 s. The crop guard rejected the out-of-range start; the prediction is native-valid without fallback. This is the twelfth recorded boundary rejection.

## 2026-09-27 15:17 Asia/Shanghai — official Tool checkpoint (active)

- Durable unique traces: 212/1,550; native-valid: 212; fallbacks: 0; statuses: `{"success": 200, "success_with_tool_failure": 12}`; tool requests: 212; failed crop actions: 12; transient infrastructure statuses: 0. Run remains active.

## 2026-09-27 15:19 Asia/Shanghai — official Tool checkpoint (active)

- Durable unique traces: 220/1,550; native-valid: 220; fallbacks: 0; statuses: `{"success": 208, "success_with_tool_failure": 12}`; tool requests: 220; failed crop actions: 12; transient infrastructure statuses: 0. Run remains active.

## 2026-09-27 15:20 Asia/Shanghai — Tool boundary rejection detail

- qid 4831 requested `[150,160]` for a clip with a 150 s local duration. The crop guard rejected it; final prediction was native-valid with no fallback. This is the thirteenth recorded boundary rejection.

## 2026-09-27 15:22 Asia/Shanghai — official Tool checkpoint (active)

- Durable unique traces: 231/1,550; native-valid: 231; fallbacks: 0; statuses: `{"success": 218, "success_with_tool_failure": 13}`; tool requests: 231; failed crop actions: 13; transient infrastructure statuses: 0. Run remains active.

## 2026-09-27 15:24 Asia/Shanghai — official Tool checkpoint (active)

- Durable unique traces: 239/1,550; native-valid: 239; fallbacks: 0; statuses: `{"success": 226, "success_with_tool_failure": 13}`; tool requests: 239; failed crop actions: 13; transient infrastructure statuses: 0. Run remains active.

## 2026-09-27 15:25 Asia/Shanghai — Tool boundary rejection detail

- qid 4495 requested `[150,153]` on a clip with 150 s official local duration. The crop guard rejected it; the final answer is native-valid with no fallback. This is the fourteenth recorded boundary rejection.

## 2026-09-27 15:26 Asia/Shanghai — Tool boundary rejection detail

- qid 10307 requested `[160,163]` on a clip with 150 s official local duration. The crop guard rejected it; final prediction is native-valid without fallback. This is the fifteenth recorded boundary rejection.

## 2026-09-27 15:27 Asia/Shanghai — Tool boundary rejection detail

- qid 561 requested `[150,165]` on a 150 s official local clip. The guard rejected the out-of-range start, and the final prediction remained native-valid without fallback. This is the sixteenth recorded boundary rejection.

## 2026-09-27 15:27 Asia/Shanghai — Tool boundary rejection detail

- qid 5672 requested `[185,200]` for a clip whose official local duration is 150 s. The crop guard rejected it; prediction remained native-valid with no fallback. This is the seventeenth recorded boundary rejection.

## 2026-09-27 15:28 Asia/Shanghai — Tool boundary rejection detail

- qid 3254 requested `[150,160]` on a 150 s official local clip. Guard rejected it; final answer is native-valid with no fallback. This is the eighteenth recorded boundary rejection.

## 2026-09-27 15:30 Asia/Shanghai — official Tool checkpoint (active)

- Durable unique traces: 263/1,550; native-valid: 263; fallbacks: 0; statuses: `{"success": 245, "success_with_tool_failure": 18}`; tool requests: 263; failed crop actions: 18; transient infrastructure statuses: 0. Run remains active.

## 2026-09-27 15:32 Asia/Shanghai — official Tool checkpoint (active)

- Durable unique traces: 270/1,550; native-valid: 270; fallbacks: 0; statuses: `{"success": 252, "success_with_tool_failure": 18}`; tool requests: 270; failed crop actions: 18; transient infrastructure statuses: 0. Run remains active.

## 2026-09-27 15:33 Asia/Shanghai — official Tool checkpoint (active)

- Durable unique traces: 278/1,550; native-valid: 278; fallbacks: 0; statuses: `{"success": 260, "success_with_tool_failure": 18}`; tool requests: 278; failed crop actions: 18; transient infrastructure statuses: 0. Run remains active.

## 2026-09-27 15:35 Asia/Shanghai — official Tool checkpoint (active)

- Durable unique traces: 285/1,550; native-valid: 285; fallbacks: 0; statuses: `{"success": 267, "success_with_tool_failure": 18}`; tool requests: 285; failed crop actions: 18; transient infrastructure statuses: 0. Run remains active.

## 2026-09-27 15:37 Asia/Shanghai — official Tool checkpoint (active)

- Durable unique traces: 293/1,550; native-valid: 293; fallbacks: 0; statuses: `{"success": 275, "success_with_tool_failure": 18}`; tool requests: 293; failed crop actions: 18; transient infrastructure statuses: 0. Run remains active.

## 2026-09-27 15:38 Asia/Shanghai — Tool boundary rejection detail

- qid 7567 requested `[660,810]`, matching the source-video offset in the clip filename and exceeding the 150 s official local duration. Guard rejected it; answer is native-valid with no fallback. This is the nineteenth recorded crop boundary rejection.

## 2026-09-27 15:39 Asia/Shanghai — Tool boundary rejection detail

- qid 5748 requested `[210,360]` on a clip with 150 s official local duration. Guard rejected the out-of-range start; prediction is native-valid with no fallback. This is the twentieth recorded crop boundary rejection.

## 2026-09-27 15:40 Asia/Shanghai — official Tool checkpoint (active)

- Durable unique traces: 305/1,550; native-valid: 305; fallbacks: 0; statuses: `{"success": 285, "success_with_tool_failure": 20}`; tool requests: 305; failed crop actions: 20; transient infrastructure statuses: 0. Run remains active.

## 2026-09-27 15:42 Asia/Shanghai — official Tool checkpoint (active)

- Durable unique traces: 313/1,550; native-valid: 313; fallbacks: 0; statuses: `{"success": 293, "success_with_tool_failure": 20}`; tool requests: 313; failed crop actions: 20; transient infrastructure statuses: 0. Run remains active.

## 2026-09-27 15:45 Asia/Shanghai — official Tool checkpoint (active)

- Durable unique traces: 323/1,550; native-valid: 323; fallbacks: 0; statuses: `{"success": 303, "success_with_tool_failure": 20}`; tool requests: 323; failed crop actions: 20; transient infrastructure statuses: 0. Run remains active.

## 2026-09-27 16:35 Asia/Shanghai — Tool run resumed after interruption

- The previous runner process is no longer active; durable artifacts contain 325 unique terminal qids with statuses `{"success": 304, "success_with_tool_failure": 21}`. Failed crop actions recorded so far: 21; all final predictions are native-valid and there are zero fallbacks.
- The same frozen command/output directory is being resumed. The service at port 18008 returned the expected LongVT-RFT revision with context 65,536; GPUs 6/7 UUIDs and availability were rechecked.

## 2026-09-27 16:36 Asia/Shanghai — resumed Tool checkpoint (active)

- Durable unique traces: 331/1,550; native-valid: 331; fallbacks: 0; statuses: `{"success": 310, "success_with_tool_failure": 21}`; failed crop actions: 21; transient infrastructure statuses: 0. Resumed runner remains active.

## 2026-09-27 16:37 Asia/Shanghai — Tool boundary rejection detail (resumed run)

- qid 5043 requested `[150,170]` on a clip whose official local duration is 150 s. The crop guard rejected it; the final answer is native-valid without fallback. This is the 22nd recorded boundary rejection.

## 2026-09-27 16:38 Asia/Shanghai — resumed Tool checkpoint (active)

- Durable unique traces: 341/1,550; native-valid: 341; fallbacks: 0; statuses: `{"success": 319, "success_with_tool_failure": 22}`; failed crop actions: 22; transient infrastructure statuses: 0. Runner remains active.

## 2026-09-27 16:40 Asia/Shanghai — Tool boundary rejection detail (resumed run)

- qid 7065 requested `[150,160]` on a 150 s official local clip. The guard rejected it; prediction remained native-valid without fallback. This is the 23rd recorded boundary rejection.

## 2026-09-27 16:41 Asia/Shanghai — Tool boundary rejection detail (resumed run)

- qid 7028 requested `[150,155]` on a 150 s official local clip. The guard rejected it; the final prediction is native-valid with no fallback. This is the 24th recorded boundary rejection.

## 2026-09-27 16:43 Asia/Shanghai — resumed Tool checkpoint (active)

- Durable unique traces: 362/1,550; native-valid: 362; fallbacks: 0; statuses: `{"success": 338, "success_with_tool_failure": 24}`; failed crop actions: 24; transient infrastructure statuses: 0. Runner remains active.

## 2026-09-27 16:45 Asia/Shanghai — Tool boundary rejection detail (resumed run)

- qid 10157 requested `[210,360]` on a clip with 150 s official local duration. Guard rejected it; final prediction is native-valid with no fallback. This is the 25th recorded boundary rejection.

## 2026-09-27 16:46 Asia/Shanghai — resumed Tool checkpoint (active)

- Durable unique traces: 376/1,550; native-valid: 376; fallbacks: 0; statuses: `{"success": 351, "success_with_tool_failure": 25}`; failed crop actions: 25; transient infrastructure statuses: 0. Runner remains active.

## 2026-09-27 16:48 Asia/Shanghai — resumed Tool checkpoint (active)

- Durable unique traces: 384/1,550; native-valid: 384; fallbacks: 0; statuses: `{"success": 359, "success_with_tool_failure": 25}`; failed crop actions: 25; transient infrastructure statuses: 0. Runner remains active.

## 2026-09-27 16:49 Asia/Shanghai — Tool boundary rejection detail (resumed run)

- qid 5224 requested `[150,160]` on a clip with 150 s official local duration. The guard rejected it; prediction was native-valid with no fallback. This is the 26th recorded boundary rejection.

## 2026-09-27 16:51 Asia/Shanghai — Tool boundary rejection detail (resumed run)

- qid 1487 requested `[150,170]` on a 150 s official local clip. The guard rejected it; the final prediction is native-valid without fallback. This is the 27th recorded boundary rejection.

## 2026-09-27 16:51 Asia/Shanghai — resumed Tool checkpoint (active)

- Durable unique traces: 396/1,550; native-valid: 396; fallbacks: 0; statuses: `{"success": 369, "success_with_tool_failure": 27}`; failed crop actions: 27; transient infrastructure statuses: 0. Runner remains active.

## 2026-09-27 16:53 Asia/Shanghai — resumed Tool checkpoint (active)

- Durable unique traces: 403/1,550; native-valid: 403; fallbacks: 0; statuses: `{"success": 376, "success_with_tool_failure": 27}`; failed crop actions: 27; transient infrastructure statuses: 0. Runner remains active.

## 2026-09-27 16:55 Asia/Shanghai — resumed Tool checkpoint (active)

- Durable unique traces: 409/1,550; native-valid: 409; fallbacks: 0; statuses: `{"success": 382, "success_with_tool_failure": 27}`; failed crop actions: 27; transient infrastructure statuses: 0. Runner remains active.

## 2026-09-27 16:57 Asia/Shanghai — resumed Tool checkpoint (active)

- Durable unique traces: 417/1,550; native-valid: 417; fallbacks: 0; statuses: `{"success": 390, "success_with_tool_failure": 27}`; failed crop actions: 27; transient infrastructure statuses: 0. Runner remains active.

## 2026-09-27 16:58 Asia/Shanghai — Tool boundary rejection detail (resumed run)

- qid 2513 requested `[150,153]` on a clip with 150 s official local duration. The guard rejected it; final prediction is native-valid with no fallback. This is the 28th recorded boundary rejection.

## 2026-09-27 17:00 Asia/Shanghai — resumed Tool checkpoint (active)

- Durable unique traces: 428/1,550; native-valid: 428; fallbacks: 0; statuses: `{"success": 400, "success_with_tool_failure": 28}`; failed crop actions: 28; transient infrastructure statuses: 0. Runner remains active.

## 2026-09-27 17:02 Asia/Shanghai — resumed Tool checkpoint (active)

- Durable unique traces: 436/1,550; native-valid: 436; fallbacks: 0; statuses: `{"success": 408, "success_with_tool_failure": 28}`; failed crop actions: 28; transient infrastructure statuses: 0. Runner remains active.

## 2026-09-27 17:03 Asia/Shanghai — Tool boundary rejection detail (resumed run)

- qid 825 requested `[210,360]` on a clip with 150 s official local duration. The guard rejected it; final prediction is native-valid with no fallback. This is the 29th recorded boundary rejection.

## 2026-09-27 17:04 Asia/Shanghai — Tool boundary rejection detail (resumed run)

- qid 10220 requested `[150,153]` on a 150 s official local clip. The guard rejected it; prediction is native-valid without fallback. This is the 30th recorded boundary rejection.

## 2026-09-27 17:04 Asia/Shanghai — resumed Tool checkpoint (active)

- Durable unique traces: 448/1,550; native-valid: 448; fallbacks: 0; statuses: `{"success": 418, "success_with_tool_failure": 30}`; failed crop actions: 30; transient infrastructure statuses: 0. Runner remains active.

## 2026-09-27 17:06 Asia/Shanghai — Tool boundary rejection detail (resumed run)

- qid 4266 requested `[150,170]` on a 150 s official local clip. Guard rejected it; final prediction is native-valid with no fallback. This is the 31st recorded boundary rejection.

## 2026-09-27 17:08 Asia/Shanghai — resumed Tool checkpoint (active)

- Durable unique traces: 465/1,550; native-valid: 465; fallbacks: 0; statuses: `{"success": 433, "success_with_tool_failure": 32}`; failed crop actions: 32; transient infrastructure statuses: 0. Runner remains active.

## 2026-09-27 17:09 Asia/Shanghai — Tool failure-ledger reconciliation

- Reconciled traces against prior progress entries and added the two missing per-qid details. qid 8730 requested `[150,160]` on a 150 s local clip; qid 3748 requested `[160,170]` on a 150 s local clip. Both were rejected by the crop guard and both final predictions are native-valid without fallback. Total boundary rejections observed so far: 32.

## 2026-09-27 17:11 Asia/Shanghai — resumed Tool checkpoint (active)

- Durable unique traces: 476/1,550; native-valid: 476; fallbacks: 0; statuses: `{"success": 444, "success_with_tool_failure": 32}`; failed crop actions: 32; transient infrastructure statuses: 0. Runner remains active.

## 2026-09-27 17:13 Asia/Shanghai — resumed Tool checkpoint (active)

- Durable unique traces: 485/1,550; native-valid: 485; fallbacks: 0; statuses: `{"success": 453, "success_with_tool_failure": 32}`; failed crop actions: 32; transient infrastructure statuses: 0. Runner remains active.

## 2026-09-27 17:15 Asia/Shanghai — resumed Tool checkpoint (active)

- Durable unique traces: 493/1,550; native-valid: 493; fallbacks: 0; statuses: `{"success": 461, "success_with_tool_failure": 32}`; failed crop actions: 32; transient infrastructure statuses: 0. Runner remains active.

## 2026-09-27 17:17 Asia/Shanghai — resumed Tool checkpoint (active)

- Durable unique traces: 501/1,550; native-valid: 501; fallbacks: 0; statuses: `{"success": 469, "success_with_tool_failure": 32}`; tool requests: 497; failed crop actions: 32; transient infrastructure statuses: 0. Runner remains active.

## 2026-09-27 17:19 Asia/Shanghai — resumed Tool checkpoint (active)

- Durable unique traces: 509/1,550; native-valid: 509; fallbacks: 0; statuses: `{"success": 477, "success_with_tool_failure": 32}`; tool requests: 505; failed crop actions: 32; transient infrastructure statuses: 0. Runner remains active.

## 2026-09-27 17:20 Asia/Shanghai — resumed Tool checkpoint (active)

- Durable unique traces: 517/1,550; native-valid: 517; fallbacks: 0; statuses: `{"success": 485, "success_with_tool_failure": 32}`; tool requests: 513; failed crop actions: 32; transient infrastructure statuses: 0. Runner remains active.

## 2026-09-27 17:22 Asia/Shanghai — resumed Tool checkpoint (active)

- Durable unique traces: 526/1,550; native-valid: 526; fallbacks: 0; statuses: `{"success": 494, "success_with_tool_failure": 32}`; failed crop actions: 32; transient infrastructure statuses: 0. Runner remains active.

## 2026-09-27 17:24 Asia/Shanghai — resumed Tool checkpoint (active)

- Durable unique traces: 533/1,550; native-valid: 533; fallbacks: 0; statuses: `{"success": 501, "success_with_tool_failure": 32}`; failed crop actions: 32; transient infrastructure statuses: 0. Runner remains active.

## 2026-09-27 17:26 Asia/Shanghai — resumed Tool checkpoint (active)

- Durable unique traces: 539/1,550; native-valid: 539; fallbacks: 0; statuses: `{"success": 507, "success_with_tool_failure": 32}`; failed crop actions: 32; transient infrastructure statuses: 0. Runner remains active.

## 2026-09-27 21:03 Asia/Shanghai — Tool resume checkpoint before detached relaunch

- The interactive runner is no longer active. Durable artifacts contain 539 unique qids; native-valid: 539; fallbacks: 0; statuses: `{"success": 507, "success_with_tool_failure": 32}`; failed crop actions: 32. No partially written qid was found.
- Service identity rechecked: LongVT-RFT revision `16ed9314e4ea600dc4682b172a614252f2f383ad`, context 65,536, port 18008; GPUs 6/7 are available for requests. A detached same-directory resume is being launched.

## 2026-09-27 21:04 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 543/1,550; native-valid: 543; fallbacks: 0; statuses: `{"success": 510, "success_with_tool_failure": 33}`; failed crop actions: 33. Detached runner PID 33749 is active; its stdout/stderr log is `results/full_v2b/longvt_tool/runner_detached.log`.

## 2026-09-27 21:05 Asia/Shanghai — Tool boundary rejection detail (detached run)

- qid 5275 requested `[290,300]` on a clip with 150 s official local duration. The crop guard rejected it; prediction remained native-valid without fallback. This is the 33rd recorded boundary rejection.

## 2026-09-27 21:06 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 549/1,550; native-valid: 549; fallbacks: 0; statuses: `{"success": 516, "success_with_tool_failure": 33}`; failed crop actions: 33. Detached runner PID 33749 remains active.

## 2026-09-27 21:07 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 553/1,550; native-valid: 553; fallbacks: 0; statuses: `{"success": 520, "success_with_tool_failure": 33}`; failed crop actions: 33. Detached runner PID 33749 remains active.

## 2026-09-27 21:08 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 557/1,550; native-valid: 557; fallbacks: 0; statuses: `{"success": 524, "success_with_tool_failure": 33}`; failed crop actions: 33. Detached runner PID 33749 remains active.

## 2026-09-27 21:10 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 564/1,550; native-valid: 564; fallbacks: 0. Detached runner PID 33749 remains active.
- qid 7548 requested `[150,170]` on a clip with 150 s local duration; guard rejected it. Prediction is native-valid, no fallback. This is the 34th recorded boundary rejection.

## 2026-09-27 21:10 Asia/Shanghai — detached Tool status snapshot

- Durable unique traces: 566/1,550; statuses: `{"success": 532, "success_with_tool_failure": 34}`; native-valid: 566; fallbacks: 0; failed crop actions: 34.

## 2026-09-27 21:11 Asia/Shanghai — detached Tool checkpoint and boundary detail

- Durable unique traces: 573/1,550; native-valid: 573; fallbacks: 0; status counts: `{'success': 538, 'success_with_tool_failure': 35}`. Runner PID 33749 remains active.
- qid 6383 requested `[270,290]` beyond the 150 s official local duration; guard rejected it and prediction remained valid. This is the 35th boundary rejection.

## 2026-09-27 21:13 Asia/Shanghai — detached Tool checkpoint and boundary detail

- Durable unique traces: 581/1,550; native-valid: 581; fallbacks: 0; statuses: `{"success": 544, "success_with_tool_failure": 37}`. Runner PID 33749 remains active.
- qid 3658 requested `[150,180]` beyond the 150 s official local duration; the guard rejected it and prediction remained valid. This is the 36th boundary rejection.

## 2026-09-27 21:14 Asia/Shanghai — detached Tool failure-ledger reconciliation

- Trace reconciliation found a new qid alongside the prior checkpoint: qid 8385 requested `[158,168]`, beyond the 150 s official local duration. The crop guard rejected it; prediction remained native-valid with no fallback. This is the 37th boundary rejection.
- Current durable trace count: 582/1,550; status counts: `{"success": 545, "success_with_tool_failure": 37}`. Runner PID 33749 remains active.

## 2026-09-27 21:15 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 588/1,550; native-valid: 588; fallbacks: 0; statuses: `{"success": 551, "success_with_tool_failure": 37}`; failed crop actions: 37. PID 33749 remains active.

## 2026-09-27 21:17 Asia/Shanghai — detached Tool checkpoint and boundary detail

- Durable unique traces: 595/1,550; native-valid: 595; fallbacks: 0; statuses: `{"success": 557, "success_with_tool_failure": 38}`. Runner PID 33749 remains active.
- qid 2600 requested `[210,230]` on a 150 s official local clip; crop guard rejected the out-of-range start, and the prediction remained native-valid. This is the 38th boundary rejection.

## 2026-09-27 21:18 Asia/Shanghai — detached Tool checkpoint and boundary detail

- Durable unique traces: 602/1,550; native-valid: 602; fallbacks: 0; statuses: `{"success": 563, "success_with_tool_failure": 39}`. Runner PID 33749 remains active.
- qid 4572 requested `[350,360]` on a 150 s official local clip; guard rejected it and prediction remained valid. This is the 39th boundary rejection.

## 2026-09-27 21:20 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 610/1,550; native-valid: 610; fallbacks: 0; statuses: `{"success": 571, "success_with_tool_failure": 39}`; failed crop actions: 39. Runner PID 33749 remains active.

## 2026-09-27 21:21 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 616/1,550; native-valid: 616; fallbacks: 0; statuses: `{"success": 577, "success_with_tool_failure": 39}`; failed crop actions: 39. Runner PID 33749 remains active.

## 2026-09-27 21:23 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 623/1,550; native-valid: 623; fallbacks: 0; statuses: `{"success": 584, "success_with_tool_failure": 39}`; failed crop actions: 39. Runner PID 33749 remains active.

## 2026-09-27 21:25 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 631/1,550; native-valid: 631; fallbacks: 0; statuses: `{"success": 592, "success_with_tool_failure": 39}`; failed crop actions: 39. Runner PID 33749 remains active.

## 2026-09-27 21:26 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 637/1,550; native-valid: 637; fallbacks: 0; statuses: `{"success": 598, "success_with_tool_failure": 39}`; failed crop actions: 39. Runner PID 33749 remains active.

## 2026-09-27 21:28 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 642/1,550; native-valid: 642; fallbacks: 0; statuses: `{"success": 603, "success_with_tool_failure": 39}`; failed crop actions: 39. Runner PID 33749 remains active.

## 2026-09-27 21:29 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 648/1,550; native-valid: 648; fallbacks: 0; statuses: `{"success": 609, "success_with_tool_failure": 39}`; failed crop actions: 39. Runner PID 33749 remains active.

## 2026-09-27 21:31 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 653/1,550; native-valid: 653; fallbacks: 0; statuses: `{"success": 614, "success_with_tool_failure": 39}`; failed crop actions: 39. Runner PID 33749 remains active.

## 2026-09-27 21:33 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 660/1,550; native-valid: 660; fallbacks: 0; statuses: `{"success": 620, "success_with_tool_failure": 40}`; failed crop actions: 40. Runner PID 33749 remains active.

## 2026-09-27 21:34 Asia/Shanghai — failure-ledger reconciliation and detached checkpoint

- qid 5537 requested `[150,160]` on a 150 s official local clip; the crop guard rejected it. Prediction remained native-valid without fallback.
- qid 9884 requested `[220,250]` on a 150 s official local clip; the crop guard rejected it. Prediction remained native-valid without fallback.
- Cumulative failed crop actions: 41; durable unique traces: 665/1,550; statuses: `{"success": 624, "success_with_tool_failure": 41}`. PID 33749 remains active.

## 2026-09-27 21:34 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 667/1,550; native-valid: 667; fallbacks: 0; statuses: `{"success": 626, "success_with_tool_failure": 41}`; failed crop actions: 41. Failure detail is reconciled against all current tool traces. PID 33749 remains active.

## 2026-09-27 21:36 Asia/Shanghai — detached Tool checkpoint and boundary detail

- Durable unique traces: 676/1,550; native-valid: 676; fallbacks: 0; statuses: `{"success": 634, "success_with_tool_failure": 42}`. PID 33749 remains active.
- qid 10217 requested `[150,153]` beyond the official 150 s local duration; the crop guard rejected it and the prediction remained valid. This is the 42nd boundary rejection.

## 2026-09-27 21:38 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 685/1,550; native-valid: 685; fallbacks: 0; statuses: `{"success": 643, "success_with_tool_failure": 42}`; failed crop actions: 42. PID 33749 remains active.

## 2026-09-27 21:40 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 692/1,550; native-valid: 692; fallbacks: 0; statuses: `{"success": 650, "success_with_tool_failure": 42}`; failed crop actions: 42. PID 33749 remains active.

## 2026-09-27 21:42 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 700/1,550; native-valid: 700; fallbacks: 0; statuses: `{"success": 658, "success_with_tool_failure": 42}`; failed crop actions: 42. PID 33749 remains active.

## 2026-09-27 21:44 Asia/Shanghai — detached Tool checkpoint and boundary detail

- Durable unique traces: 711/1,550; native-valid: 711; fallbacks: 0; statuses: `{"success": 668, "success_with_tool_failure": 43}`. PID 33749 remains active.
- qid 9574 requested `[290,305]` beyond the 150 s official local duration; guard rejected it and prediction remained valid. This is the 43rd boundary rejection.

## 2026-09-27 21:46 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 718/1,550; native-valid: 718; fallbacks: 0; statuses: `{"success": 675, "success_with_tool_failure": 43}`; failed crop actions: 43. PID 33749 remains active.

## 2026-09-27 21:49 Asia/Shanghai — detached Tool checkpoint and boundary detail

- Durable unique traces: 729/1,550; native-valid: 729; fallbacks: 0; statuses: `{"success": 684, "success_with_tool_failure": 45}`; failed crop actions: 45. Runner PID 33749 remains active.
- Newly reconciled qid 5758 requested `[150,180]` and qid 1678 requested `[150,170]` beyond their official 150 s local clip durations; both crop calls were rejected by the duration guard, while predictions remained native-valid and no fallback was applied. These are boundary failures, not infrastructure failures.

## 2026-09-27 21:51 Asia/Shanghai — detached Tool checkpoint and boundary detail

- Durable unique traces: 736/1,550; native-valid: 736; fallbacks: 0; statuses: `{"success": 690, "success_with_tool_failure": 46}`; failed crop actions: 46. Runner PID 33749 remains active.
- qid 8340 requested `[150,160]` beyond its official 150 s local clip duration; the crop guard rejected it, while the prediction remained native-valid without fallback.

## 2026-09-27 21:52 Asia/Shanghai — detached Tool checkpoint and boundary detail

- Durable unique traces: 742/1,550; native-valid: 742; fallbacks: 0; statuses: `{"success": 695, "success_with_tool_failure": 47}`; failed crop actions: 47. Runner PID 33749 remains active.
- qid 8152 requested `[210,360]` beyond its official 150 s local clip duration; the crop guard rejected it, while the prediction remained native-valid without fallback.

## 2026-09-27 21:54 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 748/1,550; native-valid: 748; fallbacks: 0; statuses: `{"success": 701, "success_with_tool_failure": 47}`; failed crop actions: 47. Runner PID 33749 remains active (33749       50:27 Ssl).

## 2026-09-27 21:55 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 754/1,550; native-valid: 754; fallbacks: 0; statuses: `{"success": 707, "success_with_tool_failure": 47}`; failed crop actions: 47. Runner PID 33749: 33749       51:43 Ssl.

## 2026-09-27 21:56 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 760/1,550; native-valid: 760; fallbacks: 0; statuses: `{"success": 713, "success_with_tool_failure": 47}`; failed crop actions: 47. Runner PID 33749: 33749       53:01 Ssl.

## 2026-09-27 21:57 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 766/1,550; native-valid: 766; fallbacks: 0; statuses: `{"success": 719, "success_with_tool_failure": 47}`; failed crop actions: 47. Runner PID 33749: 33749       54:17 Ssl.

## 2026-09-27 21:59 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 771/1,550; native-valid: 771; fallbacks: 0; statuses: `{"success": 724, "success_with_tool_failure": 47}`; failed crop actions: 47. Runner PID 33749: 33749       55:35 Ssl.

## 2026-09-27 22:00 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 777/1,550; native-valid: 777; fallbacks: 0; statuses: `{"success": 730, "success_with_tool_failure": 47}`; failed crop actions: 47. Runner PID 33749: 33749       56:50 Ssl.

## 2026-09-27 22:01 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 781/1,550; native-valid: 781; fallbacks: 0; statuses: `{"success": 732, "success_with_tool_failure": 49}`; failed crop actions: 49. Runner PID 33749: 33749       58:04 Ssl.
- Newly reconciled qid 1894: crop `{'video_path': '/path/to/storage/longvt-repro/benchmark_data/QVHighlights/videos_full/videos/bowYAwxuG-c_60.0_210.0.mp4', 'start_time': 160.0, 'end_time': 170.0}` rejected with `ValueError('crop_start_outside_official_local_duration')`; prediction native-valid, no fallback.
- Newly reconciled qid 7826: crop `{'video_path': '/path/to/storage/longvt-repro/benchmark_data/QVHighlights/videos_full/videos/bowYAwxuG-c_210.0_360.0.mp4', 'start_time': 210.0, 'end_time': 360.0}` rejected with `ValueError('crop_start_outside_official_local_duration')`; prediction native-valid, no fallback.

## 2026-09-27 22:03 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 787/1,550; native-valid: 787; fallbacks: 0; statuses: `{"success": 738, "success_with_tool_failure": 49}`; failed crop actions: 49. Runner PID 33749: 33749       59:24 Ssl.

## 2026-09-27 22:04 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 793/1,550; native-valid: 793; fallbacks: 0; statuses: `{"success": 743, "success_with_tool_failure": 50}`; failed crop actions: 50. Runner PID 33749: 33749    01:00:38 Ssl.
- Newly reconciled qid 4163: crop `{'video_path': '/path/to/storage/longvt-repro/benchmark_data/QVHighlights/videos_full/videos/WC9KLy8Py-k_210.0_360.0.mp4', 'start_time': 210.0, 'end_time': 360.0}` rejected with `ValueError('crop_start_outside_official_local_duration')`; prediction native-valid, no fallback.

## 2026-09-27 22:05 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 798/1,550; native-valid: 798; fallbacks: 0; statuses: `{"success": 748, "success_with_tool_failure": 50}`; failed crop actions: 50. Runner PID 33749: 33749    01:01:55 Ssl.

## 2026-09-27 22:06 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 804/1,550; native-valid: 804; fallbacks: 0; statuses: `{"success": 752, "success_with_tool_failure": 52}`; failed crop actions: 52. Runner PID 33749: 33749    01:03:09 Rsl.
- Newly reconciled qid 8220: crop `{'video_path': '/path/to/storage/longvt-repro/benchmark_data/QVHighlights/videos_full/videos/av5DZl8MvcQ_210.0_360.0.mp4', 'start_time': 290.0, 'end_time': 300.0}` rejected with `ValueError('crop_start_outside_official_local_duration')`; prediction native-valid, no fallback.
- Newly reconciled qid 5765: crop `{'video_path': '/path/to/storage/longvt-repro/benchmark_data/QVHighlights/videos_full/videos/qHSeudOaXwk_210.0_360.0.mp4', 'start_time': 150.0, 'end_time': 160.0}` rejected with `ValueError('crop_start_outside_official_local_duration')`; prediction native-valid, no fallback.

## 2026-09-27 22:08 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 808/1,550; native-valid: 808; fallbacks: 0; statuses: `{"success": 756, "success_with_tool_failure": 52}`; failed crop actions: 52. Runner PID 33749: 33749    01:04:23 Ssl.

## 2026-09-27 22:09 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 813/1,550; native-valid: 813; fallbacks: 0; statuses: `{"success": 761, "success_with_tool_failure": 52}`; failed crop actions: 52. Runner PID 33749: 33749    01:05:38 Ssl.

## 2026-09-27 22:10 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 819/1,550; native-valid: 819; fallbacks: 0; statuses: `{"success": 766, "success_with_tool_failure": 53}`; failed crop actions: 53. Runner PID 33749: 33749    01:07:04 Ssl.
- Newly reconciled qid 5864: crop `{'video_path': '/path/to/storage/longvt-repro/benchmark_data/QVHighlights/videos_full/videos/FlIZP8D0_gQ_210.0_360.0.mp4', 'start_time': 150.0, 'end_time': 360.0}` rejected with `ValueError('crop_start_outside_official_local_duration')`; prediction native-valid, no fallback.

## 2026-09-27 22:11 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 825/1,550; native-valid: 825; fallbacks: 0; statuses: `{"success": 770, "success_with_tool_failure": 55}`; failed crop actions: 55. Runner PID 33749: 33749    01:08:19 Ssl.
- Newly reconciled qid 9985: crop `{'video_path': '/path/to/storage/longvt-repro/benchmark_data/QVHighlights/videos_full/videos/4G9S-eoLgX4_360.0_510.0.mp4', 'start_time': 360.0, 'end_time': 390.0}` rejected with `ValueError('crop_start_outside_official_local_duration')`; prediction native-valid, no fallback.
- Newly reconciled qid 1285: crop `{'video_path': '/path/to/storage/longvt-repro/benchmark_data/QVHighlights/videos_full/videos/S1DZb9SkrK8_60.0_210.0.mp4', 'start_time': 150.0, 'end_time': 155.0}` rejected with `ValueError('crop_start_outside_official_local_duration')`; prediction native-valid, no fallback.

## 2026-09-27 22:13 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 830/1,550; native-valid: 830; fallbacks: 0; statuses: `{"success": 775, "success_with_tool_failure": 55}`; failed crop actions: 55. Runner PID 33749: 33749    01:09:33 Rsl.

## 2026-09-27 22:14 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 835/1,550; native-valid: 835; fallbacks: 0; statuses: `{"success": 780, "success_with_tool_failure": 55}`; failed crop actions: 55. Runner PID 33749: 33749    01:10:51 Ssl.

## 2026-09-27 22:15 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 840/1,550; native-valid: 840; fallbacks: 0; statuses: `{"success": 784, "success_with_tool_failure": 56}`; failed crop actions: 56. Runner PID 33749: 33749    01:12:05 Ssl.
- Newly reconciled qid 10063: crop `{'video_path': '/path/to/storage/longvt-repro/benchmark_data/QVHighlights/videos_full/videos/c-fzjF9ubUc_210.0_360.0.mp4', 'start_time': 150.0, 'end_time': 160.0}` rejected with `ValueError('crop_start_outside_official_local_duration')`; prediction native-valid, no fallback.

## 2026-09-27 22:16 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 846/1,550; native-valid: 846; fallbacks: 0; statuses: `{"success": 790, "success_with_tool_failure": 56}`; failed crop actions: 56. Runner PID 33749: 33749    01:13:18 Rsl.

## 2026-09-27 22:18 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 851/1,550; native-valid: 851; fallbacks: 0; statuses: `{"success": 795, "success_with_tool_failure": 56}`; failed crop actions: 56. Runner PID 33749: 33749    01:14:31 Ssl.

## 2026-09-27 22:19 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 855/1,550; native-valid: 855; fallbacks: 0; statuses: `{"success": 798, "success_with_tool_failure": 57}`; failed crop actions: 57. Runner PID 33749: 33749    01:15:45 Ssl.
- Newly reconciled qid 3249: crop `{'video_path': '/path/to/storage/longvt-repro/benchmark_data/QVHighlights/videos_full/videos/raTrFwJMoJo_210.0_360.0.mp4', 'start_time': 160.0, 'end_time': 170.0}` rejected with `ValueError('crop_start_outside_official_local_duration')`; prediction native-valid, no fallback.

## 2026-09-27 22:20 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 859/1,550; native-valid: 859; fallbacks: 0; statuses: `{"success": 802, "success_with_tool_failure": 57}`; failed crop actions: 57. Runner PID 33749: 33749    01:17:00 Ssl.

## 2026-09-27 22:21 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 863/1,550; native-valid: 863; fallbacks: 0; statuses: `{"success": 806, "success_with_tool_failure": 57}`; failed crop actions: 57. Runner PID 33749: 33749    01:18:15 Ssl.

## 2026-09-27 22:23 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 868/1,550; native-valid: 868; fallbacks: 0; statuses: `{"success": 811, "success_with_tool_failure": 57}`; failed crop actions: 57. Runner PID 33749: 33749    01:19:28 Ssl.

## 2026-09-27 22:24 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 873/1,550; native-valid: 873; fallbacks: 0; statuses: `{"success": 815, "success_with_tool_failure": 58}`; failed crop actions: 58. Runner PID 33749: 33749    01:20:43 Ssl.
- Newly reconciled qid 7569: crop `{'video_path': '/path/to/storage/longvt-repro/benchmark_data/QVHighlights/videos_full/videos/2I9-kvemtSU_660.0_810.0.mp4', 'start_time': 710.0, 'end_time': 730.0}` rejected with `ValueError('crop_start_outside_official_local_duration')`; prediction native-valid, no fallback.

## 2026-09-27 22:25 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 879/1,550; native-valid: 879; fallbacks: 0; statuses: `{"success": 821, "success_with_tool_failure": 58}`; failed crop actions: 58. Runner PID 33749: 33749    01:21:57 Ssl.

## 2026-09-27 22:26 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 884/1,550; native-valid: 884; fallbacks: 0; statuses: `{"success": 825, "success_with_tool_failure": 59}`; failed crop actions: 59. Runner PID 33749: 33749    01:23:10 Ssl.
- Newly reconciled qid 6658: crop `{'video_path': '/path/to/storage/longvt-repro/benchmark_data/QVHighlights/videos_full/videos/aXgE_cVxJi0_0.0_179.0.mp4', 'start_time': 10.8, 'end_time': 64.4}` rejected with `ValueError('tool_video_path_mismatch')`; prediction native-valid, no fallback.

## 2026-09-27 22:28 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 889/1,550; native-valid: 889; fallbacks: 0; statuses: `{"success": 830, "success_with_tool_failure": 59}`; failed crop actions: 59. Runner PID 33749: 33749    01:24:29 Ssl.

## 2026-09-27 22:29 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 895/1,550; native-valid: 895; fallbacks: 0; statuses: `{"success": 835, "success_with_tool_failure": 60}`; failed crop actions: 60. Runner PID 33749: 33749    01:25:48 Ssl.
- Newly reconciled qid 4909: crop `{'video_path': '/path/to/storage/longvt-repro/benchmark_data/QVHighlights/videos_full/videos/EUTccoeKVZg_210.0_360.0.mp4', 'start_time': 210.0, 'end_time': 260.0}` rejected with `ValueError('crop_start_outside_official_local_duration')`; prediction native-valid, no fallback.

## 2026-09-27 22:30 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 901/1,550; native-valid: 901; fallbacks: 0; statuses: `{"success": 840, "success_with_tool_failure": 61}`; failed crop actions: 61. Runner PID 33749: 33749    01:27:07 Ssl.
- Newly reconciled qid 4637: crop `{'video_path': '/path/to/storage/longvt-repro/benchmark_data/QVHighlights/videos_full/videos/dsmZngVgytA_210.0_360.0.mp4', 'start_time': 150.0, 'end_time': 155.0}` rejected with `ValueError('crop_start_outside_official_local_duration')`; prediction native-valid, no fallback.

## 2026-09-27 22:32 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 907/1,550; native-valid: 907; fallbacks: 0; statuses: `{"success": 845, "success_with_tool_failure": 62}`; failed crop actions: 62. Runner PID 33749: 33749    01:28:23 Ssl.
- Newly reconciled qid 5070: crop `{'video_path': '/path/to/storage/longvt-repro/benchmark_data/QVHighlights/videos_full/videos/m0cZFcXa5sc_210.0_360.0.mp4', 'start_time': 150.0, 'end_time': 210.0}` rejected with `ValueError('crop_start_outside_official_local_duration')`; prediction native-valid, no fallback.

## 2026-09-27 22:33 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 913/1,550; native-valid: 913; fallbacks: 0; statuses: `{"success": 850, "success_with_tool_failure": 63}`; failed crop actions: 63. Runner PID 33749: 33749    01:29:45 Ssl.
- Newly reconciled qid 6192: crop `{'video_path': '/path/to/storage/longvt-repro/benchmark_data/QVHighlights/videos_full/videos/m0cZFcXa5sc_60.0_210.0.mp4', 'start_time': 150.0, 'end_time': 170.0}` rejected with `ValueError('crop_start_outside_official_local_duration')`; prediction native-valid, no fallback.

## 2026-09-27 22:34 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 919/1,550; native-valid: 919; fallbacks: 0; statuses: `{"success": 856, "success_with_tool_failure": 63}`; failed crop actions: 63. Runner PID 33749: 33749    01:31:01 Ssl.

## 2026-09-27 22:36 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 925/1,550; native-valid: 925; fallbacks: 0; statuses: `{"success": 861, "success_with_tool_failure": 64}`; failed crop actions: 64. Runner PID 33749: 33749    01:32:40 Ssl.
- Newly reconciled qid 8143: crop `{'video_path': '/path/to/storage/longvt-repro/benchmark_data/QVHighlights/videos_full/videos/czN-9IsQXoU_210.0_360.0.mp4', 'start_time': 150.0, 'end_time': 158.0}` rejected with `ValueError('crop_start_outside_official_local_duration')`; prediction native-valid, no fallback.

## 2026-09-27 22:37 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 931/1,550; native-valid: 931; fallbacks: 0; statuses: `{"success": 867, "success_with_tool_failure": 64}`; failed crop actions: 64. Runner PID 33749: 33749    01:34:20 Ssl.

## 2026-09-27 22:39 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 937/1,550; native-valid: 937; fallbacks: 0; statuses: `{"success": 872, "success_with_tool_failure": 65}`; failed crop actions: 65. Runner PID 33749: 33749    01:35:34 Ssl.
- Newly reconciled qid 8182: crop `{'video_path': '/path/to/storage/longvt-repro/benchmark_data/QVHighlights/videos_full/videos/xVJMPmmW7rE_210.0_360.0.mp4', 'start_time': 150.0, 'end_time': 160.0}` rejected with `ValueError('crop_start_outside_official_local_duration')`; prediction native-valid, no fallback.

## 2026-09-27 22:40 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 942/1,550; native-valid: 942; fallbacks: 0; statuses: `{"success": 877, "success_with_tool_failure": 65}`; failed crop actions: 65. Runner PID 33749: 33749    01:36:52 Rsl.

## 2026-09-27 22:42 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 951/1,550; native-valid: 951; fallbacks: 0; statuses: `{"success": 886, "success_with_tool_failure": 65}`; failed crop actions: 65. Runner PID 33749: 33749    01:38:53 Ssl.

## 2026-09-27 22:44 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 956/1,550; native-valid: 956; fallbacks: 0; statuses: `{"success": 890, "success_with_tool_failure": 66}`; failed crop actions: 66. Runner PID 33749: 33749    01:40:27 Rsl.
- Newly reconciled qid 2418: crop `{'video_path': '/path/to/storage/longvt-repro/benchmark_data/QVHighlights/videos_full/videos/kPXnelu5wZk_360.0_510.0.mp4', 'start_time': 150.0, 'end_time': 157.0}` rejected with `ValueError('crop_start_outside_official_local_duration')`; prediction native-valid, no fallback.

## 2026-09-27 22:45 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 962/1,550; native-valid: 962; fallbacks: 0; statuses: `{"success": 896, "success_with_tool_failure": 66}`; failed crop actions: 66. Runner PID 33749: 33749    01:41:59 Ssl.

## 2026-09-27 22:47 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 969/1,550; native-valid: 969; fallbacks: 0; statuses: `{"success": 902, "success_with_tool_failure": 67}`; failed crop actions: 67. Runner PID 33749: 33749    01:43:43 Ssl.
- Newly reconciled qid 4974: crop `{'video_path': '/path/to/storage/longvt-repro/benchmark_data/QVHighlights/videos_full/videos/xjV76k420Rc_210.0_360.0.mp4', 'start_time': 210.0, 'end_time': 360.0}` rejected with `ValueError('crop_start_outside_official_local_duration')`; prediction native-valid, no fallback.

## 2026-09-27 22:48 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 976/1,550; native-valid: 976; fallbacks: 0; statuses: `{"success": 909, "success_with_tool_failure": 67}`; failed crop actions: 67. Runner PID 33749: 33749    01:45:17 Ssl.

## 2026-09-27 22:50 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 982/1,550; native-valid: 982; fallbacks: 0; statuses: `{"success": 914, "success_with_tool_failure": 68}`; failed crop actions: 68. Runner PID 33749: 33749    01:46:34 Ssl.
- Newly reconciled qid 6980: crop `{'video_path': '/path/to/storage/longvt-repro/benchmark_data/QVHighlights/videos_full/videos/oQeSC0A1wuc_510.0_660.0.mp4', 'start_time': 150.0, 'end_time': 160.0}` rejected with `ValueError('crop_start_outside_official_local_duration')`; prediction native-valid, no fallback.

## 2026-09-27 22:51 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 987/1,550; native-valid: 987; fallbacks: 0; statuses: `{"success": 918, "success_with_tool_failure": 69}`; failed crop actions: 69. Runner PID 33749: 33749    01:47:50 Rsl.
- Newly reconciled qid 2611: crop `{'video_path': '/path/to/storage/longvt-repro/benchmark_data/QVHighlights/videos_full/videos/Y2GVVC_cD_E_60.0_210.0.mp4', 'start_time': 150.0, 'end_time': 156.0}` rejected with `ValueError('crop_start_outside_official_local_duration')`; prediction native-valid, no fallback.

## 2026-09-27 22:52 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 992/1,550; native-valid: 992; fallbacks: 0; statuses: `{"success": 923, "success_with_tool_failure": 69}`; failed crop actions: 69. Runner PID 33749: 33749    01:49:05 Ssl.

## 2026-09-27 22:54 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 998/1,550; native-valid: 998; fallbacks: 0; statuses: `{"success": 928, "success_with_tool_failure": 70}`; failed crop actions: 70. Runner PID 33749: 33749    01:50:23 Ssl.
- Newly reconciled qid 2374: crop `{'video_path': '/path/to/storage/longvt-repro/benchmark_data/QVHighlights/videos_full/videos/HZBdPoVFViw_210.0_360.0.mp4', 'start_time': 210.0, 'end_time': 260.0}` rejected with `ValueError('crop_start_outside_official_local_duration')`; prediction native-valid, no fallback.

## 2026-09-27 22:55 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1003/1,550; native-valid: 1003; fallbacks: 0; statuses: `{"success": 933, "success_with_tool_failure": 70}`; failed crop actions: 70. Runner PID 33749: 33749    01:51:39 Ssl.

## 2026-09-27 22:56 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1009/1,550; native-valid: 1009; fallbacks: 0; statuses: `{"success": 939, "success_with_tool_failure": 70}`; failed crop actions: 70. Runner PID 33749: 33749    01:52:59 Rsl.

## 2026-09-27 22:57 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1015/1,550; native-valid: 1015; fallbacks: 0; statuses: `{"success": 945, "success_with_tool_failure": 70}`; failed crop actions: 70. Runner PID 33749: 33749    01:54:18 Ssl.

## 2026-09-27 22:59 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1021/1,550; native-valid: 1021; fallbacks: 0; statuses: `{"success": 951, "success_with_tool_failure": 70}`; failed crop actions: 70. Runner PID 33749: 33749    01:55:41 Ssl.

## 2026-09-27 23:00 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1028/1,550; native-valid: 1028; fallbacks: 0; statuses: `{"success": 958, "success_with_tool_failure": 70}`; failed crop actions: 70. Runner PID 33749: 33749    01:57:07 Ssl.

## 2026-09-27 23:02 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1033/1,550; native-valid: 1033; fallbacks: 0; statuses: `{"success": 963, "success_with_tool_failure": 70}`; failed crop actions: 70. Runner PID 33749: 33749    01:58:26 Ssl.

## 2026-09-27 23:03 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1040/1,550; native-valid: 1040; fallbacks: 0; statuses: `{"success": 970, "success_with_tool_failure": 70}`; failed crop actions: 70. Runner PID 33749: 33749    01:59:55 Ssl.

## 2026-09-27 23:04 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1045/1,550; native-valid: 1045; fallbacks: 0; statuses: `{"success": 975, "success_with_tool_failure": 70}`; failed crop actions: 70. Runner PID 33749: 33749    02:01:09 Ssl.

## 2026-09-27 23:06 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1050/1,550; native-valid: 1050; fallbacks: 0; statuses: `{"success": 980, "success_with_tool_failure": 70}`; failed crop actions: 70. Runner PID 33749: 33749    02:02:24 Ssl.

## 2026-09-27 23:07 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1056/1,550; native-valid: 1056; fallbacks: 0; statuses: `{"success": 985, "success_with_tool_failure": 71}`; failed crop actions: 71. Runner PID 33749: 33749    02:03:39 Rsl.
- Newly reconciled qid 1304: crop `{'video_path': '/path/to/storage/longvt-repro/benchmark_data/QVHighlights/videos_full/videos/h_XoA5-_8W0_210.0_360.0.mp4', 'start_time': 180.0, 'end_time': 186.0}` rejected with `ValueError('crop_start_outside_official_local_duration')`; prediction native-valid, no fallback.

## 2026-09-27 23:08 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1062/1,550; native-valid: 1062; fallbacks: 0; statuses: `{"success": 991, "success_with_tool_failure": 71}`; failed crop actions: 71. Runner PID 33749: 33749    02:04:58 Ssl.

## 2026-09-27 23:09 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1067/1,550; native-valid: 1067; fallbacks: 0; statuses: `{"success": 996, "success_with_tool_failure": 71}`; failed crop actions: 71. Runner PID 33749: 33749    02:06:17 Ssl.

## 2026-09-27 23:11 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1073/1,550; native-valid: 1073; fallbacks: 0; statuses: `{"success": 1001, "success_with_tool_failure": 72}`; failed crop actions: 72. Runner PID 33749: 33749    02:07:42 Rsl.
- Newly reconciled qid 1906: crop `{'video_path': '/path/to/storage/longvt-repro/benchmark_data/QVHighlights/videos_full/videos/YBhpID-rR6o_210.0_360.0.mp4', 'start_time': 210.0, 'end_time': 360.0}` rejected with `ValueError('crop_start_outside_official_local_duration')`; prediction native-valid, no fallback.

## 2026-09-27 23:12 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1079/1,550; native-valid: 1079; fallbacks: 0; statuses: `{"success": 1005, "success_with_tool_failure": 74}`; failed crop actions: 74. Runner PID 33749: 33749    02:09:14 Ssl.
- Newly reconciled qid 8859: crop `{'video_path': '/path/to/storage/longvt-repro/benchmark_data/QVHighlights/videos_full/videos/-4Mlqc7PbZY_510.0_660.0.mp4', 'start_time': 150.0, 'end_time': 170.0}` rejected with `ValueError('crop_start_outside_official_local_duration')`; prediction native-valid, no fallback.
- Newly reconciled qid 3629: crop `{'video_path': '/path/to/storage/longvt-repro/benchmark_data/QVHighlights/videos_full/videos/CltHKz3Zfxc_210.0_360.0.mp4', 'start_time': 190.0, 'end_time': 210.0}` rejected with `ValueError('crop_start_outside_official_local_duration')`; prediction native-valid, no fallback.

## 2026-09-27 23:14 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1085/1,550; native-valid: 1085; fallbacks: 0; statuses: `{"success": 1010, "success_with_tool_failure": 75}`; failed crop actions: 75. Runner PID 33749: 33749    02:10:35 Ssl.
- Newly reconciled qid 1842: crop `{'video_path': '/path/to/storage/longvt-repro/benchmark_data/QVHighlights/videos_full/videos/DDiq5C34KL4_210.0_360.0.mp4', 'start_time': 150.0, 'end_time': 170.0}` rejected with `ValueError('crop_start_outside_official_local_duration')`; prediction native-valid, no fallback.

## 2026-09-27 23:17 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1098/1,550; native-valid: 1098; fallbacks: 0; statuses: `{"success": 1022, "success_with_tool_failure": 76}`; failed crop actions: 76. Runner PID 33749: 33749    02:13:46 Rsl.
- Newly reconciled qid 8824: crop `{'video_path': '/path/to/storage/longvt-repro/benchmark_data/QVHighlights/videos_full/videos/eeeYIo6nxcA_510.0_660.0.mp4', 'start_time': 150.0, 'end_time': 160.0}` rejected with `ValueError('crop_start_outside_official_local_duration')`; prediction native-valid, no fallback.

## 2026-09-27 23:18 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1104/1,550; native-valid: 1104; fallbacks: 0; statuses: `{"success": 1028, "success_with_tool_failure": 76}`; failed crop actions: 76. Runner PID 33749: 33749    02:15:03 Rsl.

## 2026-09-27 23:19 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1108/1,550; native-valid: 1108; fallbacks: 0; statuses: `{"success": 1032, "success_with_tool_failure": 76}`; failed crop actions: 76. Runner PID 33749: 33749    02:16:17 Ssl.

## 2026-09-27 23:21 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1114/1,550; native-valid: 1114; fallbacks: 0; statuses: `{"success": 1038, "success_with_tool_failure": 76}`; failed crop actions: 76. Runner PID 33749: 33749    02:17:32 Ssl.

## 2026-09-27 23:22 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1119/1,550; native-valid: 1119; fallbacks: 0; statuses: `{"success": 1043, "success_with_tool_failure": 76}`; failed crop actions: 76. Runner PID 33749: 33749    02:18:45 Ssl.

## 2026-09-27 23:23 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1124/1,550; native-valid: 1124; fallbacks: 0; statuses: `{"success": 1047, "success_with_tool_failure": 77}`; failed crop actions: 77. Runner PID 33749: 33749    02:19:59 Ssl.
- Newly reconciled qid 7398: crop `{'video_path': '/path/to/storage/longvt-repro/benchmark_data/QVHighlights/videos_full/videos/A5PIgmXGIdI_510.0_660.0.mp4', 'start_time': 150.0, 'end_time': 150.0}` rejected with `ValueError('crop_start_outside_official_local_duration')`; prediction native-valid, no fallback.

## 2026-09-27 23:24 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1130/1,550; native-valid: 1130; fallbacks: 0; statuses: `{"success": 1053, "success_with_tool_failure": 77}`; failed crop actions: 77. Runner PID 33749: 33749    02:21:12 Ssl.

## 2026-09-27 23:26 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1135/1,550; native-valid: 1135; fallbacks: 0; statuses: `{"success": 1058, "success_with_tool_failure": 77}`; failed crop actions: 77. Runner PID 33749: 33749    02:22:25 Ssl.

## 2026-09-27 23:27 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1140/1,550; native-valid: 1140; fallbacks: 0; statuses: `{"success": 1063, "success_with_tool_failure": 77}`; failed crop actions: 77. Runner PID 33749: 33749    02:23:40 Ssl.

## 2026-09-27 23:28 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1145/1,550; native-valid: 1145; fallbacks: 0; statuses: `{"success": 1068, "success_with_tool_failure": 77}`; failed crop actions: 77. Runner PID 33749: 33749    02:24:54 Ssl.

## 2026-09-27 23:29 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1150/1,550; native-valid: 1150; fallbacks: 0; statuses: `{"success": 1073, "success_with_tool_failure": 77}`; failed crop actions: 77. Runner PID 33749: 33749    02:26:10 Ssl.

## 2026-09-27 23:31 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1155/1,550; native-valid: 1155; fallbacks: 0; statuses: `{"success": 1078, "success_with_tool_failure": 77}`; failed crop actions: 77. Runner PID 33749: 33749    02:27:23 Ssl.

## 2026-09-27 23:32 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1160/1,550; native-valid: 1160; fallbacks: 0; statuses: `{"success": 1083, "success_with_tool_failure": 77}`; failed crop actions: 77. Runner PID 33749: 33749    02:28:37 Ssl.

## 2026-09-27 23:33 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1165/1,550; native-valid: 1165; fallbacks: 0; statuses: `{"success": 1088, "success_with_tool_failure": 77}`; failed crop actions: 77. Runner PID 33749: 33749    02:29:50 Ssl.

## 2026-09-27 23:34 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1171/1,550; native-valid: 1171; fallbacks: 0; statuses: `{"success": 1094, "success_with_tool_failure": 77}`; failed crop actions: 77. Runner PID 33749: 33749    02:31:06 Ssl.

## 2026-09-27 23:35 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1176/1,550; native-valid: 1176; fallbacks: 0; statuses: `{"success": 1099, "success_with_tool_failure": 77}`; failed crop actions: 77. Runner PID 33749: 33749    02:32:19 Ssl.

## 2026-09-27 23:37 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1181/1,550; native-valid: 1181; fallbacks: 0; statuses: `{"success": 1104, "success_with_tool_failure": 77}`; failed crop actions: 77. Runner PID 33749: 33749    02:33:33 Rsl.

## 2026-09-27 23:38 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1185/1,550; native-valid: 1185; fallbacks: 0; statuses: `{"success": 1107, "success_with_tool_failure": 78}`; failed crop actions: 78. Runner PID 33749: 33749    02:34:51 Ssl.
- Newly reconciled qid 10306: crop `{'video_path': '/path/to/storage/longvt-repro/benchmark_data/QVHighlights/videos_full/videos/Ud_RsVt_9Rs_210.0_360.0.mp4', 'start_time': 210.0, 'end_time': 360.0}` rejected with `ValueError('crop_start_outside_official_local_duration')`; prediction native-valid, no fallback.

## 2026-09-27 23:39 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1191/1,550; native-valid: 1191; fallbacks: 0; statuses: `{"success": 1113, "success_with_tool_failure": 78}`; failed crop actions: 78. Runner PID 33749: 33749    02:36:06 Ssl.

## 2026-09-27 23:40 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1197/1,550; native-valid: 1197; fallbacks: 0; statuses: `{"success": 1118, "success_with_tool_failure": 79}`; failed crop actions: 79. Runner PID 33749: 33749    02:37:20 Ssl.
- Newly reconciled qid 4474: crop `{'video_path': '/path/to/storage/longvt-repro/benchmark_data/QVHighlights/videos_full/videos/2X46BBkcCeY_60.0_210.0.mp4', 'start_time': 150.0, 'end_time': 155.0}` rejected with `ValueError('crop_start_outside_official_local_duration')`; prediction native-valid, no fallback.

## 2026-09-27 23:42 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1201/1,550; native-valid: 1201; fallbacks: 0; statuses: `{"success": 1121, "success_with_tool_failure": 80}`; failed crop actions: 80. Runner PID 33749: 33749    02:38:34 Ssl.
- Newly reconciled qid 5388: crop `{'video_path': '/path/to/storage/longvt-repro/benchmark_data/QVHighlights/videos_full/videos/ZjFzkhrqIZs_210.0_360.0.mp4', 'start_time': 345.0, 'end_time': 355.0}` rejected with `ValueError('crop_start_outside_official_local_duration')`; prediction native-valid, no fallback.

## 2026-09-27 23:43 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1207/1,550; native-valid: 1207; fallbacks: 0; statuses: `{"success": 1127, "success_with_tool_failure": 80}`; failed crop actions: 80. Runner PID 33749: 33749    02:39:48 Rsl.

## 2026-09-27 23:44 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1211/1,550; native-valid: 1211; fallbacks: 0; statuses: `{"success": 1131, "success_with_tool_failure": 80}`; failed crop actions: 80. Runner PID 33749: 33749    02:41:01 Ssl.

## 2026-09-27 23:45 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1215/1,550; native-valid: 1215; fallbacks: 0; statuses: `{"success": 1134, "success_with_tool_failure": 81}`; failed crop actions: 81. Runner PID 33749: 33749    02:42:14 Ssl.
- Newly reconciled qid 4564: crop `{'video_path': '/path/to/storage/longvt-repro/benchmark_data/QVHighlights/videos_full/videos/w1UC0rFTPcM_210.0_360.0.mp4', 'start_time': 150.0, 'end_time': 170.0}` rejected with `ValueError('crop_start_outside_official_local_duration')`; prediction native-valid, no fallback.

## 2026-09-27 23:47 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1221/1,550; native-valid: 1221; fallbacks: 0; statuses: `{"success": 1139, "success_with_tool_failure": 82}`; failed crop actions: 82. Runner PID 33749: 33749    02:43:28 Ssl.
- Newly reconciled qid 401: crop `{'video_path': '/path/to/storage/longvt-repro/benchmark_data/QVHighlights/videos_full/videos/PMm8xTk8ofI_210.0_360.0.mp4', 'start_time': 210.0, 'end_time': 360.0}` rejected with `ValueError('crop_start_outside_official_local_duration')`; prediction native-valid, no fallback.

## 2026-09-27 23:48 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1226/1,550; native-valid: 1226; fallbacks: 0; statuses: `{"success": 1144, "success_with_tool_failure": 82}`; failed crop actions: 82. Runner PID 33749: 33749    02:44:41 Ssl.

## 2026-09-27 23:49 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1232/1,550; native-valid: 1232; fallbacks: 0; statuses: `{"success": 1149, "success_with_tool_failure": 83}`; failed crop actions: 83. Runner PID 33749: 33749    02:45:55 Ssl.
- Newly reconciled qid 3819: crop `{'video_path': '/path/to/storage/longvt-repro/benchmark_data/QVHighlights/videos_full/videos/uE1ciE0PCIU_210.0_360.0.mp4', 'start_time': 150.0, 'end_time': 160.0}` rejected with `ValueError('crop_start_outside_official_local_duration')`; prediction native-valid, no fallback.

## 2026-09-27 23:50 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1237/1,550; native-valid: 1237; fallbacks: 0; statuses: `{"success": 1153, "success_with_tool_failure": 84}`; failed crop actions: 84. Runner PID 33749: 33749    02:47:09 Ssl.
- Newly reconciled qid 9969: crop `{'video_path': '/path/to/storage/longvt-repro/benchmark_data/QVHighlights/videos_full/videos/iWBU-K5dbhE_210.0_360.0.mp4', 'start_time': 210.0, 'end_time': 360.0}` rejected with `ValueError('crop_start_outside_official_local_duration')`; prediction native-valid, no fallback.

## 2026-09-27 23:52 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1242/1,550; native-valid: 1242; fallbacks: 0; statuses: `{"success": 1158, "success_with_tool_failure": 84}`; failed crop actions: 84. Runner PID 33749: 33749    02:48:23 Rsl.

## 2026-09-27 23:53 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1246/1,550; native-valid: 1246; fallbacks: 0; statuses: `{"success": 1161, "success_with_tool_failure": 85}`; failed crop actions: 85. Runner PID 33749: 33749    02:49:36 Ssl.

## 2026-09-27 23:54 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1250/1,550; native-valid: 1250; fallbacks: 0; statuses: `{"success": 1165, "success_with_tool_failure": 85}`; failed crop actions: 85. Runner PID 33749: 33749    02:50:50 Ssl.

## 2026-09-27 23:55 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1256/1,550; native-valid: 1256; fallbacks: 0; statuses: `{"success": 1171, "success_with_tool_failure": 85}`; failed crop actions: 85. Runner PID 33749: 33749    02:52:04 Ssl.

## 2026-09-27 23:56 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1260/1,550; native-valid: 1260; fallbacks: 0; statuses: `{"success": 1175, "success_with_tool_failure": 85}`; failed crop actions: 85. Runner PID 33749: 33749    02:53:18 Rsl.

## 2026-09-27 23:58 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1266/1,550; native-valid: 1266; fallbacks: 0; statuses: `{"success": 1181, "success_with_tool_failure": 85}`; failed crop actions: 85. Runner PID 33749: 33749    02:54:33 Ssl.

## 2026-09-27 23:59 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1271/1,550; native-valid: 1271; fallbacks: 0; statuses: `{"success": 1186, "success_with_tool_failure": 85}`; failed crop actions: 85. Runner PID 33749: 33749    02:55:49 Ssl.

## 2026-09-28 00:00 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1277/1,550; native-valid: 1277; fallbacks: 0; statuses: `{"success": 1192, "success_with_tool_failure": 85}`; failed crop actions: 85. Runner PID 33749: 33749    02:57:04 Ssl.

## 2026-09-28 00:01 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1282/1,550; native-valid: 1282; fallbacks: 0; statuses: `{"success": 1197, "success_with_tool_failure": 85}`; failed crop actions: 85. Runner PID 33749: 33749    02:58:18 Ssl.

## 2026-09-28 00:03 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1288/1,550; native-valid: 1288; fallbacks: 0; statuses: `{"success": 1203, "success_with_tool_failure": 85}`; failed crop actions: 85. Runner PID 33749: 33749    02:59:33 Ssl.

## 2026-09-28 00:04 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1294/1,550; native-valid: 1294; fallbacks: 0; statuses: `{"success": 1207, "success_with_tool_failure": 87}`; failed crop actions: 87. Runner PID 33749: 33749    03:00:50 Ssl.
- Newly reconciled qid 4041: crop `{'video_path': '/path/to/storage/longvt-repro/benchmark_data/QVHighlights/videos_full/videos/A0JmK8Dwv3Q_360.0_510.0.mp4', 'start_time': 150.0, 'end_time': 158.0}` rejected with `ValueError('crop_start_outside_official_local_duration')`; prediction native-valid, no fallback.
- Newly reconciled qid 4094: crop `{'video_path': '/path/to/storage/longvt-repro/benchmark_data/QVHighlights/videos_full/videos/A0JmK8Dwv3Q_210.0_360.0.mp4', 'start_time': 150.0, 'end_time': 160.0}` rejected with `ValueError('crop_start_outside_official_local_duration')`; prediction native-valid, no fallback.

## 2026-09-28 00:05 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1300/1,550; native-valid: 1300; fallbacks: 0; statuses: `{"success": 1212, "success_with_tool_failure": 88}`; failed crop actions: 88. Runner PID 33749: 33749    03:02:16 Rsl.
- Newly reconciled qid 4505: crop `{'video_path': '/path/to/storage/longvt-repro/benchmark_data/QVHighlights/videos_full/videos/Ifgq2XU0G_s_210.0_360.0.mp4', 'start_time': 150.0, 'end_time': 155.0}` rejected with `ValueError('crop_start_outside_official_local_duration')`; prediction native-valid, no fallback.

## 2026-09-28 00:07 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1305/1,550; native-valid: 1305; fallbacks: 0; statuses: `{"success": 1217, "success_with_tool_failure": 88}`; failed crop actions: 88. Runner PID 33749: 33749    03:03:47 Ssl.

## 2026-09-28 00:08 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1311/1,550; native-valid: 1311; fallbacks: 0; statuses: `{"success": 1223, "success_with_tool_failure": 88}`; failed crop actions: 88. Runner PID 33749: 33749    03:05:11 Ssl.

## 2026-09-28 00:10 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1317/1,550; native-valid: 1317; fallbacks: 0; statuses: `{"success": 1229, "success_with_tool_failure": 88}`; failed crop actions: 88. Runner PID 33749: 33749    03:06:35 Ssl.

## 2026-09-28 00:11 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1323/1,550; native-valid: 1323; fallbacks: 0; statuses: `{"success": 1235, "success_with_tool_failure": 88}`; failed crop actions: 88. Runner PID 33749: 33749    03:07:57 Ssl.

## 2026-09-28 00:12 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1329/1,550; native-valid: 1329; fallbacks: 0; statuses: `{"success": 1239, "success_with_tool_failure": 90}`; failed crop actions: 90. Runner PID 33749: 33749    03:09:16 Ssl.
- Newly reconciled qid 10150: crop `{'video_path': '/path/to/storage/longvt-repro/benchmark_data/QVHighlights/videos_full/videos/zTf-3BOLi8w_210.0_360.0.mp4', 'start_time': 210.0, 'end_time': 360.0}` rejected with `ValueError('crop_start_outside_official_local_duration')`; prediction native-valid, no fallback.
- Newly reconciled qid 10302: crop `{'video_path': '/path/to/storage/longvt-repro/benchmark_data/QVHighlights/videos_full/videos/zTf-3BOLi8w_60.0_210.0.mp4', 'start_time': 160.0, 'end_time': 170.0}` rejected with `ValueError('crop_start_outside_official_local_duration')`; prediction native-valid, no fallback.

## 2026-09-28 00:14 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1334/1,550; native-valid: 1334; fallbacks: 0; statuses: `{"success": 1243, "success_with_tool_failure": 91}`; failed crop actions: 91. Runner PID 33749: 33749    03:10:37 Ssl.
- Newly reconciled qid 8574: crop `{'video_path': '/path/to/storage/longvt-repro/benchmark_data/QVHighlights/videos_full/videos/zKs0TO6ZTBI_510.0_660.0.mp4', 'start_time': 150.0, 'end_time': 160.0}` rejected with `ValueError('crop_start_outside_official_local_duration')`; prediction native-valid, no fallback.

## 2026-09-28 00:15 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1340/1,550; native-valid: 1340; fallbacks: 0; statuses: `{"success": 1249, "success_with_tool_failure": 91}`; failed crop actions: 91. Runner PID 33749: 33749    03:12:05 Ssl.

## 2026-09-28 00:17 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1346/1,550; native-valid: 1346; fallbacks: 0; statuses: `{"success": 1255, "success_with_tool_failure": 91}`; failed crop actions: 91. Runner PID 33749: 33749    03:13:25 Ssl.

## 2026-09-28 00:18 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1353/1,550; native-valid: 1353; fallbacks: 0; statuses: `{"success": 1262, "success_with_tool_failure": 91}`; failed crop actions: 91. Runner PID 33749: 33749    03:15:06 Rsl.

## 2026-09-28 00:20 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1359/1,550; native-valid: 1359; fallbacks: 0; statuses: `{"success": 1267, "success_with_tool_failure": 92}`; failed crop actions: 92. Runner PID 33749: 33749    03:16:23 Ssl.
- Newly reconciled qid 8874: crop `{'video_path': '/path/to/storage/longvt-repro/benchmark_data/QVHighlights/videos_full/videos/DLSkojSiAPs_660.0_810.0.mp4', 'start_time': 138.0, 'end_time': 142.0}` rejected with `ValueError('crop_start_outside_official_local_duration')`; prediction native-valid, no fallback.

## 2026-09-28 00:21 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1363/1,550; native-valid: 1363; fallbacks: 0; statuses: `{"success": 1271, "success_with_tool_failure": 92}`; failed crop actions: 92. Runner PID 33749: 33749    03:17:37 Ssl.

## 2026-09-28 00:22 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1368/1,550; native-valid: 1368; fallbacks: 0; statuses: `{"success": 1276, "success_with_tool_failure": 92}`; failed crop actions: 92. Runner PID 33749: 33749    03:18:52 Rsl.

## 2026-09-28 00:23 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1373/1,550; native-valid: 1373; fallbacks: 0; statuses: `{"success": 1281, "success_with_tool_failure": 92}`; failed crop actions: 92. Runner PID 33749: 33749    03:20:10 Ssl.

## 2026-09-28 00:25 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1377/1,550; native-valid: 1377; fallbacks: 0; statuses: `{"success": 1285, "success_with_tool_failure": 92}`; failed crop actions: 92. Runner PID 33749: 33749    03:21:29 Ssl.

## 2026-09-28 00:26 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1383/1,550; native-valid: 1383; fallbacks: 0; statuses: `{"success": 1291, "success_with_tool_failure": 92}`; failed crop actions: 92. Runner PID 33749: 33749    03:22:48 Ssl.

## 2026-09-28 00:27 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1388/1,550; native-valid: 1388; fallbacks: 0; statuses: `{"success": 1296, "success_with_tool_failure": 92}`; failed crop actions: 92. Runner PID 33749: 33749    03:24:06 Ssl.

## 2026-09-28 00:29 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1395/1,550; native-valid: 1395; fallbacks: 0; statuses: `{"success": 1303, "success_with_tool_failure": 92}`; failed crop actions: 92. Runner PID 33749: 33749    03:25:31 Ssl.

## 2026-09-28 00:30 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1401/1,550; native-valid: 1401; fallbacks: 0; statuses: `{"success": 1307, "success_with_tool_failure": 94}`; failed crop actions: 94. Runner PID 33749: 33749    03:26:47 Rsl.
- Newly reconciled qid 7343: crop `{'video_path': '/path/to/storage/longvt-repro/benchmark_data/QVHighlights/videos_full/videos/bEmdM0Peb_o_510.0_660.0.mp4', 'start_time': 510.0, 'end_time': 660.0}` rejected with `ValueError('crop_start_outside_official_local_duration')`; prediction native-valid, no fallback.
- Newly reconciled qid 2245: crop `{'video_path': '/path/to/storage/longvt-repro/benchmark_data/QVHighlights/videos_full/videos/yuj1y5__jXg_210.0_360.0.mp4', 'start_time': 210.0, 'end_time': 260.0}` rejected with `ValueError('crop_start_outside_official_local_duration')`; prediction native-valid, no fallback.

## 2026-09-28 00:31 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1407/1,550; native-valid: 1407; fallbacks: 0; statuses: `{"success": 1313, "success_with_tool_failure": 94}`; failed crop actions: 94. Runner PID 33749: 33749    03:28:09 Rsl.

## 2026-09-28 00:33 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1412/1,550; native-valid: 1412; fallbacks: 0; statuses: `{"success": 1318, "success_with_tool_failure": 94}`; failed crop actions: 94. Runner PID 33749: 33749    03:29:23 Ssl.

## 2026-09-28 00:34 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1418/1,550; native-valid: 1418; fallbacks: 0; statuses: `{"success": 1324, "success_with_tool_failure": 94}`; failed crop actions: 94. Runner PID 33749: 33749    03:30:41 Ssl.

## 2026-09-28 00:35 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1423/1,550; native-valid: 1423; fallbacks: 0; statuses: `{"success": 1329, "success_with_tool_failure": 94}`; failed crop actions: 94. Runner PID 33749: 33749    03:32:04 Ssl.

## 2026-09-28 00:37 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1427/1,550; native-valid: 1427; fallbacks: 0; statuses: `{"success": 1332, "success_with_tool_failure": 95}`; failed crop actions: 95. Runner PID 33749: 33749    03:33:23 Ssl.
- Newly reconciled qid 4309: crop `{'video_path': '/path/to/storage/longvt-repro/benchmark_data/QVHighlights/videos_full/videos/xHypiQpL2yQ_210.0_360.0.mp4', 'start_time': 220.0, 'end_time': 232.0}` rejected with `ValueError('crop_start_outside_official_local_duration')`; prediction native-valid, no fallback.

## 2026-09-28 00:38 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1433/1,550; native-valid: 1433; fallbacks: 0; statuses: `{"success": 1337, "success_with_tool_failure": 96}`; failed crop actions: 96. Runner PID 33749: 33749    03:34:38 Ssl.
- Newly reconciled qid 9858: crop `{'video_path': '/path/to/storage/longvt-repro/benchmark_data/QVHighlights/videos_full/videos/xivzc8l7O_A_210.0_360.0.mp4', 'start_time': 150.0, 'end_time': 170.0}` rejected with `ValueError('crop_start_outside_official_local_duration')`; prediction native-valid, no fallback.

## 2026-09-28 00:39 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1437/1,550; native-valid: 1437; fallbacks: 0; statuses: `{"success": 1341, "success_with_tool_failure": 96}`; failed crop actions: 96. Runner PID 33749: 33749    03:35:56 Ssl.

## 2026-09-28 00:40 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1442/1,550; native-valid: 1442; fallbacks: 0; statuses: `{"success": 1346, "success_with_tool_failure": 96}`; failed crop actions: 96. Runner PID 33749: 33749    03:37:14 Ssl.

## 2026-09-28 00:42 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1448/1,550; native-valid: 1448; fallbacks: 0; statuses: `{"success": 1351, "success_with_tool_failure": 97}`; failed crop actions: 97. Runner PID 33749: 33749    03:38:29 Ssl.
- Newly reconciled qid 6418: crop `{'video_path': '/path/to/storage/longvt-repro/benchmark_data/QVHighlights/videos_full/videos/EE0NcEec_mo_360.0_510.0.mp4', 'start_time': 150.0, 'end_time': 170.0}` rejected with `ValueError('crop_start_outside_official_local_duration')`; prediction native-valid, no fallback.

## 2026-09-28 00:43 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1453/1,550; native-valid: 1453; fallbacks: 0; statuses: `{"success": 1356, "success_with_tool_failure": 97}`; failed crop actions: 97. Runner PID 33749: 33749    03:39:50 Ssl.

## 2026-09-28 00:45 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1459/1,550; native-valid: 1459; fallbacks: 0; statuses: `{"success": 1361, "success_with_tool_failure": 98}`; failed crop actions: 98. Runner PID 33749: 33749    03:41:25 Ssl.
- Newly reconciled qid 6518: crop `{'video_path': '/path/to/storage/longvt-repro/benchmark_data/QVHighlights/videos_full/videos/S4z8QOxZisc_360.0_510.0.mp4', 'start_time': 150.0, 'end_time': 154.0}` rejected with `ValueError('crop_start_outside_official_local_duration')`; prediction native-valid, no fallback.

## 2026-09-28 00:46 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1463/1,550; native-valid: 1463; fallbacks: 0; statuses: `{"success": 1363, "success_with_tool_failure": 100}`; failed crop actions: 100. Runner PID 33749: 33749    03:42:45 Ssl.
- Newly reconciled qid 5490: crop `{'video_path': '/path/to/storage/longvt-repro/benchmark_data/QVHighlights/videos_full/videos/WtAOsyldvJk_360.0_510.0.mp4', 'start_time': 150.0, 'end_time': 155.0}` rejected with `ValueError('crop_start_outside_official_local_duration')`; prediction native-valid, no fallback.
- Newly reconciled qid 9058: crop `{'video_path': '/path/to/storage/longvt-repro/benchmark_data/QVHighlights/videos_full/videos/WtAOsyldvJk_510.0_660.0.mp4', 'start_time': 150.0, 'end_time': 160.0}` rejected with `ValueError('crop_start_outside_official_local_duration')`; prediction native-valid, no fallback.

## 2026-09-28 00:47 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1469/1,550; native-valid: 1469; fallbacks: 0; statuses: `{"success": 1369, "success_with_tool_failure": 100}`; failed crop actions: 100. Runner PID 33749: 33749    03:44:13 Ssl.

## 2026-09-28 00:49 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1474/1,550; native-valid: 1474; fallbacks: 0; statuses: `{"success": 1374, "success_with_tool_failure": 100}`; failed crop actions: 100. Runner PID 33749: 33749    03:45:29 Ssl.

## 2026-09-28 00:50 Asia/Shanghai — detached Tool checkpoint (active)

- Durable unique traces: 1478/1,550; native-valid: 1478; fallbacks: 0; statuses: `{"success": 1378, "success_with_tool_failure": 100}`; failed crop actions: 100. Runner PID 33749: 33749    03:46:51 Ssl.

## 2026-09-28 01:39 Asia/Shanghai — final Tool reconciliation and official evaluation completed

- Tool validation is complete: 1550/1,550 unique qids; statuses `{"success": 1443, "success_with_tool_failure": 107}`; native-valid 1550; fallbacks 0. Exact qid coverage matches the validation manifest and predictions file.
- Final failure-ledger reconciliation: 107 rejected crop calls total (106 official-local-duration boundary rejects; 1 current-video path mismatch reject). Earlier in-progress notes that described every failure as a duration-boundary reject were provisional; qid 6658 is the single path-mismatch case. All 107 predictions remained native-valid without fallback.
- Reconciled 55 rejected-call rows whose qids were not previously recorded as standalone ledger entries; arguments are reduced to requested start/end seconds.
  - qid 8014: `ValueError('crop_start_outside_official_local_duration')`, requested `[ 150.0, 160.0 ]`.
  - qid 7735: `ValueError('crop_start_outside_official_local_duration')`, requested `[ 210.0, 260.0 ]`.
  - qid 1136: `ValueError('crop_start_outside_official_local_duration')`, requested `[ 210.0, 360.0 ]`.
  - qid 5510: `ValueError('crop_start_outside_official_local_duration')`, requested `[ 150.0, 150.0 ]`.
  - qid 3834: `ValueError('crop_start_outside_official_local_duration')`, requested `[ 150.0, 170.0 ]`.
  - qid 3793: `ValueError('crop_start_outside_official_local_duration')`, requested `[ 210.0, 360.0 ]`.
  - qid 2253: `ValueError('crop_start_outside_official_local_duration')`, requested `[ 150.0, 155.0 ]`.
  - qid 9562: `ValueError('crop_start_outside_official_local_duration')`, requested `[ 150.0, 160.0 ]`.
  - qid 7604: `ValueError('crop_start_outside_official_local_duration')`, requested `[ 150.0, 160.0 ]`.
  - qid 525: `ValueError('crop_start_outside_official_local_duration')`, requested `[ 160.0, 170.0 ]`.
  - qid 5703: `ValueError('crop_start_outside_official_local_duration')`, requested `[ 350.0, 360.0 ]`.
  - qid 5690: `ValueError('crop_start_outside_official_local_duration')`, requested `[ 480.0, 510.0 ]`.
  - qid 4831: `ValueError('crop_start_outside_official_local_duration')`, requested `[ 150.0, 160.0 ]`.
  - qid 4495: `ValueError('crop_start_outside_official_local_duration')`, requested `[ 150.0, 153.0 ]`.
  - qid 10307: `ValueError('crop_start_outside_official_local_duration')`, requested `[ 160.0, 163.0 ]`.
  - qid 561: `ValueError('crop_start_outside_official_local_duration')`, requested `[ 150.0, 165.0 ]`.
  - qid 5672: `ValueError('crop_start_outside_official_local_duration')`, requested `[ 185.0, 200.0 ]`.
  - qid 3254: `ValueError('crop_start_outside_official_local_duration')`, requested `[ 150.0, 160.0 ]`.
  - qid 7567: `ValueError('crop_start_outside_official_local_duration')`, requested `[ 660.0, 810.0 ]`.
  - qid 5748: `ValueError('crop_start_outside_official_local_duration')`, requested `[ 210.0, 360.0 ]`.
  - qid 8730: `ValueError('crop_start_outside_official_local_duration')`, requested `[ 150.0, 160.0 ]`.
  - qid 5043: `ValueError('crop_start_outside_official_local_duration')`, requested `[ 150.0, 170.0 ]`.
  - qid 7065: `ValueError('crop_start_outside_official_local_duration')`, requested `[ 150.0, 160.0 ]`.
  - qid 7028: `ValueError('crop_start_outside_official_local_duration')`, requested `[ 150.0, 155.0 ]`.
  - qid 10157: `ValueError('crop_start_outside_official_local_duration')`, requested `[ 210.0, 360.0 ]`.
  - qid 5224: `ValueError('crop_start_outside_official_local_duration')`, requested `[ 150.0, 160.0 ]`.
  - qid 1487: `ValueError('crop_start_outside_official_local_duration')`, requested `[ 150.0, 170.0 ]`.
  - qid 2513: `ValueError('crop_start_outside_official_local_duration')`, requested `[ 150.0, 153.0 ]`.
  - qid 825: `ValueError('crop_start_outside_official_local_duration')`, requested `[ 210.0, 360.0 ]`.
  - qid 10220: `ValueError('crop_start_outside_official_local_duration')`, requested `[ 150.0, 153.0 ]`.
  - qid 4266: `ValueError('crop_start_outside_official_local_duration')`, requested `[ 150.0, 170.0 ]`.
  - qid 3748: `ValueError('crop_start_outside_official_local_duration')`, requested `[ 160.0, 170.0 ]`.
  - qid 5275: `ValueError('crop_start_outside_official_local_duration')`, requested `[ 290.0, 300.0 ]`.
  - qid 7548: `ValueError('crop_start_outside_official_local_duration')`, requested `[ 150.0, 170.0 ]`.
  - qid 6383: `ValueError('crop_start_outside_official_local_duration')`, requested `[ 270.0, 290.0 ]`.
  - qid 3658: `ValueError('crop_start_outside_official_local_duration')`, requested `[ 150.0, 180.0 ]`.
  - qid 8385: `ValueError('crop_start_outside_official_local_duration')`, requested `[ 158.0, 168.0 ]`.
  - qid 2600: `ValueError('crop_start_outside_official_local_duration')`, requested `[ 210.0, 230.0 ]`.
  - qid 4572: `ValueError('crop_start_outside_official_local_duration')`, requested `[ 350.0, 360.0 ]`.
  - qid 5537: `ValueError('crop_start_outside_official_local_duration')`, requested `[ 150.0, 160.0 ]`.
  - qid 9884: `ValueError('crop_start_outside_official_local_duration')`, requested `[ 220.0, 250.0 ]`.
  - qid 10217: `ValueError('crop_start_outside_official_local_duration')`, requested `[ 150.0, 153.0 ]`.
  - qid 9574: `ValueError('crop_start_outside_official_local_duration')`, requested `[ 290.0, 305.0 ]`.
  - qid 5758: `ValueError('crop_start_outside_official_local_duration')`, requested `[ 150.0, 180.0 ]`.
  - qid 1678: `ValueError('crop_start_outside_official_local_duration')`, requested `[ 150.0, 170.0 ]`.
  - qid 8340: `ValueError('crop_start_outside_official_local_duration')`, requested `[ 150.0, 160.0 ]`.
  - qid 8152: `ValueError('crop_start_outside_official_local_duration')`, requested `[ 210.0, 360.0 ]`.
  - qid 360: `ValueError('crop_start_outside_official_local_duration')`, requested `[ 222.0, 242.0 ]`.
  - qid 5787: `ValueError('crop_start_outside_official_local_duration')`, requested `[ 150.0, 170.0 ]`.
  - qid 166: `ValueError('crop_start_outside_official_local_duration')`, requested `[ 170.0, 190.0 ]`.
  - qid 2757: `ValueError('crop_start_outside_official_local_duration')`, requested `[ 150.0, 160.0 ]`.
  - qid 7203: `ValueError('crop_start_outside_official_local_duration')`, requested `[ 150.0, 160.0 ]`.
  - qid 1045: `ValueError('crop_start_outside_official_local_duration')`, requested `[ 210.0, 260.0 ]`.
  - qid 7635: `ValueError('crop_start_outside_official_local_duration')`, requested `[ 150.0, 158.0 ]`.
  - qid 193: `ValueError('crop_start_outside_official_local_duration')`, requested `[ 150.0, 155.0 ]`.
- Official Tool metrics saved to `results/full_v2b/longvt_tool/metrics.json`: Fair 33.38/35.81, Good 26.52/33.81, VeryGood 15.12/27.23 (mAP/Hit@1).
- Paired summary and source-video cluster bootstrap (10,000 replicates, seed 42, 474 source videos) saved to `analysis/full_v2b/paired_summary.json`; paired integrity/tie/tool-evidence audit and per-query CSV saved to `analysis/full_v2b/paired_audit.json` and `analysis/full_v2b/paired_diagnostics.csv`.
- Final Chinese report saved as `FINAL_REPORT_ZH.md`.

## 2026-09-28 01:40 Asia/Shanghai — report finalized and owned inference service stopped

- Updated README status from active to completed; wrote `FINAL_REPORT_ZH.md` with official results, paired cluster CIs, cost/evidence diagnostics, and protocol caveats.
- Sent SIGTERM only to verified experiment server PID 26273 after confirming its command line matched the frozen LongVT-RFT revision and port 18008. Port 18008 is closed; physical GPU 6/7 report 15 MiB and 0% utilization. No other process was stopped.
- Completed artifacts: Tool `metrics.json`, `analysis/full_v2b/paired_summary.json`, `paired_audit.json`, and `paired_diagnostics.csv`.
