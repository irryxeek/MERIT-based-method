# REZE transfer to MR.HiSum

> 发布版入口：[统一复现索引](../../../../docs/reproduction_index.md)。下文保留原实验协议和当时的执行记录；路径与 GPU 以本机配置为准，历史父运行缓存需另行提供。

This workflow applies the unchanged local QV B0 REZE scorer to a query-free
highlight task using the fixed query `Highlight of this video.` No prompt
search, title/category conditioning, training, memory, reranking or revisit
is added. This is a task adaptation, not a published MR.HiSum result from REZE.

## Frozen sample and inference

Use the first 100 IDs by ascending `sha256("42:video_id")` among the 1,888
official test videos with verified downloads and duration difference <=2s.
The other 112 official test videos are unavailable (109) or require timeline
review (3). Video selection never reads label scores. The pilot does not
automatically expand to all 2,000 test entries.

Qwen3-VL-8B-Instruct BF16, revision
`0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`, uses 3-second windows, 2 fps,
six frames per complete window, the original strict action-verification
prompt and top-20 Yes/No probabilities. Keep the original inclusive integer
second curve and nearest fill. Decode only the shorter media/annotation
timeline and retain the full integer annotation grid.
Decord reports zero frame durations for one released VP9 video. Preserve its
frame starts and reconstruct missing ends from the next PTS (FPS for the last
frame). Record raw and effective timestamps. The first preflight attempt is
preserved as `reze_mrhisum_pilot100_v1`; corrected execution uses `v2` with the
same 100 IDs, query, labels and scorer, and no sample exclusions.

## Native evaluation

The separate CPU evaluator uses unchanged MR.HiSum helpers, revision
`aeb667f71239887034c2974062a341d312762867`:

- HD: average each consecutive five seconds, retaining the last partial bin;
  top floor(50%)/floor(15%) GT bins define positives. Report macro mAP50/mAP15.
  Preserve CPU float32, author exp normalization and `torch.argsort` tie behavior.
- Summary: report the author's 15%-budget KTS/knapsack F1. Preserve `end-start`
  shot lengths and original F1 zero padding; do not change shared endpoints.
  Apply that same zero padding before calling the full metric helper because
  its unused correlation branch otherwise crashes on unequal vector lengths.
  This secondary F1 is not the QFVS semantic F1.
- Diagnostic controls: constant-score AP/F1 and exact expected AP under uniform
  random rankings. Compute video bootstrap CIs with 10,000 draws and seed 42.

The fixed generic query may limit the original action-verification prompt.
Media checks cannot prove exact semantic alignment with historical videos.
Results on this 100-video available subset are not full-test benchmark scores.

## Execution and recovery

```bash
cd ${ZEROSHOT_ROOT}/experiments/reze_mrhisum
PY=${REPRO_PYTHON}
$PY -m pytest tests -q
$PY prepare.py --count 100 --run ${ZEROSHOT_ARTIFACT_ROOT}/reze_mrhisum_pilot100_v2
```

Run `preflight_media.py` and `dispatch.py` from the frozen `source_snapshot`.
The publication environment installs h5py through the root `requirements-reproduction.txt`; the historical isolated h5py directory is not required.

Preflight checks first/middle/last windows for all selected videos with the
native processor. GPU smoke covers nine windows from the first three videos.
Save every batch of 16 windows. Numeric CUDA indices are checked against UUIDs;
use at most one otherwise idle GPU from 6/7, never terminate another task.
The persistent dispatcher allows three worker attempts, two active hours and
planned requests plus 128 for smoke/recovery, and releases the GPU before CPU
evaluation. Completed batches are validated and resumed after interruption.

Read `binding.json`, `sampling_audit.json`, `status.json`, `gpu_smoke.json`,
`worker_attempt_*.log`, `predictions/`, `evaluation.json`, and `REPORT.md`.
Original download attempts, media, and previous experiment results are retained.

The v2 pilot completed all 100 videos in one GPU worker attempt (26.04 minutes):
mAP50 58.33%, mAP15 23.97%, summary F1 39.21%. Constant-score controls are
49.40%, 13.76%, and 56.83%; exact random-ranking expected AP is 53.67% / 21.04%.
HD ranking improves over these controls, while summary F1 is below the constant
control. These remain subset results, not full-test benchmark scores.

The initial standalone evaluation failed because it imported `model` before
`shared` installed the vendor path. The development entry point now initializes
that path explicitly; a fresh-process CLI regression test covers this case.
The frozen v2 sources remain unchanged. Its CPU recovery added
`<run>/source_snapshot/vendor` to `PYTHONPATH` alongside the h5py path above,
then ran `<run>/source_snapshot/evaluate.py --run <run>`. Original failure logs
and the recovery command, environment, and unchanged prediction/source hashes
are retained under `<run>/evaluation_attempts/`. No GPU requests were added.

## Full available test set on two GPUs

The full run covers all 1,888 verified test videos (94.4% of the official
2,000), with the same model, query, windows, frames, scorer and evaluation.
The 109 missing videos and three timeline-review videos remain explicitly
excluded. The original 100-video pilot is preserved; all full-run predictions
are generated afresh in a separate directory.

Sort eligible IDs by the existing hash rule, then split by list stride into
two disjoint 944-video shards. Each shard uses one independent model replica
and an otherwise idle GPU: physical GPU 4 for shard 0, GPU 5 for shard 1.
Each verifies the numeric CUDA index against its UUID before loading weights.
This is video-level parallel execution; model tensor parallelism remains one.

`run_shard.py --run <parent> --shard-index 0` reads the frozen `plan.json`,
prepares its shard with `--all-verified --shard-count 2 --shard-index 0`,
checks native input for every video, then starts the existing bounded worker.
The second service uses index 1. Each retains batch checkpoints, a separate
request ledger, three worker attempts, 12 cumulative GPU-hours and 256 extra
requests for smoke/recovery. Native input checks also retain per-video progress.
Completed shards do not restart.

`aggregate_shards.py --run <parent>` updates the combined progress and waits
for both shard evaluations. It verifies exact video coverage, matching
protocols, source and result hashes before generating the final report.
Metrics and bootstrap intervals are recomputed over all individual videos.
Reports distinguish elapsed inference time from cumulative GPU worker time.
The three systemd timers resume work after interruption and consolidate the
results automatically; completed runs and failed-attempt evidence are retained.

Full-set preparation v1 found one repeated start PTS at the end of
`video_31260`; it was held with zero GPU requests. The full timestamp audit
found no decreasing PTS among all 1,888 videos. Preparation v2 accepts equal
starts, preserves original timestamps, and retains REZE's existing right-sided
sampling (last frame with an equal start). Missing duration ends use the next
strictly later PTS. No videos or frames are deleted, no video is retimed, and
negative timestamp steps still fail validation. The original v1 artifacts are
preserved separately from the corrected v2 run.

Sources: [REZE Appendix A](https://arxiv.org/html/2608.04480v1),
[MR.HiSum source](https://github.com/MRHiSum/MR.HiSum/tree/aeb667f71239887034c2974062a341d312762867),
[MR.HiSum paper, Sec. 4.3](https://proceedings.neurips.cc/paper_files/paper/2023/file/7f880e3a325b06e3601af1384a653038-Paper-Datasets_and_Benchmarks.pdf).
