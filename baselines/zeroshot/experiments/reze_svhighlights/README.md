# REZE original scorer on SVHighlights

> 发布版入口：[统一复现索引](https://github.com/irryxeek/MERIT-based-method/blob/main/docs/reproduction_index.md)。路径与 GPU 以本机配置为准，历史父运行缓存需另行提供。

This isolated workflow transfers the frozen QV B0 implementation to a fixed
seven-video SVHighlights development pilot. It preserves the Qwen3-VL-8B-Instruct
snapshot, strict Yes/No prompt, 3-second windows, six frames per full window,
resize, top-20 probability fallback, and inclusive-second score curve.
It adds no memory, reranking, sport cues, or model training.

## Dataset and scope

The local VideoARM store has 270 complete trimmed videos from the 320-video
dataset. Existing media/label-count receipts accept 201 and quarantine 69;
50 sources, including all basketball videos, are unavailable.

Select one video per available sport from the historical common 20-video
development prediction set, using the lowest `sha256("42:vid")`.
Selection reads prediction IDs and metadata, not metric or highlight values.
The seven videos total 15.84 hours. No automatic expansion beyond this pilot.

## Dataset interface

Use the official parser's query, `Highlight of this {sport} video.`.
REZE still scores with its original action-verification prompt. Thus this is a
cross-dataset transfer test, not a paper-reported SVHighlights reproduction.
SVHighlights retains a final one-second evaluation bin: mean the available
REZE seconds in each 2-second bin, preserving that final singleton. Check
every vector length exactly; never let the evaluator pad or truncate.

Frames are decoded in batches of 16 windows to bound memory. Frame selection,
resize pixels, metadata and score mapping are checked against frozen QV code.
Each batch saves raw probabilities, frame indices, PTS and image hashes.
Model, code, media, manifests and checkpoints have fixed identities.

## Execution

Use the existing inference environment; no dependency upgrades are needed.

```bash
PY=${REPRO_PYTHON}
cd ${ZEROSHOT_ROOT}/experiments/reze_svhighlights
"$PY" -m pytest tests -q
"$PY" prepare.py --run ${ZEROSHOT_ARTIFACT_ROOT}/reze_svhighlights_pilot7_gpu5_v2 \
  --previous-run ${ZEROSHOT_ARTIFACT_ROOT}/reze_svhighlights_pilot7_v1
```

Run `dispatch.py --run <directory>` from the frozen `source_snapshot/`.
Set `REPRO_GPU_UUIDS` before preparation. The registered recovery timer checks
only the frozen allocated GPU (GPU 5 in the current run), requires no compute
job and resumes committed chunks. It never stops unrelated jobs.
Three worker attempts and 12 active worker hours bound the run; interrupted
batches count against the cumulative request budget. Separate CPU evaluation
starts after the GPU worker exits. A completed or stopped run cannot relaunch.

Recovery v2 preserves the failed predecessor, subtracts its consumed active time
and model requests, and verifies numeric device addressing against the actual
torch UUID. Seven videos each receive first/middle/last CPU and GPU probes before
formal scoring: 21 engineering calls share the 19,062 total request cap with
19,014 formal windows and any interrupted work. No-progress timeout is 900 seconds.
GPU smoke completion is cached; its request reservations survive restart.

Read `status.json`, `worker_attempt_*.log`, `evaluation.json` and `REPORT.md`.
Compare REZE with stored Uniform-Refine AV and VideoARM AV predictions on exactly
these seven videos. Their inputs and budgets differ, and seven videos do not
support a full-dataset or statistical-significance claim.

Official sources: [SVHighlights](https://github.com/leedongkyu2019/SVHighlights),
[evaluation](https://github.com/leedongkyu2019/SVHighlights/blob/main/eval.py).
The unmodified evaluator and its license are included under `vendor/`.
