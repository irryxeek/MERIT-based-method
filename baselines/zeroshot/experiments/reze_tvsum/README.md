# REZE transfer to TVSum

> 发布版入口：[统一复现索引](../../../../docs/reproduction_index.md)。下文保留原实验协议和当时的执行记录；路径与 GPU 以本机配置为准，历史父运行缓存需另行提供。

This independent experiment applies the existing QV REZE B0 scorer with
Qwen3-VL-8B-Instruct to all 50 TVSum videos, without training or prompt tuning.
It is a cross-dataset task adaptation, not an original REZE paper TVSum result.

## Frozen inference

- Generic query: `Highlight of this video.` No title, category, label or caption
  enters the model. The action-verification prompt may be poorly suited to this
  query-free task; this baseline measures that transfer without changing it.
- Unmodified `vendor/reze_{core,backend,media}.py`, verified against QV B0.
- Three-second windows, 2 fps, six frames per full window, BF16, TP=1, one-token
  Yes/No readout, top-20 probability fallback and batch size 16.
- Preserve inclusive integer-second boundaries and adjacent-second averaging.
  The incomplete final 2-second bin is excluded exactly as in QV B0.

## Evaluation

Use the author's UMT `tvsum-ec05ad4e.zip` annotation bytes and fixed splits,
revision `bde05e3de39049bf90d8807f6b2a4ed8b42acbb0`. The archive SHA-256 is
`ec05ad4e26f2fb17cbe32b02bb16deb7745be200701e1569f5a18e20d0d47a23`.
All 50 label grids match the minimum author video/audio feature length and the
number of complete 2-second media intervals. Features are only checked for
sequence length; inference reads original video pixels.

Primary: Top-5 mAP on UMT's fixed validation 10, one video per category.
Secondary: all 50, explicitly including the 40 UMT training-split videos.
Neither split is used for training or tuning here. These two scopes must not
be combined in a comparison with published supervised-method scores.

The pinned UMT evaluator thresholds each of 20 annotators strictly above its
`torch.median`, sorts float32 predictions on CPU, and computes trapezoidal AP
within the predicted top five. Its positive denominator is the number of
positives *in that prefix*. This is not sklearn's usual full-ranking AP.
The implementation is checked against the unchanged author method body on
every completed shard. Copyright and license accompany the frozen source.

Report constant-score controls, exact expected random-ranking AP, paired
differences and 10,000 video-bootstrap resamples, seed 42. Preserve score ties.
Original frame-expanded MAT labels are not reconstructed into HD labels; the
two known extra trailing media frames remain untouched. This run does not
compute summary F1 or use MR.HiSum's mAP50/15 protocol.

## Execution

Use the existing inference virtual environment; no package installation:

```bash
PY=${REPRO_PYTHON}
CUDA_VISIBLE_DEVICES='' "$PY" experiments/reze_tvsum/dataset.py
CUDA_VISIBLE_DEVICES='' "$PY" -m pytest experiments/reze_tvsum/tests -q
"$PY" experiments/reze_tvsum/launch.py --run RUN_ROOT --unit-prefix reze-tvsum-full50-v1
"$PY" experiments/reze_tvsum/run_shard.py --run RUN_ROOT --shard-index 0
"$PY" experiments/reze_tvsum/aggregate_shards.py --run RUN_ROOT
```

Preparation freezes model/media hashes, annotation bytes, source, runtime and
exact shard membership. Each of two disjoint shards contains 25 videos. A CPU
native-processor check probes beginning/middle/end of every video before any
GPU call. Each worker then performs nine real GPU smoke calls.

Persistent user-systemd timers poll every minute. Each shard starts only on a
candidate with no compute process, <512 MiB used and <5% utilization. Shared
per-GPU file leases prevent both shards selecting the same card while its
model loads. At most two GPUs run; one free card can finish both shards
sequentially. Verify numeric CUDA index against UUID before engine loading.
Keep every completed 16-window checkpoint, failed attempt and request ledger.
Per shard: at most three worker attempts, two active worker hours, formal
requests plus 128 overhead, and a 900-second no-progress stop. Worker
heartbeats avoid counting a reboot's downtime as active GPU time. Systemd
continues after logout/reboot when the user manager and SSD are available.

GPU completion is followed by a separate CPU evaluator. The parent aggregator
requires exact disjoint coverage and matching protocol/file hashes, then
recomputes scopes and confidence intervals from per-video results.
Final artifacts: `evaluation.json`, `per_video_metrics.jsonl`,
`hd_predictions.jsonl`, `REPORT.md`, `complete.json`; live progress is in
parent/shard `status.json`. Existing MR.HiSum, QFVS and QV runs are untouched.

## Sources

- [UMT author implementation and fixed split](https://github.com/TencentARC/UMT/tree/bde05e3de39049bf90d8807f6b2a4ed8b42acbb0/datasets)
- [Author annotation archive](https://huggingface.co/yeliudev/UMT/resolve/main/datasets/tvsum-ec05ad4e.zip)
- [TVSum original release](https://github.com/yalesong/tvsum)
- [REZE Appendix A](https://arxiv.org/html/2608.04480v1)
