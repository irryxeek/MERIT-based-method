# REZE transfer to QFVS

> 发布版入口：[统一复现索引](../../../../docs/reproduction_index.md)。下文保留原实验协议和当时的执行记录；路径与 GPU 以本机配置为准，历史父运行缓存需另行提供。

This isolated workflow reuses the unchanged local QV B0 REZE scorer:
Qwen3-VL-8B-Instruct BF16, 3-second windows, 2 fps, six frames per full window,
native video processing, strict action-verification prompt and top-20 Yes/No
probabilities. It adds no training, memory, reranking or visual revisit.
QFVS is a new task interface, not a result published by the REZE paper.

## Scope and queries

`pilot4` selects one of 45 released queries per video, using the smallest
`sha256("42:vid:query_name")`, and covers all four complete videos.
`full180` uses all 180 released nonempty queries. Scope is frozen before
inference; label values and results never select queries or tune prompts.
The pilot does not automatically expand to full180.

Render a query as `The video shows either {concept_1} or {concept_2}.`
OR follows QFVS's definition that either query concept makes a shot relevant.
Expand only `Cupglass`, `Musicalinstrument` and `Petsanimal` to readable names.
The original REZE system and user templates are otherwise unchanged.

## Native readout and evaluation

Average the unchanged REZE integer-second curve within each native 5-second
shot. Use annotation counts, not a uniform ceil-duration rule: the final
P01–P03 shots have partial coverage; P04 has an unannotated 0.2-second tail.
There is no score padding, new annotation, interpolation change or vector
truncation to conceal a mismatch.

Select `floor(0.02 * native_shot_count)` shots, following pinned UniVTG's
default summary budget. Resolve equal scores by earlier shot index and retain
this choice in the protocol. Native semantic IoU and maximum-weight bipartite
matching produce precision, recall and F1. Main metrics use the oracle;
secondary metrics average the three human references. They are distinct
protocols. The only metric guard returns zero for no semantic overlap, where
the released helper otherwise divides by zero.

## Execution

```bash
cd ${ZEROSHOT_ROOT}/experiments/reze_qfvs
PY=${REPRO_PYTHON}
"$PY" -m pytest tests -q
"$PY" prepare.py --scope pilot4 --run ${ZEROSHOT_ARTIFACT_ROOT}/reze_qfvs_pilot4_v1
```

Run `preflight_media.py` and `dispatch.py` from the frozen `source_snapshot`.
The worker first verifies 12 actual GPU requests spanning video beginnings,
middles and ends, then scores all windows. Batches of 16 bound memory and
save frame IDs, PTS, pixel hashes, raw token probabilities and progress.
Code, media, model and label identities are frozen. Labels are parsed only
by a separate CPU evaluator after the inference process exits.

The persistent queue checks GPUs 4–7 and selects at most one with no other
compute process. vLLM 0.12 receives a numeric CUDA index; the worker verifies
the actual GPU UUID before loading weights. Existing tasks are not stopped.
The pilot has at most three worker attempts and 12 active worker hours.
Interrupted requests remain charged; already committed batches are validated
and reused. Completed and stopped runs do not relaunch.

Read `status.json`, `gpu_smoke.json`, `worker_attempt_*.log`,
`predictions/`, `evaluation.json` and `REPORT.md`.

Sources: [REZE Appendix A](https://arxiv.org/html/2608.04480v1),
[QFVS task and relevance definition](https://arxiv.org/html/1707.04960),
[pinned native evaluator](https://github.com/showlab/UniVTG/blob/32659ac7aeba21742a63274f30eba785fc57e247/eval/qfvs.py),
[2% default](https://github.com/showlab/UniVTG/blob/32659ac7aeba21742a63274f30eba785fc57e247/main/config.py#L177).
