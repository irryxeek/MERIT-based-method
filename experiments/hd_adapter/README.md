# MERIT dense highlight adapter

> 发布版配置与依赖见[统一复现索引](../../docs/reproduction_index.md)。下文的运行编号、测试数量与硬件描述属于原实验记录；本轮发布不启动这些历史服务。

This isolated experiment implements **M0** (the existing MERIT pipeline), **D0**
(dense visual initialization), and **H1** (D0 plus MERIT-guided local rescoring).
The original `src/merit_qvh/` and completed validation artifacts are unchanged.

## Protocol

- Use the pinned Qwen3-VL-8B-Instruct and Qwen3-Embedding-4B weights from the original run.
- Keep MERIT's 10-second memory, 1 fps, 448-pixel images, three keys, retrieval rounds,
  neighbor text and decision prompts. Run this pipeline once to obtain M0 and the
  retrieval trace shared by H1.
- D0 scores every complete 2-second bin using its nearest real midpoint frame,
  longest edge 672, JPEG quality 90, at most 24 images/request, and scores in `[0,4]`.
- H1 maps selected memory anchors to intersecting bins, takes their union, sorts by
  time, and batches up to 24. Each bin is updated once. Initialization and local
  rescoring use exactly the same JPEG bytes. Other scores are retained.
- H1 uses the original query and HD prompt, with the explicit selected IDs; initial
  scores do not affect retrieval. No VideoARM controller or denser local frames are added.
- Each group retains every planned qid. A failed group produces an explicitly
  marked full-length zero vector. A failed M0 final scorer can still supply a
  complete retrieval trace to H1; a failed retrieval cannot.

## Implementation and provenance

`pipeline.py` wraps the unchanged legacy pipeline. `client.py` keeps each query's
events separate, locks identical requests across workers and atomically checkpoints
every attempt. Terminal failures are not silently regenerated; interrupted requests
have at most two recoveries. One locked embedding model serves all query workers.

`media.py` and `prompts.py` derive from the frozen VideoARM comparison archive at
`comparisons/qvhighlights/2026-10-15_194620/videoarm/open_source_qvh/videoarm/qvhighlights/`.
The system/HD prompt strings and frame-selection behavior are retained. Media helper
imports, formatting and explicit non-strict adjacent-PTS `zip` calls were adapted.
HD score validation additionally enforces finite values in `[0,4]`.

The installed processor needs explicit `size.shortest_edge=65536` and
`size.longest_edge=1048576`; `max_pixels` alone was ineffective in local verification.
`processor_audit.py` checks effective grids and actual API prompt-token counts.
The service profiles 90 images at 448×448; actual HD requests have at most 24 images
of longest edge 672. Capacity is measured with the embedding model resident.

## Run and validate

Use the existing inference environment; no dependency upgrades are required.
In addition to the root dependencies, this adapter needs PyAV, ffprobe and vLLM.
Supply local model/data paths and historical exclusion manifests in a copy of the
original configuration, with the `hd` settings documented in the implementation report.
The local deployment uses `configs/hd_adapter.local.json`.

```bash
export PYTHONPATH="$PWD/src:$PWD/experiments"
"$MERIT_PY" -m pytest tests -q
"$MERIT_PY" -m ruff check experiments/hd_adapter tests/test_hd_adapter.py
"$MERIT_PY" -m hd_adapter.runner prepare --config configs/hd_adapter.local.json
"$MERIT_PY" -m hd_adapter.benchmark --config configs/hd_adapter.local.json --directory "$PREFLIGHT_DIR"
"$MERIT_PY" -m hd_adapter.processor_audit --config configs/hd_adapter.local.json --directory "$PREFLIGHT_DIR"
"$MERIT_PY" -m hd_adapter.runner freeze --config configs/hd_adapter.local.json
"$MERIT_PY" -m hd_adapter.runner audit-models --run-root "$RUN_ROOT"
"$MERIT_PY" -m hd_adapter.runner run --run-root "$RUN_ROOT"
```

The benchmark warms all payloads and replays 10 real initial batches plus two
90-image capacity requests. It selects the smallest failure-free concurrency within
5% of the best throughput. Its outputs are diagnostics, never experiment predictions.
Do not reuse a benchmark directory for another timing measurement: its request cache
would invalidate timing. This is not a complete cold-query speedup benchmark.

## Artifacts and unattended execution

`prepare` fixes 100 exploration queries and 100 source-disjoint confirmation queries,
excluding prior smoke/dev sources. GT is stored separately and read only by evaluation.
The runner executes only exploration. Confirmation requires a separate candidate decision.

Each run contains an immutable protocol/source/data snapshot, model audit, request
ledger, D0 checkpoints, legacy memories/queries, combined M0/D0/H1 results, progress,
session timing, and official evaluator predictions/reports. Actual stage costs count
unique requests and retries; shared computation must not be added twice.

The local user service `merit-hd-v2-supervisor` runs from the frozen source snapshot.
It manages only `merit-hd-v2-run-model` on assigned GPUs 4/5, verifies their UUIDs and
availability before model startup, resumes interrupted work, and stops its model on
completion or exhaustion. Limits: 100 queries, 6 hours, two worker restarts, and a
30-minute inactivity threshold. Three consecutive query failures stop new submissions
and create `circuit_breaker.json`; inspect it before any new attempt.

```bash
systemctl --user status merit-hd-v2-supervisor
journalctl --user -u merit-hd-v2-supervisor -n 30 --no-pager
```

Systemd user services survive client disconnection. Host failure or GPU reassignment
can still interrupt work; bounded recovery preserves attempts instead of guaranteeing
uninterrupted execution. No external API, tracking service, new training or full-val
execution is part of this experiment.
