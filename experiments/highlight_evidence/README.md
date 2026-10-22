# Highlight evidence prototype

> 发布版配置与依赖见[统一复现索引](../../docs/reproduction_index.md)。下文的运行编号、测试数量与硬件描述属于原实验记录；本轮发布不启动这些历史服务。

Experimental four-cue memory, bounded evidence acquisition and comparative scoring.
The new [R1/R2 frozen-memory reader experiment](../../docs/highlight_memory_reader_run_2026-10-17.md)
uses `memory_reader.py`: preserve the full text input, return known clip IDs and
citations, then conditionally shorten the target output to 15 clips per request.
It adds no memory construction, image input or tools. The current CPU suite has
136 passing tests. R1 completed all20 queries: VeryGood mAP/Hit@1 is 25.55/30.00,
versus cached R0 21.60/30.00 and D0 26.07/30.00. R1 passed the development gate,
so R2 was not run. The paired gain interval crosses zero; this is not independent
confirmation. All20 raw responses passed replay audit and GPUs4/5 were released.

```bash
PYTHONPATH=src:experiments python -m highlight_evidence.memory_reader prepare \
  --source /path/to/completed/full-memory-run --root /path/to/new/reader-run --repo "$PWD"
MERIT_READER_RUN=/path/to/new/reader-run
PYTHONPATH="$MERIT_READER_RUN/source/src:$MERIT_READER_RUN/source/experiments" \
  python -m highlight_evidence.memory_reader run --root "$MERIT_READER_RUN"
PYTHONPATH=src:experiments python -m highlight_evidence.memory_reader audit --root "$MERIT_READER_RUN"
```

A fixed twenty-query training pilot is integrated in `pilot.py`; see the
[run report](../../docs/highlight_evidence_pilot_2026-10-16.md) for its current status.
Integration does not establish an accuracy improvement. The frozen `src/merit_qvh`
and `hd_adapter` sources and outputs are unchanged.

The follow-up [matched memory-granularity protocol](../../docs/highlight_memory_granularity_2026-10-16.md)
uses `granularity.py`. C10 and F2 share frozen pixels, candidates, prior context,
scoring and request caps. Only observation supports differ. Original E1 remains
a historical reference. All 20 queries remain in evaluation; only the 14 with
existing candidates receive new inference. The run is complete: C10/F2 VeryGood
mAP is 25.62/25.26, Hit@1 is 35/35, with six/two new memory failures. Only six
queries completed both arms; no granularity benefit was established. See the
[archived artifacts](../../docs/highlight_memory_granularity_2026-10-16.md).
The current CPU suite has 83 passing tests.

## Implemented

- `memory.py`: configurable value facets, query-independent visual extraction,
  event intervals, frame citations, `before` / `possible_link` edges, separate
  event/value retrieval keys. `unknown` / `not_applicable` have null intensity.
  Graph validation checks provenance and temporal structure, not semantic truth.
- `planner.py`: explicit query criteria, temporal evidence coverage, cumulative
  limits, unique original-PTS sampling, adjacent candidate comparison scheduling,
  `all` / `topk` display selection, complete dense-vector updates.
- `controller.py`: check evidence → plan bounded views → decode → verify → stop.
  Failed views consume budget; unresolved criteria remain explicit. A sufficiently
  evidenced negative answer resolves a criterion without becoming a highlight.
- `media.py`: exact-frame PyAV decoding, maximum edge 672, JPEG quality 90. Dense
  sampling supplies additional frames; it does not synthesize slow motion.
- `inference.py`: optional live-client hooks for focused verification and pairwise
  comparison. Both keep the original query. Comparison decisions cite both events.
- `cli.py`: real-video CPU decoding smoke, with no model requests.
- `pilot_components.py`: contextual four-cue extraction, previous/current event
  references and links, query policy, three-way retrieval, gap compilation and scoring.
- `pilot.py`: frozen source/data/model audit, cached D0 alignment, four workers,
  E1/E2/E3 checkpoints, failure fallback, request accounting and official evaluation.

## Run and verify

Use the repository's Python 3.10+ inference environment and existing pinned
dependencies, including PyAV and FFmpeg. No dependency upgrade is needed.

```bash
python -m pytest -q tests/test_highlight_evidence.py
python -m ruff check experiments/highlight_evidence tests/test_highlight_evidence.py

PYTHONPATH=src:experiments python -m highlight_evidence.cli \
  --video /path/to/video.mp4 --duration 150 --start 40 --end 46 \
  --query 'A player celebrates after scoring' \
  --question 'Does the player raise both arms during this interval?' \
  --output /path/to/new/smoke-directory
```

The smoke writes original PTS, pixel hashes, source hashes and up to 24 JPEGs.
The output directory must be new. Real model hooks accept the existing
`hd_adapter.client.Client` interface; callers must supply a **new protocol and
request cache**, a verified endpoint, and actual image evidence. Tests use a
scripted fake client and do not measure VLM reasoning quality.

Initial prototype verification: 15 behavior tests and all 62 then-current CPU tests passed;
Ruff check/format passed. Pilot integration increased the suite to 75 passing tests.
A real six-second QVHighlights interval decoded into
24 distinct original frames with zero model calls. Frozen legacy sources/data
and all offline-analysis inputs still match their recorded hashes.

Default acquisition budget: two views, at most 24 new frames per view, 48 total;
up to three adjacent candidate comparisons. These are prototype caps. Transport
and format retries belong to the existing client and add actual attempts; memory
construction, scoring and comparisons are additional costs, not part of the
48-new-frame cap. Cumulative counters must persist if a caller resumes a query.

## Frozen pilot

Use the existing inference environment. Prepare a **new** directory, then run its
source snapshot. The server must serve the audited model at the frozen endpoint,
with the GPU UUIDs/configuration recorded in `protocol.json`.

```bash
PYTHONPATH=src:experiments python -m highlight_evidence.pilot prepare \
  --config configs/hd_adapter.local.json --baseline /path/to/frozen/hd-run \
  --root /path/to/new/pilot --repo "$PWD"

MERIT_PILOT=/path/to/new/pilot
CUDA_VISIBLE_DEVICES=4 \
  PYTHONPATH="$MERIT_PILOT/source/src:$MERIT_PILOT/source/experiments" \
  python -m highlight_evidence.pilot run --root "$MERIT_PILOT"

PYTHONPATH=src python scripts/analyze_highlight_pilot.py \
  --run-root "$MERIT_PILOT" --output /path/to/new/diagnosis
```

The run command resumes cached terminal queries/requests; failed responses stay
failed. A circuit-breaker file prevents restarting a failed engineering gate.
After correcting code, prepare a new protocol and keep the previous artifacts.
The currently launched systemd worker survives terminal disconnection, has a
90-minute cap, and stops its own model service on exit. It does not start a larger
experiment. A machine reboot requires inspecting status before explicitly resuming.

Selection is deterministic from the existing training exploration pool, without
looking at saliency labels. E1/E2/E3 are shared-compute checkpoints, not matched
ablations. New phases fall back to the last valid vector on error; this differs
from frozen H1's whole-query zero fallback and must be reported.

## Remaining research work

The user supplied four cue names: `event_cue`, `emotion_cue`, `novelty_cue`, and
`importance_cue`. The [configuration](../../configs/highlight_value_schema.example.json)
uses those names with proposed operational definitions; a copy is frozen in each
pilot protocol. Earlier discussion mentioned five types; no fifth is inferred.
These are coexisting evidence facets, not mutually exclusive event categories.
Novelty requires an observed reference; importance requires event/goal context,
with query-specific significance assessed online. Neither is established by
visual intensity alone. The pilot references previously observed events from two
chunks, with dual-endpoint citation and temporal checks on cross-window links.
Invalid optional links are omitted with rejection records and a `partial` status;
valid nodes remain usable. Gap questions choose from enumerated six-second views.
Cross-shot entity identity, long-range event chains, global novelty, score
calibration and matched-budget controls remain open. The pilot compares 10-second
window proxies; it does not yet produce consolidated event-level all/top-k outputs.
Pairwise results are evidence records; they do not arbitrarily add points to scores.
`all` applies to supplied candidates and is not an exhaustiveness claim.

See the [research proposal](../../docs/highlight_evidence_research_2026-10-16.md)
for design choices, controls, novelty limits and the staged experiment plan.

## Matched granularity experiment

Prepare a new directory from the completed twenty-query pilot. No embedding
model or source-video decoding is used; the existing representative JPEGs are
verified before inference. Use the source snapshot after preparing:

```bash
PYTHONPATH=src:experiments python -m highlight_evidence.granularity prepare \
  --source /path/to/frozen/pilot_v4 --root /path/to/new/granularity --repo "$PWD"

MERIT_GRANULARITY=/path/to/new/granularity
PYTHONPATH="$MERIT_GRANULARITY/source/src:$MERIT_GRANULARITY/source/experiments" \
  python -m highlight_evidence.granularity run --root "$MERIT_GRANULARITY"
```

The endpoint and authorized GPU UUIDs are frozen in `protocol.json`. With local
vLLM 0.12, verify those UUIDs then use integer indices in `CUDA_VISIBLE_DEVICES`.
Each arm uses three memory calls and one E1 call per active query. The shared
8,192-token memory cap accommodates five F2 observations; actual tokens can
differ. One observation per unit does not imply a distinct action per unit.
No retrieval reranking, source views, E2/E3, or experiment expansion is performed.

### Structured memory repair

Use `prepare --robust` in a new directory to freeze the v2 engineering repair.
Serve vLLM with `--structured-outputs-config '{"backend":"xgrammar","disable_any_whitespace":true}'`.
The controller supplies unit metadata; the model returns schema-constrained
observations. One bounded retry patches only invalid fields. A cue still lacking
valid support becomes explicit `unknown` with its rejected original retained.
Invalid descriptions fail their window; other windows continue. E1 sees the same
fixed candidate images and bins, and only successful windows update D0.

At the repair stage, the full CPU suite passed 94 tests. See the
[repair report](../../docs/highlight_memory_repair_2026-10-16.md) for the frozen
twenty-query rerun, separate completion/degradation accounting and provenance.

## C10 memory-consumption ablation

`python -m highlight_evidence.memory_ablation prepare --source /path/to/repair-run --root /path/to/new/run`
freezes the same twenty queries and audited C10 memory. A uses frames with empty
memory, B adds descriptions/timestamps/citations, C adds structured cues, and D
reuses the original complete C10 scorer result. Values are removed from both
selected and reference nodes in B; B/C/D keep the same node pool. Only A/B/C issue
new requests: 42 logical scorer calls, no new memory construction or source frames.

Run the prepared source snapshot, then use the `audit` action with a new external
`--output` file to check actual payloads and cached D provenance. At that stage the CPU
suite passed 99 tests. See the [ablation report](../../docs/highlight_memory_ablation_2026-10-16.md)
for controls, commands, costs and limitations.

## Full-video memory-only scoring

`python -m highlight_evidence.memory_only prepare --source /path/to/repair-run --root /path/to/new/run`
freezes all twenty original queries and their full midpoint-frame grids. Fresh
query-independent F2 memory covers 1,499 clips through 300 visual requests.
Each query then receives two full-video text-only scoring requests: descriptions
alone, and the identical descriptions with structured cues. No candidate selection,
tool decision, graph, source revisit, pairwise comparison, or D0 score enters inference.

Run from the frozen source paths, then call `memory_only audit --root ... --output ...`
to verify actual requests and source hashes. D0 is stored separately for evaluation.
An incomplete memory disables both scorers; a failed scorer affects only its arm.
Failures retain a declared flat 2.0 vector and stay in the full evaluation denominator.
Context overflow is an explicit failure, never silent truncation. Model-generated
cue descriptions are shared, so this is a reader-field ablation, not separate
construction protocols. The CPU suite passes 110 tests. See the
[memory-only report](../../docs/highlight_memory_only_2026-10-16.md).

The fixed20 run is complete: D0 / Memory_text / Memory_cue VeryGood mAP is
26.07 / 21.60 / 16.53, Hit@1 is 30 / 30 / 20. Both new arms completed all20;
four unsupported cues in two windows of one query were explicitly downgraded.
All 340 logical requests and 356 actual responses passed provenance/replay audits.
GPU services have exited. These explored training results do not support replacing
D0 with this memory-only implementation.
