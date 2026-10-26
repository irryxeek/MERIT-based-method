# ReVALUE v3: R0, R1 and R2

## Object-presence verifier probe (2026-10-27)

`object_presence_probe.py prepare|run|analyze` freezes 30 object claims on 15
source frames and compares direct vs. feature-check prompts. Reference labels
are held outside model text. It allows 60 calls, zero retries, 160 output tokens,
2,048 reserved tokens per call, 122,880 total, and one 900-second GPU lifecycle.
The fixture is `fixtures/object_presence_cases_v1.json`; use `--help` for paths.

All 60 calls completed. Both arms agreed with the pre-inference assistant
references on 24/30 cases and retained all 14 clear positives; both made definite
choices on all six ambiguous cases. Feature checking changed no status and used
20.70% more tokens. Actual-tokenizer CPU grammar checks confirmed all three
statuses were reachable. The verifier was not promoted; no new HD predictions
were produced. See the object-presence protocol/results in the project docs.
The full ReVALUE CPU suite now has 150 passing tests.

## Memory capacity and output-contract probes (2026-10-26)

`memory_codec.py` provides a deterministic, reversible source-table encoding;
its complete legend and tables count toward the 1,024-token memory limit.
`memory_compaction_probe.py --run NEW --grounding-run COMPLETED --protocol FILE`
audits frozen A/B/C views on CPU. All 14 views passed; same-fact tokens decreased
25.30%, and whole old entries restored 104 fact IDs across views. This has not
been tested for comparator interpretation or HD improvement.

`observer_output_probe.py prepare|run` performs the separately frozen 37-request
character-limit probe with unchanged source pixels and prompt. It enforces
37 attempts, zero retries, 2,048 reserved tokens per call, 75,776 total tokens,
and a single 900-second GPU lifecycle. `observer_output_review.py --run RUN`
renders source-verified, caption-free review sheets after completion.

All 37 requests completed. Removing the two character caps completed clipped
text but left unsupported visual claims; the observer was not promoted, and
A/B/C comparisons or a new 100-query evaluation were not launched. See the
memory-compaction results and both protocols in the project docs. Existing
production observers and frozen predictions remain unchanged.

## Temporal evidence repair and still-frame probe (2026-10-26)

`evidence_grounding.py` computes source PTS and target membership from verified
frames. Scoped context and boundary facts retain their observation anchor but
cannot serve as direct candidate citations. `memory.py` preserves this metadata
inside the token cap, and `comparison.py` validates it. New R2 bindings may opt
into `evidence_interface: temporal_v1`; historical/default behavior is preserved.

`frame_observer_probe.py prepare|worker|finish` freezes and runs an isolated
672-call diagnostic on all original observation frames. The still observer
duplicates each source frame for the existing video encoder and assigns source
references in code. This tests visible static content, not motion or HD accuracy.
`frame_observer_review.py` renders the preregistered review frames without model
calls. See the frame-grounding repair protocol and results in the project docs.

## Offline diagnosis and observer ablation (2026-10-26)

`r2_diagnose.py --run COMPLETED_R2 --output NEW_DIRECTORY --frames` checks the
completed evaluation, replays the frozen ranker for all 100 queries/four arms,
and prepares 23 case records plus source-verified review sheets for all seven
observed queries. It makes no model calls; source checks do not establish factual accuracy.

`r2_observation_ablation.py --run FROZEN_PROBE` executes a separately frozen
28-request question-exposure probe, preserving the original observation pixels,
PTS, model, schema and output caps. Only the observer question is neutralized;
the original question remains in audit metadata and is excluded from the rendered
model prompt. Use its immutable manifest/snapshot and retain the shared ledger.

The completed probe had 28 valid requests but continuing source/time citation
errors. It was not promoted into scoring, and no new HD metrics or R3/R4 run were
produced. See the research/publication docs dated 2026-10-26 for the offline
diagnosis, observation-probe protocol and results. Production prompts remain unchanged.

## R2: paired local observations (2026-10-25)

`r2_prepare.py`, `r2_worker.py`, `r2_dispatch.py` and `r2_evaluate.py` implement the
next frozen explore100 experiment: E0, C_direct4, E_memory, E_uniform, E_targeted.
R1 comparator/memory requests remain unchanged; exact verified R1 reuse contributes
to each method's cold cost. R2 observes only actionable uncertainty, with identical
questions and matched new-frame counts. All B0 source frames are excluded.
Both observation memories remain within 1,024 tokens, without raw-log access.

Use the inference environment to run `r2_prepare.py --help`; prepare a new directory
with the R1 run, passed observer run, execution plan and R2 implementation protocol.
Then run `r2_dispatch.py --run RUN` from `RUN/source_snapshot/experiments/revalue/`.
It adds idle GPUs dynamically, using exclusive query claims and a globally locked
request ledger. Total GPU time, including concurrent workers, is limited to six hours.
The original 9,200-request ceiling is reduced by verified exact reuse; retries remain
bounded. Keep all 100 predictions, including parent fallbacks on any budget stop.
It evaluates R2 and stops; it does not automatically launch R3/R4 or full validation.

R0/R1 instructions and their immutable history follow.

This isolated implementation preserves the historical REZE and candidate-dispersion experiments.
It implements the first two stages of the ReVALUE execution plan dated 2026-10-23.

## Current scope

- R0: strict TaskSpec, immutable visual facts, separate purpose judgments, bounded memory,
  four-state comparison, real Scan/Segment/Stitch observations, feasible event-selection
  and controller primitives. CPU fixtures cover the A/B/C/D feedback contracts. Fixed-twenty
  GPU engineering checks validate native 24-frame input, references and budget accounting.
- R1: E0 audited P_pair; C_direct4; M_plain; M_decision. Same five C0 candidates and old
  pixels, same comparator for the three new arms, at most 1,024 tokenizer tokens per memory.
  Only five original scores are permuted. No new R1 video evidence.
- In the original R0/R1 campaigns, R2/R3 adaptive model policies were **not enabled**. A CPU controller primitive is not a
  demonstrated complete agent. Human semantic review remains blank until performed.

## Run

From the publication repository root, source `configs/paths.example.sh` with
local overrides, then change to `"$ZEROSHOT_ROOT"`. Set `REZE_MODEL_PATH` to the
verified local model, `QVH_EVALUATOR_ROOT` to the matching official evaluator,
`ZEROSHOT_ARTIFACT_ROOT` to historical runs, and `REPRO_PYTHON` to the audited
inference interpreter. `prepare.py` also accepts `--pair-run` and
`--candidate-decision` for explicit historical inputs.

These publication path adapters do not replace the original frozen experiment
snapshots. The fixture's metadata uses path aliases; its claims and labels are
unchanged. Source and publication hashes are recorded in the publication manifest.
The repository does not include the original model, media, or run caches.

Use a new output directory for any reproduction; the following names document
historical inputs. Launch from the zero-shot repository root:

```bash
"$REPRO_PYTHON" -m pytest experiments/revalue/tests -q
"$REPRO_PYTHON" experiments/revalue/prepare.py \
  --run $ZEROSHOT_ARTIFACT_ROOT/revalue_r0_observer_v5_2026-10-25 \
  --repair-from $ZEROSHOT_ARTIFACT_ROOT/revalue_r01_v4_2026-10-25 \
  --repair-plan $MERIT_ROOT/docs/revalue_observer_content_repair_v5_2026-10-25.md \
  --repair-observation-content \
  --execution-plan $MERIT_ROOT/docs/revalue_execution_plan_v3_2026-10-23.md
```

Run `dispatch.py --run RUN` **from RUN/source_snapshot/experiments/revalue/** after preparation.
New runs scan all GPUs by default (`gpu_candidates: "all"`), following the user's
2026-10-25 resource authorization. Use `prepare.py --gpu-uuids UUID ...` to freeze a
narrower allowlist when needed. A device must have no compute process, less than
512 MiB used memory and less than 5% utilization, then pass a second check immediately
before launch. Workers verify the chosen UUID. The dispatcher gates R1 on R0,
preserves attempt ledgers across restarts, and stops after R1 evaluation. Each stage
has a two-active-GPU-hour ceiling, three worker launches, and a 900-second stall limit.
Queue expiry is 72 hours. Failed formats are not regenerated; transient requests have
bounded, conservatively billed recovery. Semantic failures fall back to the audited E0
scores and remain in the twenty-query denominator.

This changes future preparations only. Existing frozen runs retain their original
GPU allowlists and sources: v4 R1 continues on GPU 4 with GPU 4/5 recovery eligibility.
The current worker holds an exclusive ledger lock; do not start concurrent workers
against the same stage. Future parallel runs must partition queries, keep separate
ledgers and aggregate total GPU time within the original budget. No GPU is reserved.

## Engineering repairs v2/v3/v4

The preserved v1 gate failed on three 512-token observation truncations and one
missing comparison citation; a 24-frame CUDA OOM also consumed a bounded retry.
V2 uses compact per-request grammar for comparisons/observations, bilateral decision
citations, short observation text, and 1 GiB KV cache with a 6,144-token context.
It preserves pixels, model and output caps. V2's new diagnostic RPC was rejected by
vLLM before its first generation. V3 uses nonfatal external NVML snapshots instead;
these are before/after device readings, not within-inference peaks.

Only the 160 unchanged successful TaskSpec/fact/memory requests may be reused after
full request/response hash and semantic verification. All 120 comparison and 20
observation calls are rerun. Prior attempts, tokens, active time and worker launches
carry over: the cumulative R0 limits remain 500 attempts and two GPU hours. One
declared startup-only exception permits a fourth R0 launch to correct v2's diagnostic
failure; its cost remains charged. The exception rejects requests with generated
outputs. R1 retains its three-launch limit.
R1 regenerates its own shared requests and retains its original budgets. Never remove
the failed run, clear its ledger, or repair an immutable snapshot in place.

V3 was stopped when real observations returned empty arrays despite passing the old
schema. V4 requires at least one source-cited observation, allowing explicit unknowns
instead of invented events; its R0 gate independently rejects empty facts. It reuses
160 shared requests and 42 unchanged verified comparisons, and generates 78 remaining
comparisons plus 20 observations. All 350 prior attempts and their active time remain
charged; 448 total attempts are planned, below the 500 cap. A separate declared repair
iteration raises R0's launch ceiling to five; R1 remains three and both time caps remain
two hours. The inherited frame manifest is rechecked against real pixels at runtime.

V4 observations were nonempty but all 69 entries were unknown. V5 restores visible
actions/context as facts and separates unresolved details. It is an isolated R0-only
diagnostic: reuse all 280 unchanged requests and regenerate only the 20 observations,
then stop. The historical v5 run preferred GPU 5. The already running v4 R1 uses its immutable snapshot and
continues unchanged; no duplicate R1 is launched. This declared sixth R0 launch keeps
the 500-attempt/two-hour ceilings (468 total attempts planned). Semantic correctness
and highlight benefit require separate evaluation even if visible facts return.

## Artifacts and interpretation

### R2 repair and rerun (2026-10-26)

The completed R2 v2 covered 100 queries but made zero source observations and
fell back on seven TaskSpec failures. Use `r2_repair_prepare.py` for the new v3
run; `r2_prepare.py` documents the historical v2 preparation and must be used
only from its original frozen snapshot.

V3 constrains TaskSpec to literal query spans and explicit output limits. An
empty LLM plan for a directional conflict gets a logged, deterministic bilateral
attribution probe; supported ties remain untouched. Model plans and controller
probes are reported separately. Compact grammar uses explicit prefix tries to
prevent the installed compiler from admitting unlisted mixed old/new fact IDs.
The semantic citation checks remain strict.

Run the fixed eight-task/one-conflict real-source engineering check with
`r2_repair_smoke.py --run SMOKE_RUN` using its frozen manifest and idle-GPU
environment. Its old comparison state is an engineering fixture, not a new
benchmark prediction. Require a passed gate before preparing the formal rerun:

```bash
python experiments/revalue/r2_repair_prepare.py \
  --run NEW_RUN --previous-run COMPLETED_R2 --smoke-run PASSED_SMOKE \
  --execution-plan ORIGINAL_PLAN.md --implementation-plan REPAIR_PROTOCOL.md
python NEW_RUN/source_snapshot/experiments/revalue/r2_dispatch.py --run NEW_RUN
```

The 100-query run retains the five conditions, old candidate pixels, observation
budget, official evaluator, six-GPU-hour ceiling and all-query denominator.
Reuse is limited to 558 unchanged fact/memory requests plus eight exact new-task
smoke requests. All dependent comparisons are regenerated. Actual nonzero paired
observation is now a gate requirement. Preserve both failed smoke and completed
historical run costs; this queue does not automatically start R3/R4.

`binding.json`, `candidate_inputs.jsonl`, frozen sources and model receipts bind the run.
`R0/gate.json` must pass before `formal_protocol.json` enables R1. `R1/attempts/` retains
raw prompts, evidence, responses, tokens and failures; `R1/ledger.json` charges shared work
to each independent method while reporting actual shared compute separately.
`R1/results/` contains official metrics, per-query predictions, paired source-video
bootstrap, Markdown report, anonymous evidence and blank human-review forms.

M_decision minus M_plain tests memory retention strategy. C_direct4 minus E0 includes the
comparison protocol and rank-rule change. Twenty previously used development queries do
not establish independent performance; R0 records no accuracy for prompt selection.
