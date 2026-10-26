# REZE next-stage execution: V1 candidate diversity

V1 implements the candidate-selection contrast in the 2026-10-23 execution plan.
It reuses verified B0/C0 predictions and the unchanged P_pair comparator. C1
greedily suppresses clip-ID distance ≤2, then fills to five candidates. Only the
selected original score values are permuted. No new source frames, memory or
observer are included in this phase. V2 remains a subsequent implementation step.

Use the existing Python 3.10 inference environment and frozen model revision.
No dependency changes are needed. Run from the zero-shot repository root:

```bash
python -m pytest experiments/reze_agent_v2/tests -q
python -m ruff check experiments/reze_agent_v2
python experiments/reze_agent_v2/prepare_v1.py \
  --run "$V1_RUN" --baseline-run "$B0_RUN" --pair-run "$C0_RUN" \
  --gpu-uuid "$ALLOCATED_GPU_UUID" --after-run "$Q1_RUN" \
  --execution-plan "$MERIT_ROOT/docs/reze_next_stage_execution_plan_v2_2026-10-23.md"
python "$V1_RUN/source_snapshot/experiments/reze_agent_v2/dispatch_v1.py" --run "$V1_RUN"
```

Preparation replays all parent predictions, decodes and verifies C0/C1 pixels,
hashes weights, and freezes label-free manifests and source. The dispatcher waits
for the predecessor's terminal result and an idle assigned GPU, runs 100 engineering
requests and at most 2,000 unique formal comparisons, then evaluates all 100 queries.
Reservations survive interruption. Limits are 100 engineering requests, 2,100 formal
attempts, two GPU hours, three worker attempts and 900 seconds without progress.
Tokens are conservatively reserved before calls and bounded by the frozen B0 cost.

Use a user systemd timer to resume dispatch after disconnect/reboot. `status.json`,
`request_attempts/`, `worker_attempts.json` and `results/REPORT.md` record progress,
cost and results. `decision.json` selects C1 only for mAP ≥+1 pp, Hit@1 ≥−1 pp and
no extra fallback; otherwise V2 retains C0. This development decision does not
automatically launch confirmation or full-validation experiments.
