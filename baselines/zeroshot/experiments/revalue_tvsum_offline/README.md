# TVSum offline diagnosis and TaskSpec intervention preparation

Request marker: `MERIT_TVSUM_OFFLINE_20261027_1500_V1`.
This directory contains a CPU-only auditor and contract tests. It has no model
runner. It reads all ten frozen videos and verifies the chain from task extraction
to comparison, plus fact selection, memory retention and citations. Existing
descriptive metrics, tie sensitivity and assistant review are reused. Labels are
not loaded and metrics are not recomputed.

The sole proposed change removes the exact pseudo-object constraint
`Highlight of this video` from the fixed generic TaskSpec. Other queries,
constraints, prompts, old frames, memory and output contracts are preserved or
rejected when they do not match the frozen scope. Prepared request specifications
are **not predictions or authorization to execute**.

Run from `baselines/zeroshot` using the existing inference environment:

```bash
TVSUM_PARENT="$ZEROSHOT_ARTIFACT_ROOT/revalue_tvsum_val10_v1_2026-10-27" \
  PYTHONDONTWRITEBYTECODE=1 "$REPRO_PYTHON" -m pytest \
  experiments/revalue_tvsum_offline/tests -q -p no:cacheprovider
PYTHONDONTWRITEBYTECODE=1 "$REPRO_PYTHON" experiments/revalue_tvsum_offline/audit.py \
  --parent "$ZEROSHOT_ARTIFACT_ROOT/revalue_tvsum_val10_v1_2026-10-27" \
  --output "$ZEROSHOT_ARTIFACT_ROOT/revalue_tvsum_offline_v1_2026-10-27"
```

The output directory must already contain `request.json` and
`before_receipts.json` from preflight. The auditor verifies their marker and
protected-file hashes, refuses to overwrite a completed audit, forbids model
imports/network/process launches inside the analysis process, and writes only
new audit/preparation artifacts. A successful CPU contract is not evidence of
how a model will respond to the new input.

See the [diagnosis](../../../../docs/revalue_tvsum_offline_diagnosis_v1_2026-10-27.md)
and [single-variable protocol](../../../../docs/revalue_tvsum_taskspec_protocol_v1_2026-10-27.md).
The proposed future 400-comparison budget is separate and pending authorization;
neither unused QV budget nor unused TVSum v1 token/GPU capacity authorizes it.
