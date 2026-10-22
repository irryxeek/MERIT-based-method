# REZE Evidence v1 execution

> 发布版入口：[统一复现索引](../../../../docs/reproduction_index.md)。下文保留原实验协议和当时的执行记录；路径与 GPU 以本机配置为准，历史父运行缓存需另行提供。

This isolated implementation executes Stage A of the frozen plan at
`${MERIT_ROOT}/docs/reze_execution_plan_2026-10-19.md`.
The original REZE source and validation results remain frozen.

Completed 2026-10-19: B0 / P_abs / P_pair VeryGood mAP = 44.41 / 44.71 / 44.58,
Hit@1 = 71 / 76 / 78 on all 100 exploration queries, with no fallback.
Neither new arm met the prespecified +1 pp mAP gate; v1 ended at Stage A.
See [results](../../../../docs/reze_stage_a_results_2026-10-19.md).

## Environment and commands

Use Python 3.10.12 from `${REPRO_PYTHON}`.
The existing root `requirements.txt` describes the shared environment; every run
also binds the installed package versions. No environment upgrades are needed.

```bash
cd ${ZEROSHOT_ROOT}
PY=${REPRO_PYTHON}
$PY -m pytest tests/test_reze_evidence.py tests/test_reze_native_processor.py tests/test_reze_core.py tests/test_reze_checkpoints.py -q
$PY -m ruff check experiments/reze_evidence tests/test_reze_evidence.py
$PY experiments/reze_evidence/launch.py --gpus 5,7
```

The launcher validates model/data audits, freezes source snapshots and manifests,
runs B0 and both new scoring arms on the 21 engineering queries, checks memory
stability, then runs all 100 exploration queries and the official evaluator.
It never evaluates incomplete or engineering predictions for accuracy.
Each worker verifies its physical GPU UUID and only uses an idle assigned GPU.

The engineering run exposed two input bugs, fixed before any accuracy evaluation:
target-label punctuation merged with native timestamp tokens, and irregular tail
observations were omitted by exact-grid matching. Native processor tests also
protect shared metadata against in-place temporal padding. The failed versions
and decision records are retained under `audit/` and `runs/engineering*`.
For the final check, 20 unaffected full-comparison records are verified and reused;
qid 688 is rerun with all 40 requests. A fresh 21-query continuous stress pass uses
10 requests per query (five-candidate cycle in both arms), covering every candidate
in both video positions. Including failed reservations, this uses 2,139 requests,
within the 2,500 engineering cap. Formal explore100 still runs all 40 comparisons.

## Artifacts and recovery

The artifact root is `${ZEROSHOT_ARTIFACT_ROOT}/reze_evidence_v1`.
`campaign.json` reports the current phase. `runs/*/worker_*.json` contain progress;
`logs/`, `records/`, `attempts_*.jsonl`, `source_snapshot/` retain detailed evidence.
Completed checkpoints must replay from raw logits and match frozen bindings.
Each worker has at most two restarts; reserved failed requests remain in the budget.
Memory, attempted requests, GPU allocation time and cold/incremental tokens are recorded.

The installed user service is `reze-evidence-v1-20261019.service`:

```bash
systemctl --user status reze-evidence-v1-20261019.service
journalctl --user -u reze-evidence-v1-20261019.service --no-pager -n 40
```

During execution the service was enabled for the user default target, with user linger enabled.
After the campaign completed, the service was stopped and disabled to avoid redundant startup.
Recovery after a host reboot still requires the SSD mounts, model/data paths and
assigned GPUs to be available. Only this service's own process groups are stopped.

Stage A follows the preregistered accuracy, fallback and token gates. A rejection
ends this v1 campaign. If it qualifies, Stage B requires its conditional modules
to be implemented and verified under the same authorized plan; this launcher
reports `stage_B_required` and does not pretend those modules have run.

Validation: 44 focused tests passed, including the actual vLLM multimodal processor.
`scripts/audit_reze_evidence.py` verifies complete run receipts, all raw score replay,
2,000 matched evidence pairs, target-frame provenance, timestamp accuracy and cost totals.

Small results are copied to `results/reze_evidence_v1/`; the Markdown result report
is also copied to the MERIT `docs/` directory. No Git commit or push is performed.
