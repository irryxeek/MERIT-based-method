# Frozen Stage-A full validation

> 发布版入口：[统一复现索引](../../../../docs/reproduction_index.md)。下文保留原实验协议和当时的执行记录；路径与 GPU 以本机配置为准，历史父运行缓存需另行提供。

This extension runs both existing scoring arms on all 1,550 QVHighlights validation queries. It reuses `reze_val_v2` and does not modify the frozen method under `experiments/reze_evidence/`. The earlier exploration gate failed; this is a separately requested expansion on a previously examined validation split.

Prepare the full baseline/model audit and immutable snapshot from the project root:

```bash
${REPRO_PYTHON} experiments/reze_fullval/runner.py --prepare ${MERIT_ROOT}/configs/reze_evidence_fullval_v1.json
```

The user systemd unit `reze-evidence-fullval-20261020.service` executes the prepared snapshot. It runs both arms, checks full coverage, evaluates with the official HD evaluator, computes paired source-video bootstrap intervals, and archives results. Workers receive only the five inference fields; annotations are read by the CPU audit/evaluation stage.

```bash
systemctl --user status reze-evidence-fullval-20261020.service
journalctl --user -u reze-evidence-fullval-20261020.service -n 30 --no-pager
${REPRO_PYTHON} -m pytest -q tests/test_reze_fullval.py tests/test_reze_evidence.py tests/test_reze_checkpoints.py
```

Raw outputs and progress: `${ZEROSHOT_ARTIFACT_ROOT}/reze_evidence_fullval_v1/`. Final small archive: `results/reze_evidence_fullval_v1/`. A `Type=oneshot` unit remains `activating (start)` while inference runs. GPU UUID, occupancy, environment, source/data hashes, attempts, runtime budgets, and per-query checkpoints are checked. A failed infrastructure run is not a completed evaluation; semantic failures explicitly retain B0 predictions in the full denominator.
