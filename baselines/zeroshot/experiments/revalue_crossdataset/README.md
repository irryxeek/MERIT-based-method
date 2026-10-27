# ReVALUE TVSum val10 transfer

Fixed development transfer of B0, original P_pair, C_direct4 and E_memory.
Reuses the ten UMT validation videos, original REZE Top-5 candidates and old
frames; no titles, audio, new observations, training or model changes.
The memory comparator also sees old visual evidence, so this is not a text-only
memory experiment. The independent contribution of memory is E_memory versus
C_direct4; P_pair retains its separate Yes/No protocol.

See the [frozen protocol](../../../../docs/revalue_tvsum_transfer_protocol_2026-10-27.md).
Configure `REPRO_PYTHON`, `REZE_MODEL_PATH`, `MERIT_ROOT` and
`ZEROSHOT_ARTIFACT_ROOT` using the repository's path configuration. The required
historical B0 cache and model receipt are not bundled.

Run from `baselines/zeroshot`:

```bash
"$REPRO_PYTHON" -m pytest experiments/revalue_crossdataset/tests -q
"$REPRO_PYTHON" experiments/revalue_crossdataset/run.py prepare \
  --run "$ZEROSHOT_ARTIFACT_ROOT/revalue_tvsum_val10_v1_2026-10-27" \
  --baseline "$ZEROSHOT_ARTIFACT_ROOT/reze_tvsum_full50_v1" \
  --protocol "$MERIT_ROOT/docs/revalue_tvsum_transfer_protocol_2026-10-27.md" \
  --model-receipt "$ZEROSHOT_ARTIFACT_ROOT/revalue_r2_v3_2026-10-26/model_verification.json"
"$REPRO_PYTHON" "$ZEROSHOT_ARTIFACT_ROOT/revalue_tvsum_val10_v1_2026-10-27/source_snapshot/experiments/revalue_crossdataset/run.py" run \
  --run "$ZEROSHOT_ARTIFACT_ROOT/revalue_tvsum_val10_v1_2026-10-27"
```

The supervisor checks GPU UUID/occupancy, runs two disjoint five-video shards,
monitors every 30 seconds, cleans up owned process groups and evaluates only
after complete predictions are frozen. Limits: 670 attempts, zero retries,
4,116,480 reserved tokens, two cumulative GPU hours. Failed attempts and partial
outputs remain immutable. New versions require a new directory.

`status.json`, `shard_*/ledger.json`, `shard_*/worker.log`,
`shard_*/resource_receipt.json`, `predictions.json` and `evaluation.json`
record progress, costs, raw evidence and native UMT Top-5 mAP. The unified engine
uses the existing ReVALUE deployment configuration; historical P_pair batching
throughput is not directly comparable. Independent human content review remains
separate from structural validation and metric computation.
