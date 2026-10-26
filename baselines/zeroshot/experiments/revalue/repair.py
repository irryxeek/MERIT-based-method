"""Verify immutable failed R0 provenance and carry its costs into a repair."""

from pathlib import Path

from io_utils import digest, read_json, sha256


def worker_remaining(binding, stage, attempts):
    history = binding.get("historical_usage", {}).get(stage, {})
    seconds = binding["maximum_active_worker_seconds_per_stage"] - history.get("active_seconds", 0)
    limit = binding.get("maximum_worker_attempts_by_stage", {}).get(stage, binding["maximum_worker_attempts_per_stage"])
    launches = limit - history.get("worker_launches", 0)
    return seconds - sum(a["seconds"] for a in attempts), launches - len(attempts)


def inspect_failed_r0(run, rows, binding, reuse_comparisons=False, content_repair=False):
    run = Path(run).resolve()
    old = read_json(run / "binding.json")
    gate = read_json(run / "R0/gate.json")
    if content_repair:
        if gate["status"] != "passed" or old["protocol"] != "revalue-v3-r01-v4":
            raise ValueError("Content diagnostic requires the completed v4 engineering gate")
    elif read_json(run / "stopped.json")["stage"] != "R0" or gate["status"] != "failed" or (run / "R1").exists():
        raise ValueError("Only a terminal failed engineering run with no formal results may be repaired")
    if gate["binding_sha256"] != sha256(run / "binding.json") or gate["formal_metrics_evaluated"]:
        raise ValueError("Engineering provenance mismatch or metric-driven repair")
    for name, expected in old["source_sha256"].items():
        if sha256(run / "source_snapshot" / name) != expected:
            raise ValueError("Historical source snapshot drift")
    for name, expected in old["input_sha256"].items():
        if sha256(run / name) != expected:
            raise ValueError("Historical frozen input drift")
    for name, expected in old["parent_receipts"].items():
        if sha256(name) != expected:
            raise ValueError("Historical parent receipt drift")
    for key in ("baseline_run", "pair_run", "model_revision", "baseline_tokens"):
        if old[key] != binding[key]:
            raise ValueError(f"Repair changed {key}")
    from io_utils import legacy

    if legacy.read_jsonl(run / "inference.jsonl") != rows:
        raise ValueError("Repair changed fixed engineering queries")
    receipt_path = run / "R0/record_receipt.json"
    if sha256(receipt_path) != gate["record_receipt_sha256"]:
        raise ValueError("Historical query receipt changed")
    for qid, expected in read_json(receipt_path).items():
        if sha256(run / "R0/records" / f"{qid}.json") != expected:
            raise ValueError("Historical query record changed")
    ledger = read_json(run / "R0/ledger.json")
    attempts = ledger["attempts"]
    if len(attempts) != gate["actual_attempts"]:
        raise ValueError("Historical attempt count changed")
    charged = sum(a.get("charged_tokens", a["reserved_tokens"]) for a in attempts)
    if charged != gate["actual_charged_tokens"]:
        raise ValueError("Historical charged tokens changed")
    index, receipts = {}, {}
    for attempt in attempts:
        path = run / "R0/attempts" / f"{attempt['number']:05d}.json"
        record = read_json(path)
        if digest(record) != attempt["record_sha256"] or record["key"] != attempt["key"]:
            raise ValueError("Historical response changed")
        receipts[str(path)] = sha256(path)
        if record["status"] != "complete":
            continue
        parts = record["key"].split("/")
        permitted = {"task", "facts", "memory"}
        if reuse_comparisons:
            permitted.update({"C_direct4", "M_plain", "M_decision"})
        if parts[1] not in permitted:
            continue
        if record["request_sha256"] != digest(record["specification"]):
            raise ValueError("Historical request changed")
        if record["key"] in index:
            raise ValueError("Common request has multiple successful responses")
        index[record["key"]] = {
            "path": str(path),
            "file_sha256": receipts[str(path)],
            "record_sha256": digest(record),
            "request_sha256": record["request_sha256"],
        }
    # An early-startup failure may not yet have reached all frozen dependencies.
    inherited_path = run / "reuse_index.json"
    inherited = read_json(inherited_path) if inherited_path.exists() else {}
    for key, entry in ledger.get("reused", {}).items():
        if inherited.get(key) != entry:
            raise ValueError("Reused dependency not in the frozen index")
    for key, entry in inherited.items():
        if key in index:
            raise ValueError("Duplicate reused/generated common request")
        record = read_json(entry["path"])
        if (
            record["status"] != "complete"
            or record["key"] != key
            or sha256(entry["path"]) != entry["file_sha256"]
            or digest(record) != entry["record_sha256"]
        ):
            raise ValueError("Previously reused dependency drift")
        index[key] = entry
        receipts[entry["path"]] = entry["file_sha256"]
    expected, allowed_comparisons = set(), set()
    from io_utils import parents

    for row in rows:
        _, pair = parents(binding, row)
        qid = str(row["qid"])
        expected.update([f"{qid}/task", f"{qid}/memory/M_plain", f"{qid}/memory/M_decision"])
        expected.update(f"{qid}/facts/{c}" for c in pair["candidate_ids"])
        a, b = pair["candidate_ids"][:2]
        for arm in ("C_direct4", "M_plain", "M_decision"):
            allowed_comparisons.update({f"{qid}/{arm}/{a}/{b}", f"{qid}/{arm}/{b}/{a}"})
    extra = set(index) - expected
    if not expected <= set(index) or len(expected) != 160:
        raise ValueError("All 160 unchanged shared requests must be valid and auditable")
    if extra and (not reuse_comparisons or not extra <= allowed_comparisons):
        raise ValueError("Repair reuse includes a changed or out-of-scope request")
    workers = read_json(run / "R0/worker_attempts.json")
    if any(a["status"] in {"running", "interrupted"} for a in workers):
        raise ValueError("Repair requires settled worker accounting")
    prior = old.get("historical_usage", {}).get("R0", {})
    history = {
        "attempts": prior.get("attempts", 0) + len(attempts),
        "charged_tokens": prior.get("charged_tokens", 0) + charged,
        "active_seconds": prior.get("active_seconds", 0) + sum(a["seconds"] for a in workers),
        "worker_launches": prior.get("worker_launches", 0) + len(workers),
    }
    new_requests = len(rows) * 7 - len(extra)
    remaining_attempts = 500 - history["attempts"]
    if remaining_attempts < new_requests:
        raise ValueError("Insufficient cumulative R0 request budget for a complete repair")
    for name in (
        "binding.json",
        "R0/gate.json",
        "R0/ledger.json",
        "R0/worker_attempts.json",
        "R0/record_receipt.json",
        "candidate_inputs.jsonl",
    ):
        receipts[str(run / name)] = sha256(run / name)
    if not content_repair:
        receipts[str(run / "stopped.json")] = sha256(run / "stopped.json")
    return (
        index,
        history,
        receipts,
        {
            "requests": new_requests,
            "extra_attempts": remaining_attempts - new_requests,
            "arm_incremental_tokens": {},
        },
    )


def monitor_startup_recovery(run):
    """One declared fourth R0 launch for v2 diagnostics failing before generation."""
    run = Path(run)
    binding = read_json(run / "binding.json")
    attempts = read_json(run / "R0/ledger.json")["attempts"]
    if binding["protocol"] != "revalue-v3-r01-v2" or len(attempts) != 1:
        raise ValueError("Startup exception only covers the frozen v2 monitoring failure")
    raw = read_json(run / "R0/attempts" / f"{attempts[0]['number']:05d}.json")
    if (
        raw["status"] != "runtime_error"
        or "is not serializable" not in raw.get("error", "")
        or "function" not in raw["error"]
        or "output_tokens" in raw
    ):
        raise ValueError("Startup exception cannot cover model failures or semantic regeneration")
    return {
        "original_R0_launch_ceiling": 3,
        "revised_R0_launch_ceiling": 4,
        "reason": "One diagnostic RPC failed before model generation; preserve all costs and permit one corrected startup",
        "request_and_active_time_caps_unchanged": True,
        "failed_response_sha256": sha256(run / "R0/attempts" / f"{attempts[0]['number']:05d}.json"),
    }


def observation_contract_recovery(run):
    """Declare one repair iteration for the empty-observation validation loophole."""
    run = Path(run)
    if read_json(run / "binding.json")["protocol"] != "revalue-v3-r01-v3":
        raise ValueError("Observation repair requires the frozen v3 diagnostic")
    failures = []
    for attempt in read_json(run / "R0/ledger.json")["attempts"]:
        if attempt["key"].endswith("/observation") and attempt["status"] == "complete":
            path = run / "R0/attempts" / f"{attempt['number']:05d}.json"
            raw = read_json(path)
            if raw["parsed"] == {"facts": [], "unknown": []}:
                failures.append({"path": str(path), "sha256": sha256(path)})
    if not failures:
        raise ValueError("No empty-observation validation failure was established")
    return {
        "previous_R0_launch_ceiling": 4,
        "revised_R0_launch_ceiling": 5,
        "reason": "Require a nonempty source-cited observation; empty outputs must not pass R0",
        "request_and_active_time_caps_unchanged": True,
        "empty_observation_receipts": failures,
    }


def observation_content_recovery(run):
    """A single isolated observation-only diagnostic; the running R1 is frozen."""
    run = Path(run)
    if read_json(run / "binding.json")["protocol"] != "revalue-v3-r01-v4":
        raise ValueError("Content diagnostic is restricted to the audited v4 regression")
    observations = []
    for attempt in read_json(run / "R0/ledger.json")["attempts"]:
        if attempt["key"].endswith("/observation") and attempt["status"] == "complete":
            raw = read_json(run / "R0/attempts" / f"{attempt['number']:05d}.json")
            observations.append(raw["parsed"])
    if len(observations) != 20 or not all(
        value["facts"] and all(f["kind"] == "unknown" for f in value["facts"]) for value in observations
    ):
        raise ValueError("The diagnostic requires the recorded all-unknown regression on all twenty queries")
    return {
        "previous_R0_launch_ceiling": 5,
        "revised_R0_launch_ceiling": 6,
        "reason": "Restore visible observations instead of replacing every fact with an unresolved detail",
        "R1_parent_unchanged": str(run),
        "request_and_active_time_caps_unchanged": True,
        "scope": "20 new observations only; no new R1 and no HD evaluation",
    }
