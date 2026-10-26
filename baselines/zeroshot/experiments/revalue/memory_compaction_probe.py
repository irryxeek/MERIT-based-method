"""CPU-only, frozen A/B/C memory capacity audit; never loads an inference engine."""

import argparse
import shutil
import sys
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path

from backend import MODEL as MODEL_PATH
from comparison import allowed_references
from evidence_grounding import source_anchor, validate_source
from frame_observer import has_visible_claim
from io_utils import compact, digest, read_json, sha256, utc_now, write_json
from memory import retrieve
from memory_codec import append_entry, decode, encode

OBSERVED = (5172, 881, 9841, 7971, 2973, 3478, 1129)
CONTROLS = (9429, 5370, 10272, 8963, 1058, 8025, 9196, 5073, 764, 7795, 5967, 6468, 1910, 7050, 10249, 3202)
ARMS = ("E_uniform", "E_targeted")
MODEL = Path(MODEL_PATH)
CAPACITY = 1024


def count(value, tokenizer):
    return len(tokenizer.encode(compact(value), add_special_tokens=False))


def old_candidates(items, retained, targets):
    """Sort only previously active whole entries, with stable original-index ties."""
    values = []
    for position, item in enumerate(items):
        overlap = retained.intersection(item["fact_ids"])
        if overlap:
            if overlap != set(item["fact_ids"]):
                raise ValueError("S0 contains a partial original entry")
            continue
        anchors = [source_anchor(source) for source in item["sources"].values()]
        values.append(((not any(a in targets for a in anchors), min(anchors), sorted(item["fact_ids"]), position), item))
    return [deepcopy(item) for _, item in sorted(values, key=lambda pair: pair[0])]


def source_stats(items):
    sources = {identity: source for item in items for identity, source in item["sources"].items()}
    return {
        "entries": len(items), "facts": len(sources),
        "non_unknown_facts": sum(s["kind"] != "unknown" for s in sources.values()),
        "direct_candidates": sorted({s["candidate"] for s in sources.values() if s["candidate"] is not None and s["kind"] != "unknown"}),
        "scope_counts": dict(Counter(s.get("grounding", {}).get("scope", "legacy_candidate") for s in sources.values())),
    }


def view_audit(audit, parent, tokenizer):
    memory = audit["scoped_memory"]
    a = retrieve(memory, tokenizer)
    old = retrieve(parent["memories"]["E_memory"], tokenizer)
    b = encode(a["items"])
    decoded_b = decode(b)
    roundtrip = compact(decoded_b) == compact(a["items"])
    # Object insertion order of source dictionaries is not a semantic change.
    if decoded_b != a["items"]:
        raise ValueError("Compact representation changed S0")
    for item in decoded_b:
        for source in item["sources"].values():
            validate_source(source)
    frames = {
        int(candidate): [f"s{frame['frame_index']}" for frame in trace["frames"]]
        for candidate, trace in parent["candidate_frames"].items()
    }
    refs_a = {c: allowed_references(c, frames, a) for c in frames}
    refs_b = {c: allowed_references(c, frames, {"items": decoded_b}) for c in frames}
    c = deepcopy(b)
    kept = {identity for item in decoded_b for identity in item["fact_ids"]}
    targets = {v["action"]["candidate"] for v in parent["observations"][audit["arm"]]}
    additions, skipped = [], []
    for entry in old_candidates(old["items"], kept, targets):
        proposed = append_entry(c, entry)
        if count(proposed, tokenizer) <= CAPACITY:
            c = proposed
            additions.append(entry)
            kept.update(entry["fact_ids"])
        else:
            skipped.append(entry["fact_ids"])
    decoded_c = decode(c)
    if decoded_c != a["items"] + additions:
        raise ValueError("Appending changed existing contents or order")
    ta, tb, tc = (count(view, tokenizer) for view in (a, b, c))
    old_ids = set(parent["memories"]["E_memory"]["retained_fact_ids"])
    a_ids = {i for item in a["items"] for i in item["fact_ids"]}
    added_ids = {i for entry in additions for i in entry["fact_ids"]}
    if not added_ids <= old_ids or added_ids & a_ids:
        raise ValueError("Restored evidence is not an evicted active old fact")
    # Tokenization is not additive: this is a fixed text-only counterfactual,
    # not a claim that text/metadata token fractions can be allocated uniquely.
    text_tokens = count([item["text"] for item in a["items"]], tokenizer)
    statistics = {
        "qid": audit["qid"], "arm": audit["arm"] ,
        "A_tokens": ta, "B_tokens": tb, "C_tokens": tc,
        "saved_A_minus_B": ta - tb, "text_only_reference_tokens": text_tokens,
        "A_non_text_overhead_reference": ta - text_tokens,
        "B_non_text_overhead_reference": tb - text_tokens,
        "roundtrip_equal": decoded_b == a["items"], "roundtrip_serialization_equal": roundtrip,
        "allowed_references_equal": refs_a == refs_b,
        "B_not_longer": tb <= ta, "B_C_within_capacity": tb <= CAPACITY and tc <= CAPACITY,
        "A": source_stats(a["items"]), "B": source_stats(decoded_b), "C": source_stats(decoded_c),
        "old_fact_count": len(old_ids), "A_old_fact_losses": len(old_ids - a_ids),
        "C_old_fact_losses": len(old_ids - (a_ids | added_ids)),
        "added_old_entries": len(additions), "added_old_fact_ids": sorted(added_ids),
        "added_non_unknown_entries": sum(any(s["kind"] != "unknown" for s in item["sources"].values()) for item in additions),
        "added_non_unknown_facts": sum(s["kind"] != "unknown" for item in additions for s in item["sources"].values()),
        "skipped_whole_entry_ids": skipped,
    }
    return {"A": a, "B": b, "C": c, "statistics": statistics, "allowed_references_A_B": refs_a}


def observation_audit(grounding):
    records = [read_json(path) for path in sorted((grounding / "frames").glob("*.json"))]
    raws = [read_json(record["response_path"]) for record in records if record["status"] == "complete"]
    if len(records) != 672 or len(raws) != 672:
        raise ValueError("Expected the complete frozen still-frame probe")
    ledger = read_json(grounding / "probe/ledger.json")
    indexed = {attempt["key"]: attempt for attempt in ledger["attempts"]}
    for record, raw in zip(records, raws, strict=True):
        if digest(raw) != indexed[record["key"]]["record_sha256"] or raw["parsed"] != record["result"]:
            raise ValueError("Observation raw/ledger mismatch")
    findings = read_json(grounding / "review/assistant_findings.json")
    return {
        "records": len(records), "technical_status": dict(Counter(r["status"] for r in records)),
        "visible_without_claim": sum(not has_visible_claim(r["result"]["visible"]) for r in records),
        "visible_at_160_character_cap": sum(len(r["result"]["visible"]) == 160 for r in records),
        "uncertainty_at_80_character_cap": sum(len(r["result"]["uncertainty"]) == 80 for r in records),
        "finish_reasons": dict(Counter(r["finish_reason"] for r in raws)),
        "historical_input_tokens": sum(r["prompt_tokens"] for r in raws),
        "historical_output_tokens": sum(r["output_tokens"] for r in raws),
        "historical_review": {"path": str(grounding / "review/assistant_findings.json"), "sha256": sha256(grounding / "review/assistant_findings.json"), "findings": findings},
        "interpretation": "Character-cap hits are a structural risk, not independently labeled semantic errors. Historical assistant findings are reused, not a new or independent human review.",
        "content_gate_passed": False,
        "reason": "Frozen legacy facts and still-frame replacement both have unresolved observed errors; no repaired automatic observer has passed the active-fact audit.",
        "new_model_calls": 0, "independent_human_review_count": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--grounding-run", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    run, grounding = args.run.resolve(), args.grounding_run.resolve()
    if run.exists():
        raise FileExistsError("Use a new run directory; completed/failed experiments are immutable")
    reference = read_json(grounding / "manifest.json")
    if reference["qids"] != list(OBSERVED):
        raise ValueError("Observed cohort differs from frozen plan")
    parent = Path(reference["parent_run"])
    for path, expected in reference["parent_receipts"].items():
        if sha256(path) != expected:
            raise ValueError(f"Parent artifact drift: {path}")
    parents = [grounding / name for name in ("manifest.json", "probe/ledger.json", "binding_audit.json", "review/assistant_findings.json", "resource_receipt.json")]
    parents += list((grounding / "binding_audit").glob("*.json"))
    parents += [parent / "R2/records" / f"{qid}.json" for qid in (*OBSERVED, *CONTROLS)]
    parents += [parent / "binding.json", parent / "R2/ledger.json"]
    parents += list((grounding / "frames").glob("*.json"))
    parents += [Path(read_json(p)["response_path"]) for p in (grounding / "frames").glob("*.json")]
    hashes = {str(path): sha256(path) for path in sorted(set(parents))}
    run.mkdir(parents=True)
    shutil.copy2(args.protocol, run / "protocol.md")
    source = Path(__file__).resolve().parent
    source_hashes = {}
    for path in sorted(source.rglob("*.py")):
        target = run / "source_snapshot" / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        source_hashes[str(path)] = sha256(path)
    tokenizer_hashes = {str(p): sha256(p) for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "config.json") if (p := MODEL / name).exists()}
    manifest = {
        "created_at": utc_now(), "mode": "cpu_only", "capacity": CAPACITY,
        "observed_qids": list(OBSERVED), "compatibility_qids": list(CONTROLS),
        "parent_run": str(parent), "grounding_run": str(grounding),
        "parent_sha256": hashes, "source_sha256": source_hashes,
        "protocol_sha256": sha256(run / "protocol.md"), "tokenizer_sha256": tokenizer_hashes,
        "model_revision": reference["model_revision"], "new_model_call_limit": 0,
        "command": [sys.executable, *sys.argv],
        "passport": {"Origin Skill": "experiment-agent", "Origin Mode": "run", "Origin Date": utc_now(), "Verification Status": "UNVERIFIED", "Version Label": "memory_compaction_cpu_v1"},
    }
    write_json(run / "manifest.json", manifest)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    results = []
    for qid in OBSERVED:
        record = read_json(parent / "R2/records" / f"{qid}.json")
        for arm in ARMS:
            audit = read_json(grounding / "binding_audit" / f"{qid}_{arm}.json")
            result = view_audit(audit, record, tokenizer)
            write_json(run / "views" / f"{qid}_{arm}.json", result)
            results.append(result["statistics"])
            print(f"q{qid} {arm}: A={result['statistics']['A_tokens']} B={result['statistics']['B_tokens']} C={result['statistics']['C_tokens']}, added={result['statistics']['added_non_unknown_entries']}", flush=True)
    compatibility = []
    for qid in CONTROLS:
        path = parent / "R2/records" / f"{qid}.json"
        record = read_json(path)
        if record["observations"] or any(record["arms"][arm]["scores"] != record["arms"]["E_memory"]["scores"] for arm in ARMS):
            raise ValueError("No-observation compatibility path changed")
        compatibility.append({"qid": qid, "path": str(path), "record_sha256": hashes[str(path)], "candidate_frames_sha256": digest(record["candidate_frames"]), "task_sha256": digest(record["task"]), "memory_sha256": digest(record["memories"]["E_memory"]), "predictions_sha256": {arm: digest(record["arms"][arm]["scores"]) for arm in ("E_memory", *ARMS)}, "reused_exact_record": True})
    write_json(run / "compatibility.json", compatibility)
    audit = observation_audit(grounding)
    write_json(run / "observation_audit.json", audit)
    gates = {
        "all_roundtrips_equal": all(v["roundtrip_equal"] for v in results),
        "all_reference_permissions_equal": all(v["allowed_references_equal"] for v in results),
        "all_B_not_longer": all(v["B_not_longer"] for v in results),
        "all_B_C_within_capacity": all(v["B_C_within_capacity"] for v in results),
        "at_least_seven_views_restore_non_unknown_entry": sum(v["added_non_unknown_entries"] > 0 for v in results) >= 7,
        "sixteen_compatibility_records_unchanged": len(compatibility) == 16,
    }
    for path, expected in hashes.items():
        if sha256(path) != expected:
            raise ValueError(f"Parent changed during audit: {path}")
    summary = {
        "at": utc_now(), "status": "complete", "gates": gates, "cpu_gate_passed": all(gates.values()),
        "views": results, "views_restoring_non_unknown_entry": sum(v["added_non_unknown_entries"] > 0 for v in results),
        "total_A_tokens": sum(v["A_tokens"] for v in results), "total_B_tokens": sum(v["B_tokens"] for v in results),
        "total_A_old_fact_losses": sum(v["A_old_fact_losses"] for v in results), "total_C_old_fact_losses": sum(v["C_old_fact_losses"] for v in results),
        "added_non_unknown_facts_across_views": sum(v["added_non_unknown_facts"] for v in results),
        "independent_source_videos": 7, "capacity_views": 14, "compatibility_controls": 16,
        "new_model_calls": 0, "gpu_service_seconds": 0, "new_hd_predictions": 0,
        "elapsed_seconds": time.monotonic() - started, "manifest_sha256": sha256(run / "manifest.json"),
        "content_gate_passed": audit["content_gate_passed"],
    }
    write_json(run / "summary.json", summary)
    decision = "stop_cpu_gate_failed" if not summary["cpu_gate_passed"] else "cpu_passed_observer_content_gate_pending"
    write_json(run / "decision.json", {"at": utc_now(), "decision": decision, "gpu_comparison_launched": False, "full_100_launched": False, "failed_cpu_gates": [k for k, v in gates.items() if not v], "content_gate_passed": False})
    write_json(run / "complete.json", {"status": "complete", "summary_sha256": sha256(run / "summary.json"), "decision_sha256": sha256(run / "decision.json"), "at": utc_now()})
    print(compact({"decision": decision, "gates": gates, "seconds": summary["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
