"""Read frozen TVSum traces and prepare one TaskSpec intervention, without inference.

Only stdlib and CPU Pydantic contracts are used. No backend is imported; selected
pure definitions are extracted from the hash-verified frozen source by AST.
Prepared requests are hypothetical inputs, never predictions or authorizations.
"""

import argparse
import ast
import hashlib
import itertools
import json
import math
import re
import sys
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

MARKER = "MERIT_TVSUM_OFFLINE_20261027_1500_V1"
QUERY = "Highlight of this video."
BAD_CONSTRAINT = {"kind": "object", "quote": "Highlight of this video", "number": None}
ARMS = ("C_direct4", "E_memory")
QV_RUNS = (
    "revalue_observation_probe_v1_2026-10-26",
    "revalue_frame_grounding_v1_2026-10-26",
    "revalue_observer_output_v2_2026-10-26",
    "revalue_object_presence_v1_2026-10-26",
)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def write_new(path, value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def remove_exact_generic_object(task):
    """A narrow intervention for the ten frozen tasks, not a generic task parser."""
    if set(task) != {"query", "request_type", "purpose", "constraints"}:
        raise ValueError("Unexpected TaskSpec fields")
    if task["query"] != QUERY or task["request_type"] != "generic":
        raise ValueError("Only the frozen generic request is in scope")
    if task["constraints"] != [BAD_CONSTRAINT]:
        raise ValueError("Refuse to remove a different or additional constraint")
    changed = deepcopy(task)
    changed["constraints"] = []
    assert {k for k in task if task[k] != changed[k]} == {"constraints"}
    return changed


def patch_specification(specification, old_task):
    changed = deepcopy(specification)
    old, new = compact(old_task), compact(remove_exact_generic_object(old_task))
    for key in ("text", "rendered_prompt"):
        if changed[key].count(old) != 1:
            raise ValueError("TaskSpec must occur exactly once in each prompt representation")
        changed[key] = changed[key].replace(old, new, 1)
    assert {k for k in specification if specification[k] != changed[k]} == {"text", "rendered_prompt"}
    return changed


def definitions(path, names, namespace):
    """Load only named pure functions/classes/constants, with explicit dependencies."""
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    if {n.name for n in nodes} != set(names):
        raise ValueError("Frozen definition missing")
    env = {"__name__": "_tvsum_offline_contract", **namespace}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), "exec"), env)
    return env


def load_contracts(snapshot):
    from typing import Annotated, Literal

    from pydantic import BaseModel, ConfigDict, Field

    base = snapshot / "experiments/revalue"
    schema_names = {"Contract", "Constraint", "TaskSpec", "FactItem", "Facts", "MemoryItem", "MemoryDraft",
                    "Comparison", "validate_task"}
    env = definitions(base / "schema.py", schema_names, {
        "BaseModel": BaseModel, "ConfigDict": ConfigDict, "Field": Field, "Literal": Literal,
        "Annotated": Annotated, "VALUE": Literal["outcome", "information", "emotion", "action", "sensory"],
        "ELIGIBILITY": Literal["satisfied", "violated", "unknown"],
    })
    env.update(definitions(base / "evidence_grounding.py",
        {"interval", "temporal_scope", "validate_source", "direct_source"}, {**env, "math": math}))
    env.update(definitions(base / "comparison.py", {"allowed_references", "validate_comparison", "rerank"},
                          {**env, "itertools": itertools}))
    env.update(definitions(base / "backend.py", {"citation_schema"}, {**env, "deepcopy": deepcopy}))
    env.update(definitions(snapshot / "experiments/reze_evidence/protocol.py", {"digest"},
                          {"json": json, "hashlib": hashlib}))
    tree = ast.parse((base / "prompts.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {"TASK", "COMPARE", "MEMORY"}:
                    env[target.id] = ast.literal_eval(node.value)
    return env


def install_offline_guard():
    """Guard this analysis process; this is not a server-wide activity assertion."""
    forbidden = {"torch", "transformers", "vllm", "openai", "anthropic", "pynvml", "cuda"}

    def guard(event, args):
        if event in {"subprocess.Popen", "os.system", "socket.connect", "socket.getaddrinfo"}:
            raise RuntimeError("Offline audit prohibits process launch and network access")
        if event == "import" and args[0].split(".")[0] in forbidden:
            raise RuntimeError("Offline audit prohibits model/GPU/API modules")

    sys.dont_write_bytecode = True
    sys.addaudithook(guard)
    return forbidden


def protected_check(receipts):
    for name, receipt in receipts.items():
        if sha(name) != receipt["sha256"]:
            raise ValueError(f"Protected artifact changed: {name}")


def budget_books(root, tvsum):
    entries = []
    for name in QV_RUNS:
        ledger_path, resource_path = root / name / "probe/ledger.json", root / name / "resource_receipt.json"
        attempts, resource = read(ledger_path)["attempts"], read(resource_path)
        seconds = resource.get("full_service_seconds", resource.get("summed_gpu_seconds", resource.get("seconds")))
        entries.append({"run": name, "requests": len(attempts),
            "tokens": sum(x.get("charged_tokens", x.get("reserved_tokens", 0)) for x in attempts),
            "gpu_seconds": seconds, "ledger_sha256": sha(ledger_path), "resource_sha256": sha(resource_path)})
    qv = {"entries": entries, "requests_used": sum(x["requests"] for x in entries),
          "tokens_used": sum(x["tokens"] for x in entries), "gpu_hours_used": sum(x["gpu_seconds"] for x in entries) / 3600,
          "request_cap": 1000, "gpu_hour_cap": 2, "remaining_is_not_tvsum_authorization": True}
    assert (qv["requests_used"], qv["tokens_used"]) == (797, 398929)
    qv["requests_remaining"] = 1000 - qv["requests_used"]
    qv["gpu_hours_remaining"] = 2 - qv["gpu_hours_used"]
    return {"QV_diagnostics": qv, "TVSum_transfer_v1": tvsum,
        "current_offline_task": {"marker": MARKER, "experiment_model_calls": 0, "api_calls": 0,
            "retries": 0, "tokens": 0, "gpu_allocations": 0, "gpu_seconds": 0},
        "proposed_independent_TVSum_taskobject_v1": {"authorization": "pending; not executable",
            "formal_requests": 400, "failure_extra_requests": 0, "retries": 0, "smoke_extra_requests": 0,
            "total_attempt_cap": 400, "absolute_token_cap": 2457600, "gpu_lifecycle_seconds_cap": 3600,
            "actual_used_requests": 0, "actual_used_tokens": 0, "actual_used_gpu_seconds": 0}}


def audit(parent, output):
    forbidden = install_offline_guard()
    request, before = read(output / "request.json"), read(output / "before_receipts.json")
    assert request["marker"] == MARKER and request["authorized_new_experiment_model_calls"] == 0
    if (output / "audit.json").exists():
        raise FileExistsError("This request marker already has a completed audit")
    protected_check(before)
    manifest, completion, evaluation = read(parent / "manifest.json"), read(parent / "complete.json"), read(parent / "evaluation.json")
    assert completion["status"] == "complete" and completion["evaluation_sha256"] == sha(parent / "evaluation.json")
    assert evaluation["predictions_sha256"] == sha(parent / "predictions.json")
    for path, expected in manifest["source_sha256"].items():
        assert sha(parent / "source_snapshot" / path) == expected
    for path, expected in manifest["input_sha256"].items():
        assert sha(parent / path) == expected
    old_descriptive = read(parent / "diagnostics/descriptive_analysis.json")
    assert old_descriptive["evaluation_sha256"] == sha(parent / "evaluation.json")
    old_review = read(parent / "review/assistant_content_review.json")
    env = load_contracts(parent / "source_snapshot")
    digest = env["digest"]
    rows = read(parent / "inference.json")
    assert len(rows) == 10 and {r["vid"] for r in rows} == set(itertools.chain.from_iterable(manifest["shards"]))
    records, raw_by_key, attempts, health = {}, {}, [], []
    for slot in range(2):
        directory = parent / f"shard_{slot}"
        done, resource = read(directory / "inference_complete.json"), read(directory / "resource_receipt.json")
        ledger = read(directory / "ledger.json")["attempts"]
        assert len(ledger) == 335 and all(x["status"] == "complete" for x in ledger)
        assert done["qids"] == manifest["shards"][slot] and done["manifest_sha256"] == sha(parent / "manifest.json")
        assert resource["exit_code"] == 0 and not resource["remaining_owned_pids"]
        for vid in done["qids"]:
            path = directory / "records" / f"{vid}.json"
            assert sha(path) == done["records_sha256"][vid]
            records[vid] = read(path)
        for item in ledger:
            path = directory / "attempts" / f"{item['number']:05d}.json"
            raw = read(path)
            assert raw["key"] == item["key"] and digest(raw) == item["record_sha256"]
            assert digest(raw["specification"]) == item["request_sha256"] == raw["request_sha256"]
            raw_by_key[raw["key"]] = (raw, str(path.relative_to(parent)), sha(path))
        attempts.extend(ledger)
        health.append({"slot": slot, "stale_status": read(directory / "status.json"),
            "authoritative_completed_attempts": len(ledger), "resource_receipt": resource,
            "nanobind_exit_warning": "reference counting issue" in (directory / "worker.log").read_text(),
            "interpretation": "completed; stale generating status is not a restart reason"})
    assert len(raw_by_key) == len(attempts) == 670
    source_chain, transitions, memory_paths, comparisons, planned, contracts = [], Counter(), [], [], [], []
    reference_flags, duplicate_references = [], []
    for row in rows:
        vid, record = row["vid"], records[row["vid"]]
        task = record["task"]
        raw_task, task_path, task_sha = raw_by_key[vid + "/task"]
        assert task == env["validate_task"](raw_task["parsed"], row["query"])
        assert raw_task["specification"]["text"] == env["TASK"].format(query=row["query"])
        assert raw_task["specification"]["schema"] == env["TaskSpec"].model_json_schema()
        new_task = remove_exact_generic_object(task)
        assert env["validate_task"]({k: v for k, v in new_task.items() if k != "query"}, row["query"]) == new_task
        assert record["candidate_ids"] == row["candidate_ids"] and record["B0"] == row["scores"]
        source_chain.append({"vid": vid, "source_attempt": task_path, "source_attempt_sha256": task_sha,
            "old_task": task, "proposed_task": new_task,
            "changed_fields": ["constraints"], "base_validator_accepts_old_and_new": True,
            "query_quote_is_substring": BAD_CONSTRAINT["quote"] in row["query"],
            "no_specific_object_named": "semantic reading of the fixed generic query; not derived from labels"})
        memory_raw, memory_path, memory_sha = raw_by_key[vid + "/memory/M_decision"]
        facts, memory = record["facts"], record["memory"]
        assert memory_raw["specification"]["evidence"]["fact_store_sha256"] == digest(facts)
        assert memory_raw["specification"]["text"] == env["MEMORY"]["M_decision"].format(facts=compact(facts))
        assert memory["view_sha256"] == digest(memory["view"])
        draft = {f for item in memory_raw["parsed"]["items"] for f in item["fact_ids"]}
        retained = set(memory["retained_fact_ids"])
        assert retained == {f for item in memory["view"]["items"] for f in item["fact_ids"]}
        assert set(memory["capacity_evictions"]) == draft - retained
        def counts(ids):
            return {"all": len(ids), "unknown": sum(facts[f]["kind"] == "unknown" for f in ids)}
        memory_paths.append({"vid": vid, "source_attempt": memory_path, "source_attempt_sha256": memory_sha,
            "facts": counts(set(facts)), "draft": counts(draft), "retained": counts(retained),
            "omitted_by_draft": counts(set(facts) - draft), "capacity_evicted": counts(draft - retained),
            "draft_fact_ids": sorted(draft), "retained_fact_ids": sorted(retained),
            "omitted_by_draft_ids": sorted(set(facts) - draft), "capacity_evicted_ids": sorted(draft - retained)})
        maps = {c: [{"id": f"s{f['frame_index']}", "pts": f["frame_timestamps"]} for f in record["candidate_frames"][str(c)]["frames"]]
                for c in record["candidate_ids"]}
        frame_ids = {c: [f["id"] for f in values] for c, values in maps.items()}
        direct = {(c["a"], c["b"]): c["result"] for c in record["arms"]["C_direct4"]["comparisons"]}
        for arm in ARMS:
            view = {} if arm == "C_direct4" else memory["view"]
            for comparison in record["arms"][arm]["comparisons"]:
                a, b, result = comparison["a"], comparison["b"], comparison["result"]
                key = f"{vid}/{arm}/{a}/{b}"
                raw, raw_path, raw_sha = raw_by_key[key]
                assert raw["parsed"] == result == env["validate_comparison"](result, a, b, frame_ids, view)
                assert raw["specification"]["text"] == env["COMPARE"].format(task=compact(task),
                    a_frames=compact(maps[a]), b_frames=compact(maps[b]), memory=compact(view))
                assert raw["specification"]["evidence"] == [record["candidate_frames"][str(a)], record["candidate_frames"][str(b)]]
                refs = {"a_evidence": env["allowed_references"](a, frame_ids, view),
                        "b_evidence": env["allowed_references"](b, frame_ids, view)}
                assert raw["specification"]["schema"] == env["citation_schema"](env["Comparison"], refs)
                patched = patch_specification(raw["specification"], task)
                planned.append({"source_key": key, "source_attempt": raw_path, "source_attempt_sha256": raw_sha,
                    "old_specification_sha256": raw["request_sha256"], "proposed_specification_sha256": digest(patched),
                    "specification": patched, "state": "prepared_input_only_not_a_prediction", "authorized": False})
                comparisons.append({"key": key, "source_attempt": raw_path, "source_attempt_sha256": raw_sha,
                    "a": a, "b": b, "result": result, "structural_validation_passed": True})
                if arm == "E_memory":
                    d = direct[a, b]
                    transitions[f"{d['preference']} -> {result['preference']}"] += 1
                    reason_ids = sorted(set(re.findall(r"\bc\d+_[fu]\d+\b", result["reason"])))
                    outside = [i for i in reason_ids if i not in facts or facts[i]["candidate"] not in (a, b)]
                    if outside:
                        reference_flags.append({"key": key, "outside_pair_reason_fact_ids": outside,
                            "reason": result["reason"], "label": "review flag; outside context alone does not prove wrong attribution"})
                for side in ("a", "b"):
                    if len(result[side + "_evidence"]) != len(set(result[side + "_evidence"])):
                        duplicate_references.append({"key": key, "side": side, "refs": result[side + "_evidence"]})
            # Synthetic CPU contracts establish validator behavior, not model behavior.
            first = record["arms"][arm]["comparisons"][0]
            for preference in ("a_better", "b_better", "tie", "insufficient"):
                synthetic = deepcopy(first["result"])
                synthetic.update(a_eligibility="unknown", b_eligibility="unknown", preference=preference,
                    missing_question="no actionable question" if preference == "insufficient" else None)
                env["validate_comparison"](synthetic, first["a"], first["b"], frame_ids, view)
                contracts.append({"vid": vid, "arm": arm, "preference": preference,
                    "unknown_eligibility_accepted": True, "kind": "synthetic_CPU_contract_not_model_prediction"})
    assert len(planned) == len(comparisons) == 400
    # Preserve existing assistant visual review as supporting evidence, without rereading GT or relabeling images.
    propagation = []
    for fact_id in ("c53_f0", "c53_f4"):
        record = records["JgHubY5Vw3Y"]
        users = [c for c in comparisons if c["key"].startswith("JgHubY5Vw3Y/E_memory/") and
                 (fact_id in c["result"]["a_evidence"] + c["result"]["b_evidence"] or fact_id in c["result"]["reason"])]
        fact_raw, fact_path, fact_sha = raw_by_key["JgHubY5Vw3Y/facts/53"]
        propagation.append({"vid": "JgHubY5Vw3Y", "fact_id": fact_id, "fact": record["facts"][fact_id],
            "source_attempt": fact_path, "source_attempt_sha256": fact_sha,
            "raw_fact_matches": fact_raw["parsed"]["facts"][int(fact_id.rsplit('f', 1)[1])]["text"] == record["facts"][fact_id]["text"],
            "retained_items": [x for x in record["memory"]["view"]["items"] if fact_id in x["fact_ids"]],
            "comparison_uses": users, "visual_assessment_source": "existing shared-context assistant review; not independent human GT"})
    total_tokens = sum(a.get("charged_tokens", a["reserved_tokens"]) for a in attempts)
    gpu_seconds = sum(x["resource_receipt"]["seconds"] for x in health)
    assert total_tokens == 1106932 and abs(gpu_seconds - 2102.2639784840867) < 1e-6
    books = budget_books(parent.parent, {"requests_used": 670, "request_cap": 670, "requests_remaining": 0,
        "tokens_used": total_tokens, "token_cap": 4116480, "gpu_seconds_used": gpu_seconds, "gpu_seconds_cap": 7200,
        "unused_token_or_time_is_not_request_authorization": True})
    for vid, expected in old_review["source_record_sha256"].items():
        slot = next(i for i, ids in enumerate(manifest["shards"]) if vid in ids)
        assert sha(parent / f"shard_{slot}/records/{vid}.json") == expected
    protected_check(before)
    imported_forbidden = sorted(forbidden & {x.split(".")[0] for x in sys.modules})
    assert not imported_forbidden
    aggregates = {stage: {kind: sum(row[stage][kind] for row in memory_paths) for kind in ("all", "unknown")}
                  for stage in ("facts", "draft", "retained", "omitted_by_draft", "capacity_evicted")}
    report = {"marker": MARKER, "status": "offline_analysis_complete_no_experiment_started",
        "at": datetime.now(timezone.utc).isoformat(), "parent_manifest_sha256": sha(parent / "manifest.json"),
        "protected_files_unchanged": len(before), "parent_attempts_unchanged": len(attempts),
        "existing_analyses_reused": {n: sha(parent / n) for n in ("diagnostics/descriptive_analysis.json",
            "diagnostics/tie_sensitivity.json", "review/assistant_content_review.json")},
        "data_scope": [row["vid"] for row in rows], "labels_read": False, "metrics_recomputed": False,
        "task_source_chain": source_chain, "comparison_transition_counts": dict(transitions),
        "memory_stage_totals": aggregates, "memory_paths": memory_paths,
        "reason_outside_pair_flags": reference_flags, "duplicate_structured_references": duplicate_references,
        "comparisons": comparisons, "known_assistant_case_propagation": propagation, "synthetic_contract_checks": contracts,
        "prepared_requests": len(planned), "new_model_predictions": 0, "health": health,
        "guard": {"network_and_subprocess_calls_prohibited": True, "model_modules_not_imported": not imported_forbidden},
        "causal_claim": "Removing the erroneous constraint has not been tested by a model; no accuracy benefit established"}
    write_new(output / "audit.json", report)
    write_new(output / "budget_books.json", books)
    with (output / "prepared_requests.jsonl").open("x") as stream:
        for entry in planned:
            stream.write(compact(entry) + "\n")
    write_new(output / "preparation_receipt.json", {"marker": MARKER, "authorized": False,
        "state": "awaiting_separate_budget_authorization_and_runner_review", "requests": 400,
        "requests_sha256": sha(output / "prepared_requests.jsonl"), "audit_sha256": sha(output / "audit.json"),
        "code_sha256": sha(Path(__file__)), "budget_books_sha256": sha(output / "budget_books.json")})
    print(json.dumps({"marker": MARKER, "protected_files_unchanged": len(before), "validated_comparisons": len(comparisons),
        "synthetic_contract_checks": len(contracts), "transitions": dict(transitions), "memory_stages": aggregates,
        "reason_review_flags": len(reference_flags), "duplicate_reference_lists": len(duplicate_references),
        "prepared_requests": len(planned), "new_model_calls": 0}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.parent.resolve(), args.output.resolve())
