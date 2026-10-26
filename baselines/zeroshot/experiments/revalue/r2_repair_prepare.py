"""Freeze a new complete R2 repair run; reuse only exact query-independent dependencies and smoke tasks."""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from backend import MODEL, Backend
from io_utils import HERE, ROOT, digest, legacy, parents, read_json, sha256, utc_now, write_json
from r2_protocol import ARMS, R1_ARMS, SharedLedger, ordered_rows
from r2_worker import memory_dependencies
from task_grounding import extract_task


def receipt(path):
    raw = read_json(path)
    if raw["status"] != "complete" or digest(raw["specification"]) != raw["request_sha256"]:
        raise ValueError("Reuse needs an immutable successful exact request")
    return {
        "path": str(path),
        "file_sha256": sha256(path),
        "record_sha256": digest(raw),
        "charged_tokens": raw["prompt_tokens"] + raw["output_tokens"],
        "original_arms": raw["specification"]["arms"],
    }


def cache_backend(directory, reuse):
    from transformers import AutoTokenizer

    class Audit:
        verified = set()

        def record_reuse(self, key, entry):
            self.verified.add(key)

    class CacheOnly(Backend):
        def call(self, key, *args, **kwargs):
            if key not in self.reuse_index:
                raise ValueError("No new model calls permitted in CPU preflight")
            return super().call(key, *args, **kwargs)

    runner = CacheOnly.__new__(CacheOnly)
    runner.directory, runner.ledger, runner.reuse_index = directory, Audit(), reuse
    runner.tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    return runner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--previous-run", type=Path, required=True)
    parser.add_argument("--smoke-run", type=Path, required=True)
    parser.add_argument("--execution-plan", type=Path, required=True)
    parser.add_argument("--implementation-plan", type=Path, required=True)
    args = parser.parse_args()
    run, previous, smoke = args.run.resolve(), args.previous_run.resolve(), args.smoke_run.resolve()
    if run.exists():
        raise ValueError("Use a new immutable repair run")
    old = read_json(previous / "binding.json")
    done, gate = read_json(previous / "complete.json"), read_json(smoke / "gate.json")
    if done["status"] != "complete" or sha256(previous / "R2/results/evaluation.json") != done["evaluation_sha256"]:
        raise ValueError("Previous completed R2 receipt drift")
    if gate["status"] != "passed" or sha256(smoke / "smoke/ledger.json") != gate["ledger_sha256"]:
        raise ValueError("Real repaired observation path must pass before formal rerun")
    for name, expected in old["source_sha256"].items():
        if sha256(previous / "source_snapshot" / name) != expected:
            raise ValueError("Historical source changed")
    for name, expected in old["input_sha256"].items():
        if sha256(previous / name) != expected:
            raise ValueError("Historical frozen input changed")
    rows = ordered_rows(legacy.read_jsonl(previous / "inference.jsonl"))
    frozen = {r["qid"]: r for r in legacy.read_jsonl(previous / "candidate_inputs.jsonl")}
    binding = {
        "protocol": "revalue-v3-r2-v3",
        "created_at": utc_now(),
        "baseline_run": old["baseline_run"],
        "pair_run": old["pair_run"],
        "r1_run": old["r1_run"],
        "observer_run": old["observer_run"],
        "repair_parent": str(previous),
        "repair_smoke": str(smoke),
        "source_sha256": {},
        "input_sha256": {},
        "parent_receipts": {},
        "gpu_candidates": "all",
        "maximum_active_worker_seconds": 6 * 3600,
        "maximum_concurrent_workers": 8,
        "maximum_worker_launches": 8,
        "maximum_worker_stall_seconds": 900,
        "queue_expiry_hours": 72,
        "model_revision": old["model_revision"],
        "stage_order": ["R2"],
        "historical_R2_usage": {
            "previous_gpu_seconds": read_json(previous / "R2/results/evaluation.json")["summed_gpu_seconds"],
            "previous_new_attempts": read_json(previous / "R2/results/evaluation.json")["actual_new_attempts"],
            "repair_smoke_gate": gate,
            "budget_scope": "new authorized rerun; prior ledger retained separately",
        },
    }
    reuse = {}
    state = read_json(previous / "R2/ledger.json")
    raw_entries = list(state.get("reused", {}).values())
    raw_entries += [
        {**a, "path": str(previous / "R2/attempts" / f"{a['number']:05d}.json")}
        for a in state["attempts"]
        if a["status"] == "complete"
    ]
    for entry in raw_entries:
        path = Path(entry["path"])
        raw = read_json(path)
        if digest(raw) != entry["record_sha256"]:
            raise ValueError("Old raw request hash changed")
        if "/facts/" in raw["key"] or raw["key"].endswith("/memory/M_decision"):
            reuse[raw["key"]] = receipt(path)
    if len(reuse) != 93 * 6:
        raise ValueError("Expected six independent dependencies for the prior 93 successful queries")
    smoke_state = read_json(smoke / "smoke/ledger.json")
    for entry in smoke_state["attempts"]:
        if entry["key"].endswith("/task"):
            path = smoke / "smoke/attempts" / f"{entry['number']:05d}.json"
            if digest(read_json(path)) != entry["record_sha256"]:
                raise ValueError("Smoke task receipt changed")
            reuse[entry["key"]] = receipt(path)
    if len(reuse) != 566:
        raise ValueError("Expected exactly eight repaired smoke task requests")
    for entry in reuse.values():
        binding["parent_receipts"][entry["path"]] = entry["file_sha256"]
    for parent, names in [
        (
            previous,
            [
                "binding.json",
                "complete.json",
                "R2/ledger.json",
                "R2/record_receipt.json",
                "R2/inference_complete.json",
                "R2/results/evaluation.json",
                "candidate_inputs.jsonl",
                "inference.jsonl",
            ],
        ),
        (smoke, ["manifest.json", "gate.json", "smoke/ledger.json", "smoke_record.json", "launch.json"]),
        (Path(old["baseline_run"]), ["binding.json", "completion.json", "record_receipt.json", "inference.jsonl"]),
        (Path(old["pair_run"]), ["binding.json", "completion.json", "record_receipt.json", "inference.jsonl"]),
    ]:
        for name in names:
            binding["parent_receipts"][str(parent / name)] = sha256(parent / name)
    run.mkdir()
    legacy.write_jsonl(run / "inference.jsonl", rows)
    write_json(run / "reuse_index.json", reuse)
    test = subprocess.run([sys.executable, "-m", "pytest", str(HERE / "tests"), "-q"], capture_output=True, text=True)
    (run / "cpu_tests.log").write_text(test.stdout + test.stderr)
    write_json(run / "cpu_tests.json", {"returncode": test.returncode, "log_sha256": sha256(run / "cpu_tests.log")})
    if test.returncode:
        raise ValueError("Repair CPU regression failed; preserve preparation")
    import torch
    from candidate_frames import prepare_candidates

    torch.set_num_threads(2)
    cache = cache_backend(run / "cache_preflight", reuse)
    tokens, fallback_tokens, audit = 0, 0, []
    for i, row in enumerate(rows):
        baseline, pair = parents(binding, row)
        item = frozen[row["qid"]]
        if sha256(row["video_path"]) != item["source_sha256"] or item["candidate_frames"] != pair["candidate_frames"]:
            raise ValueError("Inherited full-PTS source media changed")
        prefix = str(row["qid"])
        if prefix + "/task" in reuse:
            extract_task(row, cache, R1_ARMS)
        if prefix + "/memory/M_decision" in reuse:
            targets, traces, _ = prepare_candidates(row, baseline, pair["candidate_ids"])
            if traces != item["candidate_frames"]:
                raise ValueError("Candidate pixels changed")
            memory_dependencies(row, cache, targets, traces, pair["candidate_ids"])
            del targets
        tokens += sum(baseline["cost"][k] for k in ("input_tokens", "output_tokens"))
        fallback_tokens += sum(pair["arms"]["P_pair"]["cost"][k] for k in ("input_tokens", "output_tokens"))
        audit.append(
            {"qid": row["qid"], "candidate_ids": pair["candidate_ids"], "source_sha256": item["source_sha256"]}
        )
        print(f"Repaired preflight {i + 1}/100; exact requests {len(cache.ledger.verified)}/{len(reuse)}", flush=True)
    if cache.ledger.verified != set(reuse):
        raise ValueError("Every cached request must replay under the current exact specification")
    legacy.write_jsonl(run / "candidate_inputs.jsonl", [frozen[r["qid"]] for r in rows])
    write_json(run / "parent_audit.json", {"records": audit, "all_100_sources_and_parents_verified": True})
    write_json(
        run / "cache_preflight.json",
        {
            "verified_requests": len(reuse),
            "new_model_calls": 0,
            "independent_fact_memory_requests": 558,
            "new_protocol_smoke_tasks": 8,
        },
    )
    binding["baseline_tokens"] = tokens
    binding["budgets"] = {
        "requests": 9200 - len(reuse),
        "extra_attempts": 460,
        "arm_incremental_tokens": {a: 5 * tokens for a in ARMS},
        "fallback_reserve_tokens": fallback_tokens,
    }
    model = read_json(previous / "model_verification.json")
    for name, expected in model["stats"].items():
        stat = (Path(MODEL) / name).stat()
        if stat.st_size != expected["size"] or stat.st_mtime_ns != expected["mtime_ns"]:
            raise ValueError("Previously verified model changed")
    for name in ["model_verification.json", "evaluation_binding.json"]:
        shutil.copy2(previous / name, run / name)
    shutil.copy2(args.execution_plan, run / "execution_plan.md")
    shutil.copy2(args.implementation_plan, run / "implementation_plan.md")
    write_json(
        run / "plan.json",
        {
            "queries": 100,
            "selection": "unchanged original explore100 and SHA256 order",
            "arms": ["E0", *ARMS],
            "maximum_calls_before_reuse": 9200,
            "verified_reusable_requests": len(reuse),
            "task": "query-grounded constrained decoding; all 100 use new protocol, 8 verified in smoke",
            "comparison_and_memory_prompts": "unchanged; dependent comparisons regenerated for all 100",
            "planner": "original one-call LLM plus deterministic attribution probe on an empty directional-conflict plan",
            "observation_pairs": "max two; same questions and counts; up to 24 new source frames each",
            "memory_update": "unchanged deterministic 1024-token bound",
            "zero_observation_cannot_pass_gate": True,
            "no_labels_in_inference": True,
            "R3_R4_auto_start": False,
        },
    )
    names = set(legacy.source_hashes())
    names.update(
        str(p.relative_to(ROOT))
        for p in HERE.rglob("*")
        if p.is_file() and p.suffix in {".py", ".md"} and "__pycache__" not in p.parts
    )
    for name in sorted(names):
        target = run / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
        binding["source_sha256"][name] = sha256(target)
    for path in run.iterdir():
        if path.is_file():
            binding["input_sha256"][path.name] = sha256(path)
    write_json(run / "binding.json", binding)
    SharedLedger(run / "R2", binding["budgets"])
    write_json(run / "status.json", {"status": "prepared_waiting_for_idle_gpu", "stage": "R2", "updated_at": utc_now()})
    print(f"Prepared repaired R2: {run}; binding={sha256(run / 'binding.json')}", flush=True)


if __name__ == "__main__":
    main()
