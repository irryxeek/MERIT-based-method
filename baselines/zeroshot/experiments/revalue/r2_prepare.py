"""Freeze the original explore100, exact R1 reuse and the one-shot R2 protocol."""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from io_utils import HERE, ROOT, digest, legacy, parents, read_json, sha256, utc_now, write_json
from r2_protocol import ARMS, SharedLedger, ordered_rows


class ReuseAudit:
    def __init__(self):
        self.verified = set()

    def record_reuse(self, key, receipt):
        self.verified.add(key)


def cache_backend(directory, reuse):
    """No model engine; a new inference request is an error in this CPU audit."""
    from backend import MODEL, Backend
    from r2_planner import LocalPlan
    from transformers import AutoTokenizer

    class CacheOnly(Backend):
        def call(self, key, text, videos, evidence, schema, cap, arms, validate, references=None):
            if schema is LocalPlan:
                # Explicit fixture: skips unimplemented inference in cache-only audit.
                return validate({"actions": []}), {"response_path": None}
            if key not in self.reuse_index:
                raise ValueError("CPU audit attempted a new model request")
            return super().call(key, text, videos, evidence, schema, cap, arms, validate, references)

    runner = CacheOnly.__new__(CacheOnly)
    runner.directory, runner.ledger, runner.reuse_index = directory, ReuseAudit(), reuse
    runner.tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    return runner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--r1-run", type=Path, required=True)
    parser.add_argument("--observer-run", type=Path, required=True)
    parser.add_argument("--execution-plan", type=Path, required=True)
    parser.add_argument("--implementation-plan", type=Path, required=True)
    parser.add_argument("--media-preflight-from", type=Path)
    args = parser.parse_args()
    run, r1 = args.run.resolve(), args.r1_run.resolve()
    if run.exists():
        raise ValueError("Use a new immutable R2 run directory")
    old = read_json(r1 / "binding.json")
    if read_json(r1 / "complete.json")["status"] != "complete":
        raise ValueError("R1 must have completed before R2")
    if sha256(r1 / "R1/results/evaluation.json") != read_json(r1 / "complete.json")["evaluation_sha256"]:
        raise ValueError("R1 completion receipt drift")
    if read_json(args.observer_run / "R0/gate.json")["status"] != "passed":
        raise ValueError("Repaired real observation gate must pass")
    for name, expected in old["source_sha256"].items():
        if sha256(r1 / "source_snapshot" / name) != expected:
            raise ValueError("Frozen R1 source drift")
    pair_run, baseline_run = Path(old["pair_run"]), Path(old["baseline_run"])
    parent_binding, rows = legacy.load_run(pair_run)
    if parent_binding["split"] != "explore100":
        raise ValueError("Expected existing exploration split")
    rows = ordered_rows(rows)
    binding = {
        "protocol": "revalue-v3-r2-v2",
        "created_at": utc_now(),
        "baseline_run": str(baseline_run),
        "pair_run": str(pair_run),
        "r1_run": str(r1),
        "observer_run": str(args.observer_run.resolve()),
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
    }
    for root, names in [
        (
            r1,
            [
                "binding.json",
                "complete.json",
                "R0/gate.json",
                "formal_protocol.json",
                "R1/inference_complete.json",
                "R1/record_receipt.json",
                "R1/ledger.json",
                "R1/results/evaluation.json",
                "model_verification.json",
            ],
        ),
        (args.observer_run, ["binding.json", "complete.json", "R0/gate.json"]),
        (baseline_run, ["binding.json", "completion.json", "record_receipt.json", "inference.jsonl"]),
        (pair_run, ["binding.json", "completion.json", "record_receipt.json", "inference.jsonl"]),
    ]:
        for name in names:
            binding["parent_receipts"][str((root / name).resolve())] = sha256(root / name)
    reuse = {}
    for entry in read_json(r1 / "R1/ledger.json")["attempts"]:
        if "/M_plain/" in entry["key"] or entry["key"].endswith("/M_plain"):
            continue
        path = r1 / "R1/attempts" / f"{entry['number']:05d}.json"
        raw = read_json(path)
        if (
            entry["status"] != "complete"
            or digest(raw) != entry["record_sha256"]
            or digest(raw["specification"]) != raw["request_sha256"]
        ):
            raise ValueError("R1 cache did not pass exact receipt verification")
        reuse[entry["key"]] = {
            "path": str(path),
            "file_sha256": sha256(path),
            "record_sha256": entry["record_sha256"],
            "charged_tokens": entry["charged_tokens"],
            "original_arms": raw["specification"]["arms"],
        }
        binding["parent_receipts"][str(path)] = sha256(path)
    if len(reuse) != 20 * 47:
        raise ValueError("Expected exactly 47 reusable requests per R1 query")
    tokens, fallback_tokens, audit = 0, 0, []
    for row in rows:
        baseline, pair = parents(binding, row)
        tokens += sum(baseline["cost"][k] for k in ("input_tokens", "output_tokens"))
        fallback_tokens += sum(pair["arms"]["P_pair"]["cost"][k] for k in ("input_tokens", "output_tokens"))
        audit.append(
            {
                "qid": row["qid"],
                "candidate_ids": pair["candidate_ids"],
                "baseline_record_sha256": sha256(baseline_run / "records" / f"{row['qid']}.json"),
                "pair_record_sha256": sha256(pair_run / "records" / f"{row['qid']}.json"),
            }
        )
    binding["baseline_tokens"] = tokens
    binding["budgets"] = {
        "requests": 9200 - len(reuse),
        "extra_attempts": 460,
        "arm_incremental_tokens": {a: 5 * tokens for a in ARMS},
        "fallback_reserve_tokens": fallback_tokens,
    }
    run.mkdir(parents=True)
    legacy.write_jsonl(run / "inference.jsonl", rows)
    write_json(run / "reuse_index.json", reuse)
    write_json(run / "parent_audit.json", {"records": audit, "all_100_parents_verified": True})
    result = subprocess.run([sys.executable, "-m", "pytest", str(HERE / "tests"), "-q"], capture_output=True, text=True)
    (run / "cpu_tests.log").write_text(result.stdout + result.stderr)
    write_json(
        run / "cpu_tests.json",
        {"returncode": result.returncode, "log_sha256": sha256(run / "cpu_tests.log"), "executed_at": utc_now()},
    )
    if result.returncode:
        raise ValueError("R2 tests failed; preserve failed preparation")
    import torch
    from candidate_frames import prepare_candidates
    from r2_observer import source_timeline, timeline_hash

    torch.set_num_threads(2)
    from r2_worker import one_query

    cache = cache_backend(run / "cache_preflight", reuse)
    frames = []
    inherited = {}
    if args.media_preflight_from:
        previous = args.media_preflight_from.resolve()
        previous_binding = read_json(previous / "binding.json")
        if (
            previous_binding["baseline_run"] != str(baseline_run)
            or previous_binding["pair_run"] != str(pair_run)
            or sha256(previous / "candidate_inputs.jsonl") != previous_binding["input_sha256"]["candidate_inputs.jsonl"]
        ):
            raise ValueError("Inherited media preflight mismatch")
        if legacy.read_jsonl(previous / "inference.jsonl") != rows:
            raise ValueError("Inherited media preflight used a different query set")
        for name in ("binding.json", "candidate_inputs.jsonl", "inference.jsonl"):
            binding["parent_receipts"][str(previous / name)] = sha256(previous / name)
        inherited = {r["qid"]: r for r in legacy.read_jsonl(previous / "candidate_inputs.jsonl")}
        binding["media_preflight_from"] = str(previous)
    for index, row in enumerate(rows):
        baseline, pair = parents(binding, row)
        if inherited and f"{row['qid']}/task" not in reuse:
            item = inherited[row["qid"]]
            if (
                sha256(row["video_path"]) != item["source_sha256"]
                or item["candidate_frames"] != pair["candidate_frames"]
            ):
                raise ValueError("Inherited byte-verified source media changed")
            frames.append(item)
            print(f"R2 inherited full PTS preflight and rehashed source {index + 1}/100", flush=True)
            continue
        targets, traces, seconds = prepare_candidates(row, baseline, pair["candidate_ids"])
        if traces != pair["candidate_frames"]:
            raise ValueError("R2 changed C0 candidate pixels")
        reader, timestamps, seen = source_timeline(row, baseline)
        frames.append(
            {
                "qid": row["qid"],
                "candidate_frames": traces,
                "source_timeline_sha256": timeline_hash(timestamps),
                "B0_seen_indices": sorted(seen),
                "decode_seconds": seconds,
                "source_sha256": baseline["media"]["source_sha256"],
            }
        )
        if f"{row['qid']}/task" in reuse:
            replay = one_query(row, binding, "cache-only-audit", cache, frames[-1], (targets, traces, seconds))
            if (
                replay["arms"]["C_direct4"]["status"] != "complete"
                or replay["arms"]["E_memory"]["status"] != "complete"
            ):
                raise ValueError("R1 reused calls did not semantically validate under the unchanged R2 comparator")
        del targets, reader
        print(f"R2 frozen source/whole-B0 PTS {index + 1}/100", flush=True)
    legacy.write_jsonl(run / "candidate_inputs.jsonl", frames)
    if cache.ledger.verified != set(reuse):
        raise ValueError("Not every declared reusable request passed current prompt/evidence/schema verification")
    write_json(
        run / "cache_preflight.json",
        {
            "verified_requests": len(reuse),
            "new_model_calls": 0,
            "planner": "empty-plan CPU fixture only; not a model prediction",
        },
    )
    model = read_json(r1 / "model_verification.json")
    for name, expected in model["stats"].items():
        stat = (Path(model["audit"]["model_path"]) / name).stat()
        if stat.st_size != expected["size"] or stat.st_mtime_ns != expected["mtime_ns"]:
            raise ValueError("Previously byte-verified model changed")
    shutil.copy2(r1 / "model_verification.json", run / "model_verification.json")
    shutil.copy2(r1 / "evaluation_binding.json", run / "evaluation_binding.json")
    shutil.copy2(args.execution_plan, run / "execution_plan.md")
    shutil.copy2(args.implementation_plan, run / "implementation_plan.md")
    write_json(
        run / "plan.json",
        {
            "queries": 100,
            "selection": "original explore100 sorted by SHA256('revalue-v3:' + qid)",
            "arms": ["E0", *ARMS],
            "maximum_calls_before_reuse": 9200,
            "verified_reusable_requests": len(reuse),
            "comparison_and_memory_prompts": "unchanged from completed R1",
            "observation_prompt": "unchanged from passed observer v5",
            "planner": "one frozen LocalPlan for actionable insufficient gaps or directional conflicts; no forced tie breaking",
            "observation_pairs": "max 2 per query, max 24 unique new source frames each; exclude all B0 frames",
            "memory_update": "deterministic whole-entry cap 1024; no raw fact bypass, no model recompression",
            "fallback": "C_direct4/E_memory -> E0; observer failures -> valid E_memory; keep all 100",
            "resource_accounting": "global atomic reservations, exact reuse included in cold method cost, summed GPU seconds",
            "gate_reference": "highest unrounded control mAP; fixed tie order E0,C_direct4,E_memory,E_uniform; Hit1 vs same control",
            "no_labels_in_inference": True,
            "R3_or_R4_auto_start": False,
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
    print(f"Prepared {run}; binding={sha256(run / 'binding.json')}; max new requests={9200 - len(reuse)}", flush=True)


if __name__ == "__main__":
    main()
