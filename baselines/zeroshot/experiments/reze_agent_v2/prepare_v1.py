"""Freeze V1 inputs, audited parents, exact source and the resource schedule."""

import argparse
import json
import shutil
from pathlib import Path

from core import ROOT, dispersed_candidates, legacy, parent_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--pair-run", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--after-run", type=Path, required=True)
    parser.add_argument("--execution-plan", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    if run.exists():
        raise ValueError("New V1 run directory required")
    baseline_run, pair_run = args.baseline_run.resolve(), args.pair_run.resolve()
    old_binding, rows = legacy.load_run(pair_run)
    if len(rows) != 100 or old_binding["split"] != "explore100":
        raise ValueError("Expected original explore100 parent")
    for parent in (baseline_run, pair_run):
        complete = json.loads((parent / "completion.json").read_text())
        if complete["status"] != "complete" or complete["queries"] != 100:
            raise ValueError("Incomplete parent")
        if complete["binding_sha256"] != legacy.sha256(parent / "binding.json"):
            raise ValueError("Parent binding mismatch")
        if complete["record_receipt_sha256"] != legacy.sha256(parent / "record_receipt.json"):
            raise ValueError("Parent receipt mismatch")
    binding = {
        "schema": "reze_agent_v2_candidate_v1",
        "stage": "V1",
        "created_at": legacy.utc_now(),
        "baseline_run": str(baseline_run),
        "pair_run": str(pair_run),
        "gpu_candidates": [args.gpu_uuid],
        "after_run": str(args.after_run.resolve()),
        "after_binding_sha256": legacy.sha256(args.after_run / "binding.json"),
        "maximum_active_worker_seconds": 7200,
        "maximum_worker_attempts": 3,
        "maximum_worker_stall_seconds": 900,
        "formal_requests": 2000,
        "maximum_formal_attempts": 2100,
        "maximum_engineering_requests": 100,
        "engineering_qids": [r["qid"] for r in sorted(rows, key=lambda r: legacy.digest([42, r["qid"]]))[:5]],
        "parent_receipts": {},
        "input_sha256": {},
        "source_sha256": {},
    }
    for parent in (baseline_run, pair_run):
        for name in ("binding.json", "completion.json", "record_receipt.json", "inference.jsonl"):
            binding["parent_receipts"][str(parent / name)] = legacy.sha256(parent / name)
    audit, tokens = [], 0
    for row in rows:
        baseline, pair = parent_records(binding, row)
        selected, selection = dispersed_candidates(baseline["prediction"]["pred_saliency_scores"])
        tokens += baseline["cost"]["input_tokens"] + baseline["cost"]["output_tokens"]
        audit.append(
            {
                "qid": row["qid"],
                "C0": pair["candidate_ids"],
                "C1": selected,
                "selection": selection,
                "baseline_record_sha256": legacy.sha256(baseline_run / "records" / f"{row['qid']}.json"),
                "pair_record_sha256": legacy.sha256(pair_run / "records" / f"{row['qid']}.json"),
            }
        )
    binding.update(
        baseline_tokens=tokens,
        maximum_incremental_tokens=5 * tokens,
        model_revision=old_binding["model_revision"],
        environment=old_binding["environment"],
    )
    run.mkdir(parents=True)
    legacy.write_jsonl(run / "inference.jsonl", rows)
    legacy.atomic_json(run / "parent_audit.json", {"queries": len(rows), "records": audit, "model_calls": 0})
    import torch
    from candidate_frames import prepare_candidates

    torch.set_num_threads(2)
    frame_receipts = []
    for index, row in enumerate(rows):
        baseline, pair = parent_records(binding, row)
        selected, _ = dispersed_candidates(baseline["prediction"]["pred_saliency_scores"])
        union = sorted(set(selected + pair["candidate_ids"]))
        targets, traces, seconds = prepare_candidates(row, baseline, union)
        for clip_id in pair["candidate_ids"]:
            if traces[str(clip_id)] != pair["candidate_frames"][str(clip_id)]:
                raise ValueError("Reconstructed C0 pixels or timestamps differ")
        frame_receipts.append(
            {
                "qid": row["qid"],
                "candidate_frames": {str(i): traces[str(i)] for i in selected},
                "decode_preprocess_seconds": seconds,
            }
        )
        del targets
        if (index + 1) % 10 == 0:
            print(f"Verified C0/C1 frame provenance: {index + 1}/100", flush=True)
    legacy.write_jsonl(run / "candidate_inputs.jsonl", frame_receipts)
    model = json.loads((ROOT / "manifests/reze_model_audit.json").read_text())
    stats = {}
    for name, expected in model["files"].items():
        path = Path(model["model_path"]) / name
        if path.stat().st_size != expected["size"] or legacy.sha256(path) != expected["sha256"]:
            raise ValueError(f"Model bytes changed: {name}")
        stats[name] = {"size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
    legacy.atomic_json(run / "model_verification.json", {"audit": model, "stats": stats})
    shutil.copy2(args.execution_plan, run / "execution_plan.md")
    original_audit = Path(args.pair_run).parents[1] / "audit/inputs_v1.json"
    shutil.copy2(original_audit, run / "evaluation_binding.json")
    plan = {
        "stage": "V1",
        "dataset": "QVHighlights explore100",
        "candidate_count": 5,
        "C0": "original score top5, ties by clip ID",
        "C1": "greedy score order, suppress distance <=2 clip IDs; fill from original order",
        "comparison": "unchanged P_pair, 20 directed comparisons per query",
        "gate": {"minimum_mAP_gain_pp": 1, "minimum_Hit1_delta_pp": -1, "fallback_increase_max": 0},
        "on_reject": "V2 uses C0",
        "next_stage": "V2 requires implemented and verified observer",
        "inference_uses_labels": False,
        "no_new_source_frames": True,
        "scope": "development only; no automatic confirm or full-validation expansion",
    }
    legacy.atomic_json(run / "plan.json", plan)
    for name in (
        "inference.jsonl",
        "parent_audit.json",
        "candidate_inputs.jsonl",
        "model_verification.json",
        "execution_plan.md",
        "plan.json",
        "evaluation_binding.json",
    ):
        binding["input_sha256"][name] = legacy.sha256(run / name)
    names = set(legacy.source_hashes())
    names.update(
        str(p.relative_to(ROOT))
        for p in (ROOT / "experiments/reze_agent_v2").rglob("*")
        if p.is_file() and p.suffix in {".py", ".md"} and "__pycache__" not in p.parts
    )
    for name in sorted(names):
        target = run / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
        binding["source_sha256"][name] = legacy.sha256(target)
    legacy.atomic_json(run / "binding.json", binding)
    legacy.atomic_json(
        run / "budget.json",
        {k: v for k, v in binding.items() if k.startswith("maximum_") or k in {"formal_requests", "baseline_tokens"}},
    )
    legacy.atomic_json(
        run / "status.json",
        {
            "status": "prepared_waiting_for_parent",
            "updated_at": legacy.utc_now(),
            "completed_queries": 0,
            "expected_queries": 100,
        },
    )
    print(json.dumps({"run": str(run), "baseline_tokens": tokens, "formal_requests": 2000}))


if __name__ == "__main__":
    main()
