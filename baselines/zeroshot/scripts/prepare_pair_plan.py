"""Bind a new full-validation P_abs/P_pair plan to a completed B0 on this machine."""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from common import ROOT, sha256

sys.path.insert(0, str(ROOT / "experiments/reze_fullval"))
from runner import check_complete, protocol  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
    args = parser.parse_args()
    if args.output.exists() or args.store.exists():
        raise FileExistsError("Select new plan and store paths; preserve historical runs")
    if len(set(args.gpus)) != len(args.gpus) or any(g < 0 for g in args.gpus):
        parser.error("Expected unique nonnegative GPU indices")
    parent = args.baseline_run.resolve()
    rows = protocol.read_jsonl(parent / "inference.jsonl")
    if len(rows) != 1550:
        raise ValueError("This plan requires the complete 1550-query B0")
    check_complete(parent, rows)
    repo = Path(os.environ.get("MERIT_ROOT", ROOT.parents[1]))
    plan = json.loads((repo / "configs/reze_evidence_fullval_v1.example.json").read_text())
    evaluator = Path(os.environ.get("QVH_EVALUATOR_ROOT", repo / "third_party/moment_detr")).resolve()
    model_audit = ROOT / "manifests/reze_model_audit.json"
    model = json.loads(model_audit.read_text())
    store = args.store.resolve()
    plan.update(
        date=datetime.now(timezone.utc).date().isoformat(),
        authorization="New reproduction plan explicitly created by the invoking operator; no inference started.",
        baseline_run=str(parent),
        baseline_completion_sha256=sha256(parent / "completion.json"),
        baseline_prediction_sha256=sha256(parent / "predictions.jsonl"),
        manifest_sha256=sha256(parent / "inference.jsonl"),
        model_audit_sha256=sha256(model_audit),
        model_revision=model["revision"],
        store=str(store),
        archive=str(store / "publication"),
        result_document=str(store / "REPORT.md"),
        physical_gpus=args.gpus,
        evaluator_root=str(evaluator),
        labels_path=str(evaluator / "data/highlight_val_release.jsonl"),
        labels_sha256=sha256(evaluator / "data/highlight_val_release.jsonl"),
        publication_note="Paths and receipts bind this new run; historical predictions and plans remain unchanged.",
    )
    plan["frozen_method_sources"] = {name: sha256(ROOT / name) for name in plan["frozen_method_sources"]}
    plan["evaluator_sha256"] = {name: sha256(evaluator / "standalone_eval" / name) for name in plan["evaluator_sha256"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
