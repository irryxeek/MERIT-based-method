"""Build fresh label-free REZE manifests and verify the pinned model on this machine."""

import argparse
import json
import os
from pathlib import Path

from common import ROOT, atomic_json, read_jsonl, sha256, utc_now, write_jsonl
from reze_backend import MODEL_PATH


def inference_rows(released_rows, video_root):
    if len(released_rows) != 1550 or len({row["qid"] for row in released_rows}) != 1550:
        raise ValueError("Expected all 1550 unique QVHighlights validation queries")
    rows = []
    for row in released_rows:
        path = (Path(video_root) / (row["vid"] + ".mp4")).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if row["duration"] < 2:
            raise ValueError("Invalid annotation duration")
        rows.append({**{key: row[key] for key in ("qid", "vid", "query", "duration")}, "video_path": str(path)})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path(MODEL_PATH))
    parser.add_argument("--model-only", action="store_true", help="Prepare only the model audit for cross-dataset runs")
    parser.add_argument(
        "--video-root",
        type=Path,
        default=Path(os.environ.get("REPRO_DATA_ROOT", "data")) / "QVHighlights/videos_full/videos",
    )
    parser.add_argument(
        "--evaluator-root", type=Path, default=Path(os.environ.get("QVH_EVALUATOR_ROOT", "third_party/moment_detr"))
    )
    args = parser.parse_args()
    audit_path = ROOT / "manifests/reze_model_audit.json"
    outputs = [audit_path]
    if not args.model_only:
        outputs.extend(
            ROOT / name
            for name in (
                "manifests/reze_val_inference.jsonl",
                "manifests/reze_smoke_inference.jsonl",
                "manifests/data_audit.json",
                "configs/protocol_v1.json",
            )
        )
    if any(path.exists() for path in outputs):
        raise FileExistsError("Preserve prepared inputs; use a fresh checkout or inspect existing files")
    template = json.loads((ROOT / "manifests/reze_model_audit.example.json").read_text())
    model = args.model.resolve()
    for name, expected in template["files"].items():
        path = model / name
        if path.stat().st_size != expected["size"] or sha256(path) != expected["sha256"]:
            raise ValueError(f"Pinned model file differs: {name}")
    audit = {
        "model": template["model"],
        "revision": template["revision"],
        "model_path": str(model),
        "files": template["files"],
        "verified_at": utc_now(),
    }
    rows = None
    if not args.model_only:
        evaluator = args.evaluator_root.resolve()
        labels = evaluator / "data/highlight_val_release.jsonl"
        rows = inference_rows(read_jsonl(labels), args.video_root)
        protocol = {
            "schema": "qvh_reze_publication_v1",
            "split": "val",
            "expected_queries": 1550,
            "evaluator_root": str(evaluator),
            "labels_path": str(labels),
            "labels_sha256": sha256(labels),
            "manifest": str(ROOT / "manifests/reze_val_inference.jsonl"),
            "smoke_manifest": str(ROOT / "manifests/reze_smoke_inference.jsonl"),
            "primary_metrics": ["HL-min-VeryGood-mAP", "HL-min-VeryGood-Hit1"],
        }
        data_audit = {
            "label_sha256": sha256(labels),
            "evaluator_hashes": {
                name: sha256(evaluator / "standalone_eval" / name) for name in ("eval.py", "utils.py")
            },
        }
    atomic_json(audit_path, audit)
    if rows is not None:
        write_jsonl(ROOT / "manifests/reze_val_inference.jsonl", rows)
        # Include a later query so the sustained 20+1 gate covers 21 distinct IDs.
        write_jsonl(ROOT / "manifests/reze_smoke_inference.jsonl", rows[:4] + rows[-1:])
        atomic_json(ROOT / "manifests/data_audit.json", data_audit)
        atomic_json(ROOT / "configs/protocol_v1.json", protocol)
    print(json.dumps({"model_verified": True, "queries": len(rows) if rows is not None else 0, "model_called": False}))


if __name__ == "__main__":
    main()
