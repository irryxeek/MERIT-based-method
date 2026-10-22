"""Freeze query IDs, media, code, model and separately held evaluation inputs."""

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from shared import HERE, atomic_json, query_text, segments_for, sha256, utc_now, write_jsonl

QV = Path(os.environ.get("ZEROSHOT_ROOT", "baselines/zeroshot"))
DATA = Path(os.environ.get("QFVS_DATA_ROOT", "data/QFVS"))
REVISION = "32659ac7aeba21742a63274f30eba785fc57e247"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--scope", choices=["pilot4", "full180"], default="pilot4")
    args = parser.parse_args()
    if not os.environ.get("REPRO_GPU_UUIDS", "").strip():
        parser.error("Set REPRO_GPU_UUIDS to the allocated NVIDIA GPU UUIDs")
    run = args.run.resolve()
    if (run / "binding.json").exists():
        raise ValueError("Run is already frozen; preserve it and choose another directory")
    run.mkdir(parents=True, exist_ok=True)
    media_audit = json.loads((DATA / "raw/video_audit.json").read_text())
    original = DATA / "data/origin_data"
    rows_by_video, evaluation_files, media_receipts = [], {}, {}
    for video in media_audit["videos"]:
        vid = video["video_id"]
        path = Path(video["path"])
        if sha256(path) != video["sha256"]:
            raise ValueError(f"Media hash changed: {vid}")
        stat = path.stat()
        names = sorted(
            p.name.removesuffix("_oracle.txt")
            for p in (original / "Query-Focused_Summaries/Oracle_Summaries" / vid).glob("*_oracle.txt")
        )
        if len(names) != 45:
            raise ValueError("Expected 45 released nonempty query names per video")
        # Names only: neither summary positions nor semantic tags influence selection.
        names.sort(key=lambda name: hashlib.sha256(f"42:{vid}:{name}".encode()).hexdigest())
        if args.scope == "pilot4":
            names = names[:1]
        selected = []
        for name in names:
            concepts = name.split("_")
            row = {
                "qid": f"{vid}__{name}",
                "vid": vid,
                "query_name": name,
                "concepts": concepts,
                "query": query_text(concepts),
                "duration": video["duration_seconds"],
                "video_path": str(path),
                "media_sha256": video["sha256"],
                "media_size": stat.st_size,
                "media_mtime_ns": stat.st_mtime_ns,
                "n_shots": video["native_shots"],
            }
            selected.append(row)
            relative = f"oracle/{vid}/{name}.txt"
            evaluation_files[relative] = (
                original / "Query-Focused_Summaries/Oracle_Summaries" / vid / f"{name}_oracle.txt"
            )
            for user in range(1, 4):
                relative = f"user{user}/{vid}/{name}.txt"
                evaluation_files[relative] = (
                    original / "Query-Focused_Summaries/User_Summaries" / vid / f"{name}_user{user}.txt"
                )
        rows_by_video.append(selected)
        media_receipts[vid] = video
    # Interleave videos so early completed queries cover all four sources.
    rows = [items[i] for i in range(len(rows_by_video[0])) for items in rows_by_video]
    write_jsonl(run / "inference.jsonl", rows)
    evaluation_files["Tags.mat"] = DATA / "source/eval/Tags.mat"
    label_hashes = {}
    for name, source in evaluation_files.items():
        target = run / "evaluation_inputs" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        label_hashes[name] = sha256(target)
    atomic_json(run / "media_verification.json", media_receipts)
    audit = json.loads((QV / "manifests/reze_model_audit.json").read_text())
    model_stats = {}
    for name, expected in audit["files"].items():
        path = Path(audit["model_path"]) / name
        before = path.stat()
        if before.st_size != expected["size"] or sha256(path) != expected["sha256"]:
            raise ValueError(f"Model hash mismatch: {name}")
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("Model mutated during verification")
        model_stats[name] = {"size": after.st_size, "mtime_ns": after.st_mtime_ns}
    atomic_json(run / "model_verification.json", {"audit": audit, "stats": model_stats, "checked_at": utc_now()})
    source_hashes = {}
    for path in sorted(HERE.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or ".pytest_cache" in path.parts:
            continue
        if path.suffix not in {".py", ".md", ".LICENSE", ".json"}:
            continue
        name = str(path.relative_to(HERE))
        target = run / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        source_hashes[name] = sha256(path)
    calls = sum(len(segments_for(row["duration"])) for row in rows)
    binding = {
        "schema": "reze_qfvs_v1",
        "created_at": utc_now(),
        "scope": args.scope,
        "dataset": "QFVS",
        "n_queries": len(rows),
        "n_videos": 4,
        "selected_qids": [r["qid"] for r in rows],
        "selection_rule": "ascending sha256('42:vid:query_name'); one per video for pilot4, all for full180",
        "selection_used_label_or_metric_values": False,
        "method": "Local QV B0 implementation of REZE Appendix A, transferred to QFVS",
        "window_seconds": 3,
        "frames_per_full_window": 6,
        "sampling_fps": 2,
        "query_template": "The video shows either {concept_1} or {concept_2}.",
        "query_semantics": "OR relevance; no label-driven query rewriting; three spelling aliases only",
        "readout": "REZE inclusive integer-second curve; mean available seconds in each native 5-second shot",
        "summary_fraction": 0.02,
        "summary_budget": "floor(native_shots * 0.02)",
        "score_ties": "stable descending sort; lower native shot index first",
        "primary_metrics": ["semantic_precision_oracle", "semantic_recall_oracle", "semantic_F1_oracle"],
        "secondary_metrics": ["semantic_precision_user_mean", "semantic_recall_user_mean", "semantic_F1_user_mean"],
        "evaluator_revision": REVISION,
        "evaluator_zero_overlap_policy": "zero P/R/F1 when all semantic edge weights are zero",
        "labels_sha256": label_hashes,
        "batch_size": 16,
        "expected_model_requests": calls,
        "maximum_charged_requests": calls + 48 + 12,
        "maximum_worker_attempts": 3,
        "maximum_active_worker_seconds": (12 if args.scope == "pilot4" else 168) * 3600,
        "maximum_worker_stall_seconds": 900,
        "smoke_windows": "first/middle/last retained window per video, 12 GPU requests before full coverage",
        "max_gpus": 1,
        "require_no_other_compute_process": True,
        "gpu_candidates": [uuid.strip() for uuid in os.environ.get("REPRO_GPU_UUIDS", "").split(",") if uuid.strip()],
        "gpu_addressing": "numeric NVML index for vLLM 0.12; verify actual torch UUID before engine load",
        "model_path": audit["model_path"],
        "model_revision": audit["revision"],
        "model_verification_sha256": sha256(run / "model_verification.json"),
        "media_verification_sha256": sha256(run / "media_verification.json"),
        "inference_sha256": sha256(run / "inference.jsonl"),
        "source_sha256": source_hashes,
        "baseline_code_sha256": {
            name: sha256(QV / "scripts" / name)
            for name in [
                "reze_core.py",
                "reze_backend.py",
                "reze_media.py",
            ]
        },
        "inference_uses_labels": False,
        "training": False,
        "QFVS_published_REZE_result": False,
    }
    atomic_json(run / "binding.json", binding)
    atomic_json(
        run / "status.json",
        {
            "status": "prepared",
            "updated_at": utc_now(),
            "scope": args.scope,
            "completed_queries": 0,
            "expected_queries": len(rows),
            "planned_requests": calls,
        },
    )
    print(json.dumps({"run": str(run), "queries": len(rows), "planned_requests": calls}), flush=True)


if __name__ == "__main__":
    main()
