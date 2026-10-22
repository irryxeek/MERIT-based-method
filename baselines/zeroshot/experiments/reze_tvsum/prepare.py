"""Freeze all 50 TVSum videos in two label-blind REZE inference shards."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import sys
from pathlib import Path

from dataset import UMT_DATA, UMT_REVISION
from media import StreamVideo
from shared import HERE, QUERY, atomic_json, sha256, utc_now, write_jsonl

QV = Path(os.environ.get("ZEROSHOT_ROOT", "baselines/zeroshot"))


def selection(audit, index=0, count=2):
    if count not in (1, 2) or not 0 <= index < count:
        raise ValueError("Invalid one/two shard selection")
    ids = sorted(audit["videos"], key=lambda vid: hashlib.sha256(f"42:{vid}".encode()).hexdigest())
    if len(ids) != 50:
        raise ValueError("All 50 TVSum videos are required")
    return ids[index::count]


def source_snapshot(target):
    hashes = {}
    for source in sorted(HERE.rglob("*")):
        relative = source.relative_to(HERE)
        if not source.is_file() or any(p.startswith(".") or p == "__pycache__" for p in relative.parts):
            continue
        if source.suffix not in {".py", ".md", ".LICENSE", ".json"}:
            continue
        dest = target / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        hashes[str(relative)] = sha256(dest)
    return hashes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--all-verified", action="store_true", required=True)
    parser.add_argument("--shard-count", type=int, choices=(1, 2), default=2)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--gpu-uuids", nargs="+", required=True)
    args = parser.parse_args()
    if len(set(args.gpu_uuids)) != len(args.gpu_uuids) or any(not s.startswith("GPU-") for s in args.gpu_uuids):
        parser.error("Expected unique GPU UUIDs")
    run = args.run.resolve()
    if run.exists():
        raise FileExistsError("Preserve previous attempts; select a new directory")
    audit = json.loads((UMT_DATA / "dataset_audit.json").read_text())
    if audit["status"] != "passed" or sha256(UMT_DATA / "tvsum_anno.json") != audit["annotation_sha256"]:
        raise ValueError("Dataset audit missing or annotation bytes changed")
    selected = selection(audit, args.shard_index, args.shard_count)
    run.mkdir(parents=True)
    rows, media, windows = [], {}, 0
    for index, vid in enumerate(selected, 1):
        receipt = audit["videos"][vid]
        path = Path(receipt["video_path"])
        before = path.stat()
        if before.st_size != receipt["media_size"] or sha256(path) != receipt["media_sha256"]:
            raise ValueError(f"Media changed: {vid}")
        if before.st_mtime_ns != path.stat().st_mtime_ns:
            raise ValueError("Media mutated during verification")
        row = {
            "qid": vid,
            "vid": vid,
            "query": QUERY,
            "duration": receipt["duration"],
            "n_seconds": int(receipt["duration"]),
            "video_path": str(path),
            "media_sha256": receipt["media_sha256"],
            "media_size": before.st_size,
            "media_mtime_ns": before.st_mtime_ns,
        }
        video = StreamVideo(row)
        if len(video.reader) != receipt["media_frames"] or int(row["duration"] / 2) != receipt["n_2s_clips"]:
            raise ValueError("Decoded video differs from frozen author grid")
        windows += len(video.segments)
        media[vid] = {"metadata": receipt, "decoded_timeline": video.describe()}
        rows.append(row)
        del video
        atomic_json(
            run / "status.json",
            {
                "status": "verifying_media",
                "updated_at": utc_now(),
                "verified_videos": index,
                "expected_queries": len(selected),
                "active_vid": vid,
                "model_called": False,
            },
        )
    write_jsonl(run / "inference.jsonl", rows)
    atomic_json(run / "media_verification.json", media)
    (run / "evaluation_inputs").mkdir()
    labels_hashes = {}
    for name in ("tvsum_anno.json", "dataset_audit.json", "download_receipt.json"):
        dest = run / "evaluation_inputs" / name
        shutil.copy2(UMT_DATA / name, dest)
        labels_hashes[name] = sha256(dest)
    model_audit = json.loads((QV / "manifests/reze_model_audit.json").read_text())
    stats = {}
    for name, expected in model_audit["files"].items():
        path = Path(model_audit["model_path"]) / name
        before = path.stat()
        if before.st_size != expected["size"] or sha256(path) != expected["sha256"]:
            raise ValueError(f"Model hash mismatch: {name}")
        if before.st_mtime_ns != path.stat().st_mtime_ns:
            raise ValueError("Model mutated during verification")
        stats[name] = {"size": before.st_size, "mtime_ns": before.st_mtime_ns}
    atomic_json(run / "model_verification.json", {"audit": model_audit, "stats": stats, "checked_at": utc_now()})
    baseline_hashes = {}
    for name in ("reze_core.py", "reze_backend.py", "reze_media.py"):
        baseline_hashes[name] = sha256(HERE / "vendor" / name)
        if baseline_hashes[name] != sha256(QV / "scripts" / name):
            raise ValueError("Original QV REZE scorer changed")
    sources = source_snapshot(run / "source_snapshot")
    binding = {
        "schema": "reze_tvsum_v1",
        "created_at": utc_now(),
        "dataset": "TVSum",
        "scope": f"full50_shard_{args.shard_index}_of_{args.shard_count}",
        "n_queries": len(rows),
        "n_videos": len(rows),
        "selected_qids": selected,
        "target_videos_across_shards": 50,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "query": QUERY,
        "query_source": "Fixed generic query; no titles, category cues, captions or labels",
        "window_seconds": 3,
        "sampling_fps": 2,
        "frames_per_full_window": 6,
        "batch_size": 16,
        "readout": "Unmodified REZE inclusive integer-second curve; mean of each consecutive pair of seconds",
        "timeline_policy": "Use original media duration; score min(metadata duration, decoded end); reject >2s discrepancy",
        "evaluation_policy": "Pinned UMT top-5 trapezoidal AP; strict per-annotator > torch.median; CPU float32 argsort",
        "evaluation_scopes": ["UMT fixed validation 10 (primary)", "all 50 zero-shot transfer (secondary)"],
        "evaluator_revision": UMT_REVISION,
        "baseline_controls": "Constant scores with author tie order; exact expected top-5 AP under uniform random ranking",
        "primary_metrics": ["Top5_mAP"],
        "summary_F1_computed": False,
        "labels_sha256": labels_hashes,
        "inference_uses_labels": False,
        "expected_model_requests": windows,
        "smoke_video_ids": selected[:3],
        "maximum_charged_requests": windows + 128,
        "maximum_worker_attempts": 3,
        "maximum_active_worker_seconds": 7200,
        "maximum_worker_stall_seconds": 900,
        "max_gpus": 1,
        "require_no_other_compute_process": True,
        "gpu_candidates": args.gpu_uuids,
        "gpu_lease_directory": (os.environ.get("ZEROSHOT_ARTIFACT_ROOT", "artifacts/zeroshot") + "/gpu_leases"),
        "model_path": model_audit["model_path"],
        "model_revision": model_audit["revision"],
        "model_verification_sha256": sha256(run / "model_verification.json"),
        "media_verification_sha256": sha256(run / "media_verification.json"),
        "inference_sha256": sha256(run / "inference.jsonl"),
        "source_sha256": sources,
        "baseline_code_sha256": baseline_hashes,
        "training": False,
        "original_REZE_paper_dataset_result": False,
    }
    atomic_json(run / "binding.json", binding)
    atomic_json(
        run / "runtime.json",
        {
            "frozen_at": utc_now(),
            "python": sys.version,
            "packages": {
                n: importlib.metadata.version(n)
                for n in ("torch", "transformers", "vllm", "decord", "numpy", "scipy", "scikit-learn")
            },
            "environment_changed": False,
        },
    )
    atomic_json(
        run / "status.json",
        {
            "status": "prepared",
            "updated_at": utc_now(),
            "expected_queries": len(rows),
            "completed_queries": 0,
            "planned_requests": windows,
        },
    )
    print(json.dumps({"run": str(run), "videos": len(rows), "planned_requests": windows}), flush=True)


if __name__ == "__main__":
    main()
