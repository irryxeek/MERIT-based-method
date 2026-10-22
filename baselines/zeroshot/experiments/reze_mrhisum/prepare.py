"""Freeze a label-blind MR.HiSum sample or all verified test media."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

from media import StreamVideo
from shared import HERE, QUERY, atomic_json, sha256, utc_now, write_jsonl

QV = Path(os.environ.get("ZEROSHOT_ROOT", "baselines/zeroshot"))
DATA = Path((os.environ.get("REPRO_DATA_ROOT", "data") + "/MR.HiSum"))
OFFICIAL = Path(os.environ.get("MRHISUM_SOURCE_ROOT", "third_party/MR.HiSum"))
REVISION = "aeb667f71239887034c2974062a341d312762867"


def select_videos(test_ids, receipts, count):
    if len(test_ids) != len(set(test_ids)) or set(receipts) != set(test_ids):
        raise ValueError("Official test split and media receipts disagree")
    eligible = [vid for vid in test_ids if receipts[vid]["status"] == "downloaded_verified"]
    eligible.sort(key=lambda vid: hashlib.sha256(f"42:{vid}".encode()).hexdigest())
    if count is None:
        count = len(eligible)
    if not 1 <= count <= len(eligible):
        raise ValueError("Insufficient verified media for fixed sample")
    return eligible[:count], eligible


def execution_limits(all_verified, shard_count=1):
    return {
        "maximum_worker_attempts": 3,
        "maximum_active_worker_seconds": (24 * 3600 // shard_count) if all_verified else 2 * 3600,
        "request_overhead": (512 // shard_count) if all_verified else 128,
    }


def shard_selection(selected, shard_index, shard_count):
    if not 1 <= shard_count <= 2 or not 0 <= shard_index < shard_count:
        raise ValueError("Use one or two disjoint shards and a valid shard index")
    result = selected[shard_index::shard_count]
    if not result:
        raise ValueError("Empty inference shard")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--count", type=int, default=100)
    selection.add_argument("--all-verified", action="store_true")
    parser.add_argument("--shard-count", type=int, choices=[1, 2], default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--gpu-uuids",
        nargs="+",
        default=[uuid.strip() for uuid in os.environ.get("REPRO_GPU_UUIDS", "").split(",") if uuid.strip()],
    )
    args = parser.parse_args()
    if (
        not args.gpu_uuids
        or len(set(args.gpu_uuids)) != len(args.gpu_uuids)
        or any(not v.startswith("GPU-") for v in args.gpu_uuids)
    ):
        parser.error("Provide unique NVIDIA GPU UUIDs")
    if args.shard_count > 1 and not args.all_verified:
        parser.error("Sharded execution requires --all-verified")
    limits = execution_limits(args.all_verified, args.shard_count)
    run = args.run.resolve()
    if run.exists():
        raise FileExistsError("Preserve previous attempts and choose a new run directory")
    availability = json.loads((DATA / "availability.json").read_text())
    if availability["status"] != "queue_finished":
        raise ValueError("Freeze sampling after the download queue has finished")
    receipts_path = Path(availability["media_index"])
    receipts = json.loads(receipts_path.read_text())
    split = json.loads((DATA / "annotations/mr_hisum_split.json").read_text())
    selected, eligible = select_videos(split["test_keys"], receipts, None if args.all_verified else args.count)
    target_count = len(selected)
    selected = shard_selection(selected, args.shard_index, args.shard_count)
    scope = f"full_available_{target_count}" if args.all_verified else f"pilot{len(selected)}"
    if args.shard_count > 1:
        scope += f"_shard_{args.shard_index}_of_{args.shard_count}"
    if len(split["test_keys"]) != 2000:
        raise ValueError("Unexpected official test split")
    run.mkdir(parents=True)
    rows, media_receipts, windows = [], {}, 0
    for index, vid in enumerate(selected, 1):
        receipt = receipts[vid]
        path = Path(receipt["path"]).resolve()
        if path.stat().st_size != receipt["bytes"] or sha256(path) != receipt["sha256"]:
            raise ValueError(f"Media hash changed: {vid}")
        duration = receipt["annotated_seconds"]
        if int(duration) != duration:
            raise ValueError("Nonintegral annotation duration")
        stat = path.stat()
        row = {
            "qid": vid,
            "vid": vid,
            "query": QUERY,
            "duration": int(duration),
            "n_seconds": int(duration),
            "video_path": str(path),
            "media_sha256": receipt["sha256"],
            "media_size": stat.st_size,
            "media_mtime_ns": stat.st_mtime_ns,
        }
        try:
            video = StreamVideo(row)
        except Exception as error:
            raise ValueError(f"Media verification failed for {vid}: {error}") from error
        windows += len(video.segments)
        media_receipts[vid] = {"download_receipt": receipt, "decoded_timeline": video.describe()}
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
                "planned_requests_so_far": windows,
                "model_called": False,
            },
        )
        if index % 25 == 0 or index == len(selected):
            print(f"Verified {index}/{len(selected)} media files", flush=True)
    write_jsonl(run / "inference.jsonl", rows)
    atomic_json(run / "media_verification.json", media_receipts)
    atomic_json(
        run / "sampling_audit.json",
        {
            "selected": selected,
            "eligible_ids": eligible,
            "official_test_count": 2000,
            "availability_snapshot": availability,
            "receipts_sha256": sha256(receipts_path),
            "excluded_media": [
                {"vid": vid, "status": receipts[vid]["status"]}
                for vid in split["test_keys"]
                if vid not in set(eligible)
            ],
            "eligible_not_selected": len(eligible) - len(selected),
            "target_videos_across_shards": target_count,
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "label_values_read": False,
        },
    )
    # Copy the release bytes without parsing label values in the preparation or GPU processes.
    annotation_audit = json.loads((DATA / "annotation_audit.json").read_text())
    labels_hashes = {}
    (run / "evaluation_inputs").mkdir()
    for name in ("mr_hisum.h5", "mr_hisum_split.json", "metadata.csv"):
        source = DATA / "annotations" / name
        if sha256(source) != annotation_audit["files"][name]["sha256"]:
            raise ValueError(f"Official annotation file changed: {name}")
        shutil.copy2(source, run / "evaluation_inputs" / name)
        labels_hashes[name] = sha256(run / "evaluation_inputs" / name)
    audit = json.loads((QV / "manifests/reze_model_audit.json").read_text())
    model_stats = {}
    for name, expected in audit["files"].items():
        path = Path(audit["model_path"]) / name
        before = path.stat()
        if before.st_size != expected["size"] or sha256(path) != expected["sha256"]:
            raise ValueError(f"Model hash mismatch: {name}")
        after = path.stat()
        if before.st_mtime_ns != after.st_mtime_ns or before.st_size != after.st_size:
            raise ValueError("Model mutated during verification")
        model_stats[name] = {"size": after.st_size, "mtime_ns": after.st_mtime_ns}
    atomic_json(run / "model_verification.json", {"audit": audit, "stats": model_stats, "checked_at": utc_now()})
    for name in ("reze_core.py", "reze_backend.py", "reze_media.py"):
        if sha256(HERE / "vendor" / name) != sha256(QV / "scripts" / name):
            raise ValueError(f"QV baseline component changed: {name}")
    for path in (HERE / "vendor/model").rglob("*.py"):
        if sha256(path) != sha256(OFFICIAL / path.relative_to(HERE / "vendor")):
            raise ValueError("Official evaluator component changed")
    source_hashes = {}
    for path in sorted(HERE.rglob("*")):
        if not path.is_file() or any(p.startswith(".") or p == "__pycache__" for p in path.relative_to(HERE).parts):
            continue
        if path.suffix not in {".py", ".md", ".LICENSE", ".json"}:
            continue
        name = str(path.relative_to(HERE))
        target = run / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        source_hashes[name] = sha256(target)
    binding = {
        "schema": "reze_mrhisum_v1",
        "created_at": utc_now(),
        "scope": scope,
        "dataset": "MR.HiSum",
        "n_queries": len(rows),
        "n_videos": len(rows),
        "selected_qids": selected,
        "official_test_count": 2000,
        "media_eligible_count": len(eligible),
        "all_verified_media_selected": args.all_verified and args.shard_count == 1,
        "all_verified_media_scope": args.all_verified,
        "target_videos_across_shards": target_count,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "excluded_status_counts": dict(
            Counter(r["status"] for r in receipts.values() if r["status"] != "downloaded_verified")
        ),
        "selection_rule": (
            f"all downloaded_verified official test videos, ordered by ascending sha256('42:video_id'); shard {args.shard_index} of {args.shard_count} by list stride"
            if args.all_verified
            else "first N ascending sha256('42:video_id') among downloaded_verified official test videos"
        ),
        "selection_used_label_or_metric_values": False,
        "method": "Local QV B0 implementation of REZE Appendix A, transferred to query-free MR.HiSum",
        "query": QUERY,
        "query_source": "Fixed generic query; no title, category, caption or label conditioning",
        "window_seconds": 3,
        "frames_per_full_window": 6,
        "sampling_fps": 2,
        "readout": "Unmodified inclusive integer-second REZE curve; official mean per consecutive 5 seconds",
        "timeline_policy": "Score min(annotation length, decoded end); <=2s difference only; retain published nearest fill",
        "zero_duration_frame_interval_repair": "Preserve start PTS and sampling rule; infer missing frame ends from next PTS, final end from FPS",
        "duplicate_start_pts_policy": "Preserve repeated starts and original right-sided search; reject decreasing PTS",
        "primary_metrics": ["mAP50", "mAP15"],
        "secondary_metrics": ["summary_F1"],
        "evaluator_revision": REVISION,
        "evaluation_policy": "Pinned author code: CPU float32, 5s means, top floor(50%)/floor(15%) GT; exp normalization and sklearn AP",
        "summary_policy": "Pinned KTS/15%-knapsack and author nfps=end-start with existing shared endpoints; no boundary repair",
        "baseline_controls": "constant-score AP and exact expected AP of uniformly random untied rankings; constant-score summary F1",
        "labels_sha256": labels_hashes,
        "batch_size": 16,
        "expected_model_requests": windows,
        "smoke_video_ids": selected[:3],
        "smoke_windows": "first/middle/last windows in first three selected videos",
        "maximum_charged_requests": windows + limits["request_overhead"],
        "maximum_worker_attempts": limits["maximum_worker_attempts"],
        "maximum_active_worker_seconds": limits["maximum_active_worker_seconds"],
        "maximum_worker_stall_seconds": 900,
        "max_gpus": 1,
        "require_no_other_compute_process": True,
        "gpu_candidates": args.gpu_uuids,
        "gpu_addressing": "numeric NVML index for vLLM 0.12, verify actual torch UUID before engine load",
        "model_path": audit["model_path"],
        "model_revision": audit["revision"],
        "model_verification_sha256": sha256(run / "model_verification.json"),
        "media_verification_sha256": sha256(run / "media_verification.json"),
        "sampling_audit_sha256": sha256(run / "sampling_audit.json"),
        "inference_sha256": sha256(run / "inference.jsonl"),
        "source_sha256": source_hashes,
        "baseline_code_sha256": {
            name: sha256(QV / "scripts" / name) for name in ("reze_core.py", "reze_backend.py", "reze_media.py")
        },
        "inference_uses_labels": False,
        "training": False,
        "paper_level_MRHiSum_REZE_reproduction": False,
    }
    atomic_json(run / "binding.json", binding)
    atomic_json(
        run / "runtime.json",
        {
            "frozen_at": utc_now(),
            "python": sys.version,
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("torch", "transformers", "vllm", "decord", "numpy", "scipy", "scikit-learn", "h5py")
            },
            "summary_metric_guard": "Apply the same zero-padding used by author F1 before calling its unused correlation branch.",
            "environment_changed": False,
        },
    )
    atomic_json(
        run / "status.json",
        {
            "status": "prepared",
            "updated_at": utc_now(),
            "scope": binding["scope"],
            "completed_queries": 0,
            "expected_queries": len(rows),
            "planned_requests": windows,
        },
    )
    print(
        json.dumps(
            {
                "run": str(run),
                "videos": len(rows),
                "planned_requests": windows,
                "annotated_seconds": sum(r["duration"] for r in rows),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
