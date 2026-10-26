"""Register a fixed, small paired SVHighlights transfer evaluation without labels."""

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path

from shared import HERE, atomic_json, read_jsonl, segments_for, sha256, utc_now

QV = Path(os.environ.get("ZEROSHOT_ROOT", "baselines/zeroshot"))
OLD = Path(os.environ.get("SVH_PAIRED_RUN", "data/SVHighlights/paired_videoarm_run"))
META = Path(os.environ.get("SVH_INFERENCE_MANIFEST", "data/SVHighlights/inference_manifest.jsonl"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--previous-run", type=Path, help="Preserve the failed predecessor and its consumed budget")
    args = parser.parse_args()
    if not os.environ.get("REPRO_GPU_UUIDS", "").strip():
        parser.error("Set REPRO_GPU_UUIDS to the allocated NVIDIA GPU UUIDs")
    run = args.run
    if (run / "binding.json").exists():
        raise SystemExit("Run already frozen; use a different directory")
    run.mkdir(parents=True, exist_ok=True)
    meta = read_jsonl(META)
    # Use prediction identities, never metric values, to define the paired pool.
    previous = {}
    for name in ["uniform_refine_av", "videoarm_adaptive_av"]:
        path = OLD / "scoring/dev" / f"{name}_predictions.json"
        previous[name] = {x["vid"] for x in json.loads(path.read_text())}
    common = set.intersection(*previous.values())
    pool = [r for r in meta if r["split"] == "dev" and r["vid"] in common]
    selected = []
    for sport in sorted({r["sport"] for r in pool}):
        candidates = [r for r in pool if r["sport"] == sport]
        selected.append(min(candidates, key=lambda r: hashlib.sha256(f"42:{r['vid']}".encode()).hexdigest()))
    if len(selected) != 7 or len(pool) != 20:
        raise ValueError("Historical paired development pool changed")
    inventory = Counter()
    for row in meta:
        count_path = OLD / "counts" / f"{row['vid']}.json"
        count = json.loads(count_path.read_text()) if count_path.exists() else {}
        inventory[count.get("status", "missing")] += 1
    rows, receipts = [], {}
    for row in selected:
        vid = row["vid"]
        media = OLD / "media" / f"{vid}.mp4"
        receipt = json.loads((OLD / "media/receipts" / f"{vid}.json").read_text())
        count = json.loads((OLD / "counts" / f"{vid}.json").read_text())
        if receipt["status"] != "complete" or count["status"] != "matched":
            raise ValueError(f"Media not accepted: {vid}")
        if receipt["source_video_id"] != row["source_video_id"]:
            raise ValueError("Wrong source video")
        for key, expected in [
            ("source_trim_start_seconds", row["source_trim_start_sec"]),
            ("source_trim_end_seconds", row["source_trim_end_sec"]),
        ]:
            if abs(receipt[key] - expected) > 1e-6:
                raise ValueError("Wrong source trim")
        duration = receipt["duration_seconds"]
        if len(range(0, int(duration), 2)) != count["label_clip_count"]:
            raise ValueError("Media and released clip counts differ")
        before = media.stat()
        digest = sha256(media)
        if digest != receipt["media_sha256"] or digest != count["media_sha256"]:
            raise ValueError(f"Video hash mismatch: {vid}")
        after = media.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("Media mutated during verification")
        rows.append(
            {
                "vid": vid,
                "sport": row["sport"],
                "query": f"Highlight of this {row['sport']} video.",
                "duration": duration,
                "video_path": str(media),
                "media_sha256": digest,
                "media_size": after.st_size,
                "media_mtime_ns": after.st_mtime_ns,
                "n_clips": count["label_clip_count"],
                "source_video_id": row["source_video_id"],
            }
        )
        receipts[vid] = {"receipt": receipt, "counts": count, "checked_at": utc_now()}
        print(json.dumps({"verified": vid, "seconds": duration}), flush=True)
    # Model weights and tokenizer are the same frozen snapshot as QV B0.
    audit = json.loads((QV / "manifests/reze_model_audit.json").read_text())
    model_stats = {}
    for name, expected in audit["files"].items():
        path = Path(audit["model_path"]) / name
        if path.stat().st_size != expected["size"] or sha256(path) != expected["sha256"]:
            raise ValueError(f"Model hash mismatch: {name}")
        model_stats[name] = {"size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
    (run / "inference.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    atomic_json(run / "media_verification.json", receipts)
    atomic_json(run / "model_verification.json", {"audit": audit, "stats": model_stats, "checked_at": utc_now()})
    source_hashes = {}
    for path in sorted(HERE.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or ".pytest_cache" in path.parts:
            continue
        if path.suffix not in {".py", ".md", ".LICENSE"}:
            continue
        name = str(path.relative_to(HERE))
        target = run / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        source_hashes[name] = sha256(path)
    calls = sum(len(segments_for(r["duration"])) for r in rows)
    inherited = {"requests": 0, "active_worker_seconds": 0}
    if args.previous_run:
        parent = args.previous_run.resolve()
        old = json.loads((parent / "binding.json").read_text())
        if old["selected_ids"] != [r["vid"] for r in rows] or not (parent / "stopped.json").exists():
            raise ValueError("Recovery must retain the stopped seven-video predecessor")
        spent = json.loads((parent / "worker_attempts.json").read_text())
        inherited = {"run": str(parent), "binding_sha256": sha256(parent / "binding.json"),
                     "attempts_sha256": sha256(parent / "worker_attempts.json"),
                     "active_worker_seconds": sum(a["seconds"] for a in spent),
                     "requests": sum(json.loads(p.read_text())["requests"]
                                     for p in (parent / "request_attempts").glob("*.json"))}
    binding = {
        "schema": "reze_original_svh_pilot7_recovery_v2",
        "created_at": utc_now(),
        "method": "REZE Appendix A independently implemented, unchanged QV B0 scorer",
        "dataset": "SVHighlights",
        "scope": "seven-video paired development pilot, not official all320",
        "selection_rule": "intersection of existing dev prediction IDs; lowest sha256('42:vid') per available sport",
        "selection_read_label_values": False,
        "selection_read_metric_values": False,
        "historical_paired_pool": sorted(common),
        "selected_ids": [r["vid"] for r in rows],
        "dataset_total": 320,
        "media_inventory": dict(inventory),
        "missing_sport": "basketball",
        "source_content_hours": sum(r["duration"] for r in rows) / 3600,
        "window_seconds": 3,
        "full_window_frames": 6,
        "sampling_fps": 2,
        "query_template": "Highlight of this {sport} video.",
        "query_source": "SVHighlights/tf_selector/parse.py: format_vhd_output",
        "readout": "REZE inclusive-second curve, mean each available 2-second bin; retain final single-second bin",
        "evaluator": "unmodified official metric functions; strict IDs and lengths checked before calls",
        "batch_size": 16,
        "expected_model_requests_upper_bound": calls,
        "maximum_charged_requests": calls + 3 * 16 - inherited["requests"],
        "maximum_worker_attempts": 3,
        "maximum_active_worker_seconds": 12 * 3600 - inherited["active_worker_seconds"],
        "maximum_worker_stall_seconds": 900,
        "require_gpu_smoke": True,
        "engineering_requests": 21,
        "inherited_budget": inherited,
        "gpu_candidates": [uuid.strip() for uuid in os.environ.get("REPRO_GPU_UUIDS", "").split(",") if uuid.strip()],
        "max_gpus": 1,
        "require_no_other_compute_process": True,
        "model_revision": audit["revision"],
        "model_path": audit["model_path"],
        "model_verification_sha256": sha256(run / "model_verification.json"),
        "media_verification_sha256": sha256(run / "media_verification.json"),
        "inference_sha256": sha256(run / "inference.jsonl"),
        "source_sha256": source_hashes,
        "previous_predictions": {
            name: {
                "path": str(OLD / "scoring/dev" / f"{name}_predictions.json"),
                "sha256": sha256(OLD / "scoring/dev" / f"{name}_predictions.json"),
            }
            for name in previous
        },
        "labels_path_for_separate_evaluation": (
            os.environ.get("REPRO_DATA_ROOT", "data") + "/SVHighlights/annotations/label.json"
        ),
        "labels_sha256": "dea5697c5a736a9c7c2a926528ade204d6c75a56f1b9f8862808a76d763992f3",
        "official_evaluator_revision": "a463fafbca8136bc6839bddc91216e494d970833",
    }
    atomic_json(run / "binding.json", binding)
    atomic_json(
        run / "status.json",
        {
            "status": "prepared_waiting_for_gpu",
            "updated_at": utc_now(),
            "completed_videos": 0,
            "expected_videos": 7,
            "planned_requests": calls,
        },
    )
    print(json.dumps({"run": str(run), "planned_requests": calls, "hours": binding["source_content_hours"]}))


if __name__ == "__main__":
    main()
