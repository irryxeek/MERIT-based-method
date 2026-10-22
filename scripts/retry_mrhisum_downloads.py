"""Retry recoverable MR.HiSum failures while preserving the completed first pass."""

import argparse
import fcntl
import json
import shutil
import time
from collections import Counter
from pathlib import Path

from download_mrhisum_videos import process_one, storage_check, summarize
from download_qfvs_videos import digest, now, write_json


def retry_reason(record):
    if record["status"].startswith("downloaded_"):
        return None
    error = record.get("error", "").lower()
    if any(
        text in error
        for text in ("private video", "video has been removed", "confirm your age", "members-only")
    ):
        return None
    if "timeoutexpired" in error:
        return "timeout"
    if "not a bot" in error:
        return "temporary_authentication"
    if "unavailable with a vpn or proxy" in error:
        return "proxy_restriction"
    return "ambiguous_source_failure"


def select_items(config, records):
    priorities = {
        "timeout": 0,
        "temporary_authentication": 1,
        "proxy_restriction": 2,
        "ambiguous_source_failure": 3,
    }
    selected, skipped = [], []
    for item in config["videos"]:
        record = records[item["video_id"]]
        reason = retry_reason(record)
        if reason:
            selected.append(
                {
                    **item,
                    "retry_reason": reason,
                    "previous_attempts": record.get("attempts", 0),
                    "previous_status": record["status"],
                }
            )
        elif not record["status"].startswith("downloaded_"):
            skipped.append(
                {
                    "video_id": item["video_id"],
                    "status": record["status"],
                    "reason": "explicit_private_removed_or_restricted_access",
                }
            )
    # Keep the original fixed manifest order within each error category.
    selected.sort(key=lambda item: priorities[item["retry_reason"]])
    return selected, skipped


def prepare(root, baseline):
    if root.exists():
        raise FileExistsError(f"Use a new retry directory: {root}")
    config = json.loads((baseline / "download_manifest.json").read_text())
    complete = json.loads((baseline / "complete.json").read_text())
    if complete["status"] != "queue_finished":
        raise ValueError("First pass must finish before preparing retries")
    records = {}
    fingerprints = {}
    for item in config["videos"]:
        path = baseline / "receipts" / f"{item['video_id']}.json"
        records[item["video_id"]] = json.loads(path.read_text())
        fingerprints[item["video_id"]] = digest(path)
    selected, skipped = select_items(config, records)
    if not selected:
        raise ValueError("No recoverable failures to retry")
    root.mkdir(parents=True)
    for folder in ("receipts", "attempts", "logs", "code"):
        (root / folder).mkdir()
    (root / "videos").symlink_to(baseline / "videos", target_is_directory=True)
    for filename in ("retry_mrhisum_downloads.py", "download_mrhisum_videos.py", "download_qfvs_videos.py"):
        shutil.copy2(Path(__file__).parent / filename, root / "code" / filename)
    write_json(root / "baseline_receipts.json", records)
    write_json(root / "baseline_receipt_hashes.json", fingerprints)
    write_json(root / "baseline_complete.json", complete)
    retry_config = {
        **config,
        "root": str(root),
        "baseline_root": str(baseline),
        "created_at": now(),
        "scope": "official_test_failures_retry_v1",
        "videos": selected,
        "skipped_failures": skipped,
        "order": "error priority, then original fixed order; no label-based selection",
        "workers": 1,
        "max_attempts_per_video": 1,
        "timeout_per_attempt_seconds": 600,
        "sleep_requests_seconds": 1,
        "between_videos_seconds": 3,
        "baseline_manifest_sha256": digest(baseline / "download_manifest.json"),
        "baseline_complete_sha256": digest(baseline / "complete.json"),
        "baseline_receipts_sha256": digest(root / "baseline_receipts.json"),
        "baseline_receipt_hashes_sha256": digest(root / "baseline_receipt_hashes.json"),
        "code_sha256": {p.name: digest(p) for p in sorted((root / "code").glob("*.py"))},
        "new_attempt_budget": len(selected),
        "network_note": "Retain existing proxy; direct YouTube connection timed out. No account cookies.",
    }
    write_json(root / "download_manifest.json", retry_config)
    report(root, retry_config, digest(root / "download_manifest.json"), "prepared")


def validate_scope(root, config):
    baseline = Path(config["baseline_root"])
    for name, key in (
        ("download_manifest.json", "baseline_manifest_sha256"),
        ("complete.json", "baseline_complete_sha256"),
    ):
        if digest(baseline / name) != config[key]:
            raise ValueError(f"First-pass {name} changed")
    for name, key in (
        ("baseline_receipts.json", "baseline_receipts_sha256"),
        ("baseline_receipt_hashes.json", "baseline_receipt_hashes_sha256"),
    ):
        if digest(root / name) != config[key]:
            raise ValueError(f"Retry snapshot {name} changed")
    for name, expected in config["code_sha256"].items():
        if digest(root / "code" / name) != expected:
            raise ValueError(f"Frozen retry code changed: {name}")
    fingerprints = json.loads((root / "baseline_receipt_hashes.json").read_text())
    for video_id, expected in fingerprints.items():
        if digest(baseline / "receipts" / f"{video_id}.json") != expected:
            raise ValueError(f"First-pass receipt changed: {video_id}")
    records = json.loads((root / "baseline_receipts.json").read_text())
    original = json.loads((baseline / "download_manifest.json").read_text())
    expected, skipped = select_items(original, records)
    if config["videos"] != expected or config["skipped_failures"] != skipped:
        raise ValueError("Retry selection differs from original failure records")
    if (root / "videos").resolve() != (baseline / "videos").resolve():
        raise ValueError("Retry media directory differs from original media directory")
    for record in records.values():
        if record["status"].startswith("downloaded_"):
            path = Path(record["path"])
            if not path.is_file() or path.stat().st_size != record["bytes"]:
                raise ValueError(f"Previously downloaded media changed: {record['video_id']}")


def merge_records(baseline, retries, selected_ids):
    effective = dict(baseline)
    for record in retries:
        video_id = record["video_id"]
        if video_id not in selected_ids or video_id not in baseline:
            raise ValueError(f"Out-of-scope retry receipt: {video_id}")
        if baseline[video_id]["status"].startswith("downloaded_"):
            raise ValueError(f"Retry would replace a previously successful video: {video_id}")
        if record["youtube_id"] != baseline[video_id]["youtube_id"]:
            raise ValueError(f"Retry YouTube ID changed: {video_id}")
        effective[video_id] = record
    return effective


def report(root, config, manifest_hash, state, **extra):
    summary = summarize(root, config, manifest_hash, state, **extra)
    baseline = json.loads((root / "baseline_receipts.json").read_text())
    retries = [json.loads(p.read_text()) for p in sorted((root / "receipts").glob("*.json"))]
    effective = merge_records(baseline, retries, {v["video_id"] for v in config["videos"]})
    counts = Counter(record["status"] for record in effective.values())
    baseline_downloaded = sum(r["status"].startswith("downloaded_") for r in baseline.values())
    downloaded = sum(n for status, n in counts.items() if status.startswith("downloaded_"))
    available = {
        "updated_at": now(),
        "status": state,
        "scope": "official_test_2000",
        "requested": len(baseline),
        "downloaded": downloaded,
        "missing": len(baseline) - downloaded,
        "counts": dict(counts),
        "baseline_downloaded": baseline_downloaded,
        "newly_downloaded": downloaded - baseline_downloaded,
        "video_bytes": sum(r.get("bytes", 0) for r in effective.values()),
        "retry_requested": len(config["videos"]),
        "retry_processed": len(retries),
        "retry_skipped_explicit_access_failures": len(config["skipped_failures"]),
        "retry_root": str(root),
        "retry_manifest_sha256": manifest_hash,
        "baseline_root": config["baseline_root"],
        "baseline_manifest_sha256": config["baseline_manifest_sha256"],
        "media_index": str(root / "effective_receipts.json"),
        "timeline_review_ids": [
            v for v, r in effective.items() if r["status"] == "downloaded_timeline_review"
        ],
        **extra,
    }
    write_json(root / "effective_receipts.json", effective)
    write_json(root / "availability.json", available)
    write_json(Path(config["baseline_root"]).parent / "availability.json", available)
    return summary


def run(root):
    config = json.loads((root / "download_manifest.json").read_text())
    # Hold both locks to prevent shared partial files being written by another queue.
    locks = []
    for directory in (Path(config["baseline_root"]), root):
        handle = (directory / "download.lock").open("a")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            raise RuntimeError("Another downloader owns the media directory") from None
        locks.append(handle)
    validate_scope(root, config)
    manifest_hash = digest(root / "download_manifest.json")
    previous = json.loads((root / "status.json").read_text())
    if previous["manifest_sha256"] != manifest_hash:
        raise ValueError("Frozen retry manifest changed")
    if (root / "complete.json").exists() or (root / "stopped.json").exists():
        print("Retry queue already ended; preserving its final state.", flush=True)
        return
    pending = [v for v in config["videos"] if not (root / "receipts" / f"{v['video_id']}.json").exists()]
    errors = 0
    report(root, config, manifest_hash, "running")
    for item in pending:
        reason = storage_check(root, config)
        if reason:
            result = report(root, config, manifest_hash, "stopped", reason=reason)
            write_json(root / "stopped.json", result)
            return
        report(root, config, manifest_hash, "running", active=[item["video_id"]])
        receipt = process_one(root, item, config)
        errors = errors + 1 if receipt["status"] == "source_or_download_error" else 0
        report(root, config, manifest_hash, "running")
        if errors >= config["maximum_consecutive_download_errors"]:
            result = report(root, config, manifest_hash, "stopped", reason="repeated_download_errors")
            write_json(root / "stopped.json", result)
            return
        time.sleep(config["between_videos_seconds"])
    result = report(root, config, manifest_hash, "queue_finished")
    write_json(root / "complete.json", result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--prepare-from", type=Path, help="Prepare a new retry queue from a completed first pass"
    )
    args = parser.parse_args()
    if args.prepare_from:
        prepare(args.root.resolve(), args.prepare_from.resolve())
    else:
        run(args.root.resolve())


if __name__ == "__main__":
    main()
