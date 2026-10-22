"""Download a frozen MR.HiSum split with resumable, bounded public-source access."""

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from download_qfvs_videos import digest, now, write_json

FORMAT = (
    "bv[ext=mp4][vcodec^=avc1][height<=480]+ba[ext=m4a]/"
    "b[ext=mp4][vcodec^=avc1][height<=480]/bv[ext=mp4][vcodec^=avc1][height<=480]"
)


def classify_failure(message):
    lower = message.lower()
    if any(
        text in lower
        for text in (
            "private video",
            "video has been removed",
            "video unavailable",
            "this video is unavailable",
            "this video is not available",
            "not available in your country",
        )
    ):
        return "source_unavailable"
    if "unavailable with a vpn or proxy" in lower:
        return "network_restricted"
    if any(text in lower for text in ("sign in", "confirm your age", "members-only")):
        return "authentication_required"
    if "decorderror" in lower or "decoded shape differs" in lower:
        return "media_decode_failed"
    return "source_or_download_error"


def validate_video(root, item):
    import decord

    video_id = item["video_id"]
    path = root / "videos" / f"{video_id}.mp4"
    info = json.loads(path.with_suffix(".info.json").read_text())
    if info["id"] != item["youtube_id"]:
        raise ValueError("Downloaded YouTube ID differs from frozen metadata")
    probe = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration,size:stream=codec_type,codec_name,width,height,r_frame_rate,nb_frames,duration",
                "-of",
                "json",
                str(path),
            ],
            timeout=30,
        )
    )
    stream = next(s for s in probe["streams"] if s["codec_type"] == "video")
    duration = float(stream.get("duration", probe["format"]["duration"]))
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Invalid video duration")
    reader = decord.VideoReader(str(path), ctx=decord.cpu(0), num_threads=1)
    samples = []
    for index in (0, len(reader) // 2, len(reader) - 1):
        frame = reader[index].asnumpy()
        if frame.shape != (stream["height"], stream["width"], 3):
            raise ValueError("Decoded shape differs from video stream")
        samples.append({"frame": index, "rgb_sha256": hashlib.sha256(frame.tobytes()).hexdigest()})
    difference = duration - item["annotated_seconds"]
    return {
        "status": "downloaded_verified" if abs(difference) <= 2 else "downloaded_timeline_review",
        "verified_at": now(),
        "video_id": video_id,
        "youtube_id": item["youtube_id"],
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": digest(path),
        "duration_seconds": duration,
        "annotated_seconds": item["annotated_seconds"],
        "duration_difference_seconds": difference,
        "duration_check_tolerance_seconds": 2,
        "decode_samples": samples,
        "ffprobe": probe,
        "has_audio": any(s["codec_type"] == "audio" for s in probe["streams"]),
        "format_id": info.get("format_id"),
        "frames": len(reader),
        "decode_scope": "First, middle and last frame; not full-video pixel decoding",
        "timeline_or_semantic_alignment_proven": False,
        "transcoded_or_trimmed": False,
    }


def process_one(root, item, config):
    video_id = item["video_id"]
    receipt_path = root / "receipts" / f"{video_id}.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if not receipt["status"].startswith("downloaded_"):
            return receipt
        path = Path(receipt["path"])
        if not path.exists() or path.stat().st_size != receipt["bytes"]:
            raise ValueError(f"Previously verified media changed: {video_id}")
        return receipt
    ledger_path = root / "attempts" / f"{video_id}.json"
    attempts = json.loads(ledger_path.read_text()) if ledger_path.exists() else []
    video_path = root / "videos" / f"{video_id}.mp4"
    error = "Attempt budget exhausted"
    category = "source_or_download_error"
    # Recover a completed media download if interrupted before writing its receipt.
    if video_path.exists() and video_path.with_suffix(".info.json").exists():
        receipt = validate_video(root, item)
        write_json(receipt_path, receipt)
        return receipt
    while len(attempts) < config["max_attempts_per_video"]:
        attempt = {"started_at": now(), "status": "started", "attempt": len(attempts) + 1}
        attempts.append(attempt)
        write_json(ledger_path, attempts)
        log_path = root / "logs" / f"{video_id}_attempt{len(attempts)}.log"
        command = [
            sys.executable,
            "-m",
            "yt_dlp",
            "--ignore-config",
            "--js-runtimes",
            f"node:{config['node_path']}",
            "--no-playlist",
            "--no-progress",
            "--socket-timeout",
            "20",
            "--retries",
            "1",
            "--fragment-retries",
            "1",
            "--extractor-retries",
            "1",
            "--abort-on-unavailable-fragments",
            "--max-filesize",
            "1G",
            "--write-info-json",
            "--merge-output-format",
            "mp4",
            "-f",
            FORMAT,
            "-o",
            str(root / "videos" / f"{video_id}.%(ext)s"),
            f"https://www.youtube.com/watch?v={item['youtube_id']}",
        ]
        if config.get("sleep_requests_seconds"):
            command[3:3] = ["--sleep-requests", str(config["sleep_requests_seconds"])]
        then = time.monotonic()
        try:
            with log_path.open("w") as log:
                result = subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=config["timeout_per_attempt_seconds"],
                    check=False,
                )
            attempt["returncode"] = result.returncode
            if result.returncode != 0:
                raise RuntimeError(log_path.read_text(errors="replace")[-3000:])
            receipt = validate_video(root, item)
            attempt.update(status="complete", finished_at=now(), seconds=time.monotonic() - then)
            write_json(ledger_path, attempts)
            write_json(receipt_path, receipt)
            return receipt
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            category = classify_failure(error)
            attempt.update(
                status="failed",
                category=category,
                error=error,
                finished_at=now(),
                seconds=time.monotonic() - then,
                log=str(log_path),
            )
            write_json(ledger_path, attempts)
            if category != "source_or_download_error":
                break
            if len(attempts) < config["max_attempts_per_video"]:
                time.sleep(3)
    receipt = {
        "status": category,
        "video_id": video_id,
        "youtube_id": item["youtube_id"],
        "finished_at": now(),
        "attempts": len(attempts),
        "error": error,
        "annotated_seconds": item["annotated_seconds"],
    }
    write_json(receipt_path, receipt)
    return receipt


def storage_check(root, config):
    free = shutil.disk_usage(root).free
    used = sum(p.stat().st_size for p in (root / "videos").iterdir() if p.is_file())
    if free < config["minimum_free_gib"] * 1024**3:
        return "minimum_free_space_reached"
    if used >= config["maximum_download_gib"] * 1024**3:
        return "download_size_budget_reached"
    return None


def summarize(root, config, manifest_hash, status, **extra):
    records = [json.loads(p.read_text()) for p in sorted((root / "receipts").glob("*.json"))]
    counts = {}
    for record in records:
        counts[record["status"]] = counts.get(record["status"], 0) + 1
    result = {
        "status": status,
        "updated_at": now(),
        "scope": config["scope"],
        "requested": len(config["videos"]),
        "processed": len(records),
        "counts": counts,
        "downloaded": sum(v for k, v in counts.items() if k.startswith("downloaded_")),
        "video_bytes": sum(r.get("bytes", 0) for r in records),
        "manifest_sha256": manifest_hash,
        **extra,
    }
    write_json(root / "status.json", result)
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--limit", type=int, help="Download only the fixed first N entries for a source check"
    )
    args = parser.parse_args()
    config = json.loads(args.manifest.read_text())
    root = Path(config["root"]).resolve()
    for name in ("receipts", "logs", "videos", "attempts"):
        (root / name).mkdir(parents=True, exist_ok=True)
    lock = (root / "download.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("Another downloader owns this queue; leaving it running.", flush=True)
        return
    ids = [v["video_id"] for v in config["videos"]]
    if len(ids) != len(set(ids)) or any(not re.fullmatch(r"video_\d+", value) for value in ids):
        raise ValueError("Invalid or duplicate dataset IDs")
    if any(not re.fullmatch(r"[\w-]{11}", v["youtube_id"]) for v in config["videos"]):
        raise ValueError("Invalid YouTube ID")
    manifest_hash = digest(args.manifest)
    if (root / "status.json").exists():
        old = json.loads((root / "status.json").read_text())
        if old.get("manifest_sha256") != manifest_hash:
            raise ValueError("Frozen download scope changed")
    items = config["videos"][: args.limit] if args.limit else config["videos"]
    pending = [x for x in items if not (root / "receipts" / f"{x['video_id']}.json").exists()]
    consecutive_failures = 0
    summarize(root, config, manifest_hash, "running")
    with concurrent.futures.ThreadPoolExecutor(max_workers=config["workers"]) as executor:
        for offset in range(0, len(pending), config["workers"]):
            reason = storage_check(root, config)
            if reason:
                result = summarize(root, config, manifest_hash, "stopped", reason=reason)
                write_json(root / "stopped.json", result)
                return
            batch = pending[offset : offset + config["workers"]]
            summarize(root, config, manifest_hash, "running", active=[x["video_id"] for x in batch])
            futures = [executor.submit(process_one, root, item, config) for item in batch]
            for future in concurrent.futures.as_completed(futures):
                receipt = future.result()
                if receipt["status"] == "source_or_download_error":
                    consecutive_failures += 1
                else:
                    consecutive_failures = 0
                summarize(root, config, manifest_hash, "running")
            if consecutive_failures >= config["maximum_consecutive_download_errors"]:
                result = summarize(root, config, manifest_hash, "stopped", reason="repeated_source_errors")
                write_json(root / "stopped.json", result)
                return
            time.sleep(1)
    result = summarize(root, config, manifest_hash, "pilot_finished" if args.limit else "queue_finished")
    if not args.limit:
        write_json(root / "complete.json", result)


if __name__ == "__main__":
    main()
