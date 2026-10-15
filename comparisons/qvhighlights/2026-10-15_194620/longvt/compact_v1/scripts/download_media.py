#!/usr/bin/env python3
"""Download exact QVHighlights YouTube segments and freeze available manifests."""

import argparse
import concurrent.futures
import json
import subprocess
from pathlib import Path


def probe(path: Path) -> dict:
    command = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration,start_time,size",
        "-of", "json", str(path),
    ]
    data = json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)
    return data["format"]


def download_one(row: dict, media_dir: Path, python: str, max_height: int) -> dict:
    target = media_dir / f'{row["vid"]}.mp4'
    record = {"candidate_rank": row["candidate_rank"], "qid": row["qid"], "vid": row["vid"], "path": str(target)}
    try:
        if not target.exists():
            command = [
                python, "-m", "yt_dlp", "--no-playlist",
                "--download-sections", f'*{row["source_start_seconds"]}-{row["source_end_seconds"]}',
                "--force-keyframes-at-cuts",
                "-f", (
                    f"bestvideo[height<={max_height}][ext=mp4]/"
                    f"bestvideo[height<={max_height}]/best[height<={max_height}]"
                ),
                "-o", str(media_dir / f'{row["vid"]}.%(ext)s'),
                f'https://www.youtube.com/watch?v={row["youtube_id"]}',
            ]
            completed = subprocess.run(command, text=True, capture_output=True, timeout=1200)
            record["download_stdout_tail"] = completed.stdout[-2000:]
            record["download_stderr_tail"] = completed.stderr[-4000:]
            if completed.returncode != 0:
                raise RuntimeError(f"yt-dlp exit {completed.returncode}")
        info = probe(target)
        actual_duration = float(info["duration"])
        record.update({"status": "available", "actual_duration": actual_duration, "size": int(info["size"])})
        if abs(actual_duration - row["duration"]) > 1.1:
            record.update({"status": "duration_mismatch", "expected_duration": row["duration"]})
    except Exception as exc:
        record.update({"status": "unavailable", "error": repr(exc)})
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--media-dir", required=True)
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--python", required=True)
    args = parser.parse_args()
    candidates = json.loads(Path(args.candidates).read_text())
    config = json.loads(Path(args.config).read_text())
    media_dir = Path(args.media_dir)
    media_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = Path(args.manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    needed = config["smoke_size"] + config["pilot_size"]

    records = []
    cursor = 0
    # Download in bounded batches so unavailable candidates are replaced by the
    # next seeded candidate without selecting on labels or model behavior.
    while sum(x["status"] == "available" for x in records) < needed and cursor < len(candidates):
        batch = candidates[cursor:cursor + config["download_workers"]]
        cursor += len(batch)
        with concurrent.futures.ThreadPoolExecutor(max_workers=config["download_workers"]) as executor:
            futures = [
                executor.submit(download_one, row, media_dir, args.python, config["media_max_height"])
                for row in batch
            ]
            for future in futures:
                record = future.result()
                records.append(record)
                print(json.dumps({
                    key: record.get(key)
                    for key in ("candidate_rank", "vid", "status", "actual_duration", "error")
                }), flush=True)
        (manifest_dir / "media_checks.json").write_text(json.dumps(records, indent=2) + "\n")

    available_ranks = {x["candidate_rank"]: x for x in records if x["status"] == "available"}
    available = []
    for row in candidates:
        if row["candidate_rank"] in available_ranks:
            item = {**row, "video_path": available_ranks[row["candidate_rank"]]["path"],
                    "actual_duration": available_ranks[row["candidate_rank"]]["actual_duration"]}
            available.append(item)
        if len(available) == needed:
            break
    if len(available) < needed:
        raise RuntimeError(f"Only {len(available)} of {needed} required media segments are available")
    smoke = available[:config["smoke_size"]]
    pilot = available[config["smoke_size"]:needed]
    (manifest_dir / "media_checks.json").write_text(json.dumps(records, indent=2) + "\n")
    (manifest_dir / "smoke.json").write_text(json.dumps(smoke, indent=2) + "\n")
    (manifest_dir / "pilot.json").write_text(json.dumps(pilot, indent=2) + "\n")
    overlap = {x["vid"] for x in smoke} & {x["vid"] for x in pilot}
    print(json.dumps({"smoke": len(smoke), "pilot": len(pilot), "video_overlap": bool(overlap)}))


if __name__ == "__main__":
    main()
