#!/usr/bin/env python3
"""Build a label-free full-validation manifest from extracted official media."""

import argparse
import json
import subprocess
from pathlib import Path

import decord

VIDEO_SUFFIXES = {".mp4", ".mkv", ".webm", ".avi", ".mov"}


def split_vid(vid: str) -> tuple[str, float, float]:
    youtube_id, start, end = vid.rsplit("_", 2)
    return youtube_id, float(start), float(end)


def probe_duration(path: Path) -> float:
    command = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path),
    ]
    data = json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)
    return float(data["format"]["duration"])


def decode_check(path: Path) -> dict:
    reader = decord.VideoReader(str(path), num_threads=1)
    if not len(reader):
        raise ValueError("video_has_no_frames")
    indices = sorted({0, len(reader) // 2, len(reader) - 1})
    shapes = [list(frame.shape) for frame in reader.get_batch(indices).asnumpy()]
    return {"frame_count": len(reader), "decoded_indices": indices, "decoded_shapes": shapes}


def index_media(media_dir: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for path in media_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
            index.setdefault(path.stem, []).append(path.resolve())
    return index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--media-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--media-checks", required=True)
    parser.add_argument("--decode-check", action="store_true")
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.annotations).read_text().splitlines() if line.strip()]
    config = json.loads(Path(args.config).read_text())
    media = index_media(Path(args.media_dir))
    checks = []
    valid_paths = {}
    expected_by_vid = {}
    for row in rows:
        expected_by_vid.setdefault(row["vid"], float(row["duration"]))
    for vid, expected_duration in expected_by_vid.items():
        matches = media.get(vid, [])
        record = {"vid": vid, "expected_duration": expected_duration, "matches": [str(x) for x in matches]}
        if not matches:
            record["status"] = "missing"
        elif len(matches) > 1:
            record["status"] = "duplicate_media_paths"
        else:
            try:
                actual_duration = probe_duration(matches[0])
                record["actual_duration"] = actual_duration
                if abs(actual_duration - expected_duration) > config["duration_tolerance_seconds"]:
                    record["status"] = "duration_mismatch"
                else:
                    if args.decode_check:
                        record["decode_check"] = decode_check(matches[0])
                    record["status"] = "available"
                    valid_paths[vid] = matches[0]
            except Exception as exc:
                record.update({"status": "decode_failure", "error": repr(exc)})
        checks.append(record)

    manifest = []
    for row in rows:
        path = valid_paths.get(row["vid"])
        if path is None:
            continue
        youtube_id, source_start, source_end = split_vid(row["vid"])
        duration = float(row["duration"])
        manifest.append({
            "qid": row["qid"], "vid": row["vid"], "query": row["query"],
            "duration": duration, "num_clips": int(duration / config["clip_seconds"]),
            "youtube_id": youtube_id, "source_start_seconds": source_start,
            "source_end_seconds": source_end, "video_path": str(path),
            "actual_duration": next(x["actual_duration"] for x in checks if x["vid"] == row["vid"]),
            "media_timeline": "local_benchmark_clip",
            "media_source": "official_qvhilights_videos_archive",
        })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    checks_path = Path(args.media_checks)
    checks_path.parent.mkdir(parents=True, exist_ok=True)
    checks_path.write_text(json.dumps(checks, indent=2) + "\n")
    statuses = sorted({row["status"] for row in checks})
    print(json.dumps({
        "queries": len(manifest), "expected_queries": len(rows),
        "unique_videos": len(valid_paths), "expected_unique_videos": len(expected_by_vid),
        "status_counts": {status: sum(row["status"] == status for row in checks) for status in statuses},
    }))


if __name__ == "__main__":
    main()
