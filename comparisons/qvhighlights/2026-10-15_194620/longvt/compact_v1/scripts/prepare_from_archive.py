#!/usr/bin/env python3
"""Extract and validate only seeded QVHighlights candidates from the official archive."""

import argparse
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

import decord


def probe(path: Path) -> dict:
    command = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration,start_time,size",
        "-of", "json", str(path),
    ]
    return json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)["format"]


def decode_check(path: Path) -> dict:
    reader = decord.VideoReader(str(path), num_threads=1)
    if len(reader) < 1:
        raise ValueError("video_has_no_frames")
    indices = sorted({0, len(reader) // 2, len(reader) - 1})
    frames = reader.get_batch(indices).asnumpy()
    return {
        "frame_count": len(reader),
        "fps": float(reader.get_avg_fps()),
        "decoded_indices": indices,
        "decoded_shapes": [list(frame.shape) for frame in frames],
    }


def extract_members(archive: Path, wanted: set[str], media_dir: Path) -> dict[str, dict]:
    media_dir.mkdir(parents=True, exist_ok=True)
    extracted: dict[str, dict] = {}
    with tarfile.open(archive, "r|gz") as handle:
        for member in handle:
            if not member.isfile():
                continue
            vid = Path(member.name).stem
            if vid not in wanted or vid in extracted:
                continue
            suffix = Path(member.name).suffix.lower() or ".mp4"
            target = media_dir / f"{vid}{suffix}"
            if not target.exists():
                source = handle.extractfile(member)
                if source is None:
                    continue
                temporary = target.with_suffix(target.suffix + ".partial")
                with source, temporary.open("wb") as output:
                    shutil.copyfileobj(source, output, length=16 * 1024 * 1024)
                os.replace(temporary, target)
            extracted[vid] = {"path": target, "archive_member": member.name}
            if len(extracted) == len(wanted):
                break
    return extracted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--media-dir", required=True)
    parser.add_argument("--manifest-dir", required=True)
    args = parser.parse_args()

    archive = Path(args.archive).resolve()
    candidates = json.loads(Path(args.candidates).read_text())
    config = json.loads(Path(args.config).read_text())
    media_dir = Path(args.media_dir).resolve()
    manifest_dir = Path(args.manifest_dir).resolve()
    media_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    needed = config["smoke_size"] + config["pilot_size"]
    extract_limit = needed + config["archive_extract_reserve"]
    extract_rows = candidates[:extract_limit]
    wanted = {row["vid"] for row in extract_rows}
    print(json.dumps({
        "stage": "extracting_single_pass", "archive": str(archive), "selected": len(wanted),
    }), flush=True)
    extracted = extract_members(archive, wanted, media_dir)

    checks = []
    valid_by_rank = {}
    selected_ranks = {row["candidate_rank"] for row in extract_rows}
    for row in candidates:
        extracted_record = extracted.get(row["vid"])
        record = {
            "candidate_rank": row["candidate_rank"], "qid": row["qid"], "vid": row["vid"],
            "archive_member": extracted_record["archive_member"] if extracted_record else None,
        }
        if row["candidate_rank"] not in selected_ranks:
            record["status"] = "not_selected_for_extraction"
        elif extracted_record is None:
            record["status"] = "missing_from_archive"
        else:
            path = extracted_record["path"]
            record["path"] = str(path)
            try:
                info = probe(path)
                decode = decode_check(path)
                actual_duration = float(info["duration"])
                record.update({
                    "status": "available", "actual_duration": actual_duration,
                    "size": int(info["size"]), "decode_check": decode,
                })
                if abs(actual_duration - float(row["duration"])) > config["duration_tolerance_seconds"]:
                    record.update({"status": "duration_mismatch", "expected_duration": row["duration"]})
                else:
                    valid_by_rank[row["candidate_rank"]] = record
            except Exception as exc:
                record.update({"status": "decode_failure", "error": repr(exc)})
        checks.append(record)
        if record["status"] != "not_selected_for_extraction":
            print(json.dumps({
                key: record.get(key)
                for key in ("candidate_rank", "vid", "status", "actual_duration")
            }), flush=True)

    available = []
    for row in candidates:
        record = valid_by_rank.get(row["candidate_rank"])
        if record is not None:
            available.append({
                **row,
                "video_path": record["path"],
                "actual_duration": record["actual_duration"],
                "media_timeline": "local_benchmark_clip",
                "media_source": "official_qvhilights_videos_archive",
            })
        if len(available) == needed:
            break

    (manifest_dir / "media_checks.json").write_text(json.dumps({
        "archive": str(archive), "archive_size_bytes": archive.stat().st_size,
        "selection_rule": "seeded candidate order; earliest archive-present, duration-valid, decodable media",
        "records": checks,
    }, indent=2) + "\n")
    if len(available) < needed:
        raise RuntimeError(
            f"Only {len(available)} of {needed} media files validated; increase archive_extract_reserve and rerun"
        )
    smoke = available[:config["smoke_size"]]
    pilot = available[config["smoke_size"]:needed]
    (manifest_dir / "smoke.json").write_text(json.dumps(smoke, indent=2) + "\n")
    (manifest_dir / "pilot.json").write_text(json.dumps(pilot, indent=2) + "\n")
    print(json.dumps({
        "smoke": len(smoke), "pilot": len(pilot),
        "video_overlap": bool({x["vid"] for x in smoke} & {x["vid"] for x in pilot}),
    }))


if __name__ == "__main__":
    main()
