"""Download and verify the official UT Egocentric release used by QFVS."""

import argparse
import hashlib
import json
import os
import stat
import subprocess
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import requests

SOURCE_PAGE = "https://vision.cs.utexas.edu/projects/egocentric_data/UT_Egocentric_Dataset.html"
DOWNLOAD_URL = "https://vision.cs.utexas.edu/projects/egocentric/download.php"


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def download(root):
    archive = root / "archives/UTE_video.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists():
        with zipfile.ZipFile(archive) as zipped:
            if not zipped.infolist():
                raise ValueError("Existing archive is empty")
        return archive
    part = archive.with_suffix(".zip.part")
    attempts_path = root / "download_attempts.json"
    attempts = json.loads(attempts_path.read_text()) if attempts_path.exists() else []
    for _ in range(6 - len(attempts)):
        offset = part.stat().st_size if part.exists() else 0
        attempt = {"started_at": now(), "previous_partial_bytes": offset, "status": "started"}
        attempts.append(attempt)
        write_json(attempts_path, attempts)
        started, last = time.monotonic(), 0.0
        try:
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            with requests.get(DOWNLOAD_URL, headers=headers, stream=True, timeout=(30, 90)) as response:
                response.raise_for_status()
                resumed = response.status_code == 206 and offset > 0
                if resumed and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise ValueError("Server returned the wrong resume range")
                if response.status_code not in (200, 206) or "zip" not in response.headers.get(
                    "Content-Type", ""
                ):
                    raise ValueError("Download did not return a ZIP archive")
                attempt.update(
                    http_status=response.status_code,
                    resumed=resumed,
                    # This PHP endpoint currently ignores Range; never append a fresh 200 response.
                    restarted_from_zero=bool(offset and not resumed),
                    response_headers={
                        k: v
                        for k, v in response.headers.items()
                        if k.lower()
                        in {
                            "content-type",
                            "content-length",
                            "content-range",
                            "content-disposition",
                            "etag",
                            "last-modified",
                        }
                    },
                )
                total = offset if resumed else 0
                with part.open("ab" if resumed else "wb") as stream:
                    for chunk in response.iter_content(4 * 1024 * 1024):
                        if not chunk:
                            continue
                        if total == 0 and not chunk.startswith(b"PK"):
                            raise ValueError("Response is not ZIP data")
                        stream.write(chunk)
                        total += len(chunk)
                        if total > 8 * 1024**3:
                            raise ValueError("Unexpected archive size above 8 GiB")
                        elapsed = time.monotonic() - started
                        if elapsed - last >= 10:
                            last = elapsed
                            status = {
                                "status": "downloading",
                                "updated_at": now(),
                                "bytes": total,
                                "attempt": len(attempts),
                                "bytes_per_second": (total - (offset if resumed else 0)) / elapsed,
                            }
                            write_json(root / "status.json", status)
                            print(json.dumps(status), flush=True)
                    stream.flush()
                    os.fsync(stream.fileno())
            with zipfile.ZipFile(part) as zipped:
                members = zipped.infolist()
                if not members:
                    raise ValueError("Archive contains no files")
            part.replace(archive)
            attempt.update(
                status="complete",
                finished_at=now(),
                bytes=archive.stat().st_size,
                seconds=time.monotonic() - started,
            )
            write_json(attempts_path, attempts)
            return archive
        except Exception as exc:
            attempt.update(status="failed", finished_at=now(), error=f"{type(exc).__name__}: {exc}")
            write_json(attempts_path, attempts)
            write_json(
                root / "status.json",
                {
                    "status": "download_retry_pending",
                    "updated_at": now(),
                    "attempt": len(attempts),
                    "error": attempt["error"],
                },
            )
            print(json.dumps(attempt), flush=True)
            if len(attempts) < 6:
                time.sleep(20)
    raise RuntimeError("Six bounded download attempts exhausted; inspect download_attempts.json")


def extract(root, archive):
    destination = root / "official_release"
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    with zipfile.ZipFile(archive) as zipped:
        infos = zipped.infolist()
        if sum(info.file_size for info in infos) > 100 * 1024**3:
            raise ValueError("Unexpected expanded archive size")
        for index, info in enumerate(infos):
            path = destination / info.filename
            if not path.resolve().is_relative_to(destination.resolve()):
                raise ValueError(f"Archive member escapes destination: {info.filename}")
            if stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError("Archive symlinks are unsupported")
            if info.is_dir():
                path.mkdir(parents=True, exist_ok=True)
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_name(path.name + ".extracting")
            hasher = hashlib.sha256()
            with zipped.open(info) as source, temp.open("wb") as target:
                for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    target.write(chunk)
                    hasher.update(chunk)
            # ZipFile verifies CRC when the member reaches EOF.
            if temp.stat().st_size != info.file_size:
                raise ValueError("Extracted member length mismatch")
            temp.replace(path)
            records.append(
                {
                    "member": info.filename,
                    "path": str(path),
                    "bytes": info.file_size,
                    "zip_crc32": f"{info.CRC:08x}",
                    "sha256": hasher.hexdigest(),
                }
            )
            if index % 1000 == 0 or path.suffix.lower() in {".mp4", ".avi", ".wmv", ".mov", ".mkv"}:
                write_json(
                    root / "status.json",
                    {
                        "status": "extracting",
                        "updated_at": now(),
                        "members_complete": len(records),
                        "members_total": len(infos),
                    },
                )
    return records


def probe_video(path):
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration,size,start_time:stream=index,codec_type,codec_name,width,height,avg_frame_rate,"
        "r_frame_rate,nb_frames,duration,start_time,time_base",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=180)
    return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "complete.json").exists():
        print("Download already complete; preserving existing release.")
        return
    archive = download(root)
    archive_hash = digest(archive)
    write_json(
        root / "archive.json",
        {
            "source_page": SOURCE_PAGE,
            "download_url": DOWNLOAD_URL,
            "path": str(archive),
            "bytes": archive.stat().st_size,
            "sha256": archive_hash,
            "verified_at": now(),
            "published_checksum_available": False,
            "publisher_face_blurring_preserved": True,
        },
    )
    records = extract(root, archive)
    videos = []
    for record in records:
        path = Path(record["path"])
        if path.suffix.lower() in {".mp4", ".avi", ".wmv", ".mov", ".mkv", ".mpg", ".mpeg", ".webm"}:
            videos.append({**record, "ffprobe": probe_video(path)})
    result = {
        "status": "downloaded_extracted_crc_verified",
        "completed_at": now(),
        "archive_sha256": archive_hash,
        "files": records,
        "videos": videos,
        "qfvs_timeline_mapping": "pending_separate_check",
        "original_media_bytes_preserved": True,
        "GPU_used": False,
    }
    write_json(root / "release_manifest.json", result)
    write_json(
        root / "complete.json",
        {
            "status": result["status"],
            "completed_at": now(),
            "files": len(records),
            "videos": len(videos),
            "manifest_sha256": digest(root / "release_manifest.json"),
        },
    )
    write_json(
        root / "status.json",
        {"status": result["status"], "updated_at": now(), "files": len(records), "videos": len(videos)},
    )
    print(json.dumps({"status": result["status"], "files": len(records), "videos": len(videos)}), flush=True)


if __name__ == "__main__":
    main()
