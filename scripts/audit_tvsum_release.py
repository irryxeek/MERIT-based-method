"""Audit official TVSum video IDs, frame-level arrays, hashes and sample decoding."""

import argparse
import csv
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import decord
import h5py
import numpy as np
from download_qfvs_videos import digest, now, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    release = root / "official_release/ydata-tvsum50-v1_1"
    manifest = json.loads((root / "release_manifest.json").read_text())
    hashes = {item["path"]: item["sha256"] for item in manifest["files"]}
    metadata = list(csv.DictReader((release / "data/ydata-tvsum50-info.tsv").open(), delimiter="\t"))
    tsv = defaultdict(list)
    with (release / "data/ydata-tvsum50-anno.tsv").open() as source:
        for row in csv.reader(source, delimiter="\t"):
            tsv[row[0]].append(np.fromstring(row[2], sep=",", dtype=np.float64))
    video_ids = {row["video_id"] for row in metadata}
    media = {p.stem: p for p in (release / "video").glob("*.mp4")}
    if len(metadata) != 50 or len(video_ids) != 50 or set(media) != video_ids or set(tsv) != video_ids:
        raise ValueError("Expected exactly 50 matching media, metadata and annotation IDs")
    findings, records, mat_ids = [], [], []
    with h5py.File(release / "matlab/ydata-tvsum50.mat", "r") as source:
        group = source["tvsum50"]
        for i in range(50):
            values = {key: source[dataset[i, 0]][()] for key, dataset in group.items()}
            video_id = "".join(chr(int(x)) for x in values["video"].ravel())
            mat_ids.append(video_id)
            nframes = int(values["nframes"].item())
            user_scores = values["user_anno"]
            mean_scores = values["gt_score"].ravel()
            if (
                user_scores.shape != (20, nframes)
                or mean_scores.shape != (nframes,)
                or not np.isfinite(user_scores).all()
                or user_scores.min() < 1
                or user_scores.max() > 5
                or not np.allclose(mean_scores, user_scores.mean(axis=0), atol=1e-6)
            ):
                raise ValueError(f"Invalid original MAT annotation: {video_id}")
            if not np.array_equal(np.stack(tsv[video_id]), user_scores):
                raise ValueError(f"TSV and MAT annotations differ: {video_id}")
            path = media[video_id]
            if digest(path) != hashes[str(path)]:
                raise ValueError(f"Media differs from extracted archive: {video_id}")
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
            reader = decord.VideoReader(str(path), ctx=decord.cpu(0), num_threads=1)
            count_matches = len(reader) == nframes
            if not count_matches:
                findings.append(
                    {"video_id": video_id, "media_frames": len(reader), "annotation_frames": nframes}
                )
            samples = []
            for index in (0, len(reader) // 2, len(reader) - 1):
                frame = reader[index].asnumpy()
                if frame.shape != (stream["height"], stream["width"], 3):
                    raise ValueError(f"Decoded frame shape differs: {video_id}")
                samples.append({"frame": index, "rgb_sha256": hashlib.sha256(frame.tobytes()).hexdigest()})
            records.append(
                {
                    "video_id": video_id,
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": hashes[str(path)],
                    "duration_seconds": duration,
                    "mat_duration_seconds": float(values["length"].item()),
                    "native_annotation_frames": nframes,
                    "decoded_reader_frame_count": len(reader),
                    "annotators": 20,
                    "media_annotation_frame_count_matches": count_matches,
                    "tsv_mat_annotations_equal": True,
                    "gt_score_equals_annotator_mean": True,
                    "sample_decode": samples,
                    "ffprobe": probe,
                }
            )
            print(
                json.dumps(
                    {"video_id": video_id, "frames": len(reader), "frame_count_matches": count_matches}
                )
            )
            del reader
    if len(set(mat_ids)) != 50 or set(mat_ids) != video_ids:
        raise ValueError("MAT IDs differ from the original release")
    result = {
        "status": "verified" if not findings else "downloaded_with_frame_alignment_findings",
        "verified_at": now(),
        "videos": records,
        "video_count": 50,
        "annotation_rows": sum(len(rows) for rows in tsv.values()),
        "total_video_bytes": sum(row["bytes"] for row in records),
        "total_video_seconds": sum(row["duration_seconds"] for row in records),
        "annotation_storage": "Both TSV and MAT contain frame-aligned scores; collection used 2-second shots.",
        "frame_alignment_findings": findings,
        "decode_scope": "First, middle and last frame per video; not full-video pixel decoding.",
        "media_or_annotations_modified": False,
        "versions": {"decord": decord.__version__, "numpy": np.__version__, "h5py": h5py.__version__},
        "script_sha256": digest(Path(__file__)),
    }
    write_json(root / "video_audit.json", result)
    write_json(root / "complete.json", {k: v for k, v in result.items() if k != "videos"})


if __name__ == "__main__":
    main()
