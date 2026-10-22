"""Check downloaded QFVS videos against native shot counts without changing media."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import decord
import numpy as np
import scipy
from download_qfvs_videos import digest, now, write_json
from scipy.io import loadmat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="QFVS root containing raw/ and source/")
    args = parser.parse_args()
    root = args.root.resolve()
    raw = root / "raw"
    manifest_path = raw / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    completion = json.loads((raw / "complete.json").read_text())
    if digest(manifest_path) != completion["manifest_sha256"]:
        raise ValueError("Release manifest differs from its completion receipt")
    tags_path = root / "source/eval/Tags.mat"
    tags = loadmat(tags_path)["Tags"][0]
    videos = {Path(item["member"]).stem: item for item in manifest["videos"]}
    if set(videos) != {"P01", "P02", "P03", "P04"} or len(tags) != 4:
        raise ValueError("Expected exactly P01–P04 and four annotation arrays")
    records = []
    for i, video_id in enumerate(sorted(videos)):
        item = videos[video_id]
        path = Path(item["path"])
        if digest(path) != item["sha256"]:
            raise ValueError(f"Extracted video hash mismatch: {video_id}")
        shot_count = len(tags[i][0])
        stream = next(s for s in item["ffprobe"]["streams"] if s["codec_type"] == "video")
        duration = float(stream["duration"])
        reader = decord.VideoReader(str(path), ctx=decord.cpu(0), num_threads=1)
        if len(reader) != int(stream["nb_frames"]):
            raise ValueError(f"Frame count mismatch: {video_id}")
        timestamps = np.asarray(reader.get_frame_timestamp(np.arange(len(reader))))
        if not (
            np.isfinite(timestamps).all()
            and np.all(timestamps[:, 1] > timestamps[:, 0])
            and np.all(np.diff(timestamps[:, 0]) > 0)
        ):
            raise ValueError(f"Invalid frame timestamps: {video_id}")
        samples = []
        for index in (0, len(reader) // 2, len(reader) - 1):
            frame = reader[index].asnumpy()
            if tuple(frame.shape) != (stream["height"], stream["width"], 3):
                raise ValueError(f"Unexpected decoded frame shape: {video_id}")
            # Sample hashes are of decoded RGB pixels, not compressed media bytes.
            samples.append(
                {
                    "frame_index": index,
                    "pts_interval_seconds": timestamps[index].tolist(),
                    "rgb_shape": list(frame.shape),
                    "rgb_sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
                }
            )
        if abs(float(timestamps[0, 0])) > 0.001 or abs(float(timestamps[-1, 1]) - duration) > 0.01:
            raise ValueError(f"Unexpected timestamp origin or endpoint: {video_id}")
        record = {
            "video_id": video_id,
            "path": str(path),
            "sha256": item["sha256"],
            "bytes": item["bytes"],
            "duration_seconds": duration,
            "decoded_reader_frame_count": len(reader),
            "native_shots": shot_count,
            "native_grid_end_seconds": shot_count * 5,
            "raw_end_minus_native_grid_end_seconds": duration - shot_count * 5,
            "native_final_shot_visible_seconds": min(5.0, duration - (shot_count - 1) * 5),
            "raw_tail_after_native_grid_seconds": max(0.0, duration - shot_count * 5),
            "count_matches_floor_or_ceil_duration_over_5": shot_count
            in {math.floor(duration / 5), math.ceil(duration / 5)},
            "all_frame_timestamps_finite_and_monotone": True,
            "decode_samples": samples,
        }
        records.append(record)
        print(json.dumps({k: v for k, v in record.items() if k != "decode_samples"}), flush=True)
        del reader
    result = {
        "status": "integrity_and_sample_decode_verified_with_native_grid_tail_differences",
        "verified_at": now(),
        "source_manifest_sha256": digest(manifest_path),
        "tags_path": str(tags_path),
        "tags_sha256": digest(tags_path),
        "native_shot_seconds": 5,
        "native_summary_index_base": 1,
        "videos": records,
        "total_video_bytes": sum(x["bytes"] for x in records),
        "total_video_seconds": sum(x["duration_seconds"] for x in records),
        "total_native_shots": sum(x["native_shots"] for x in records),
        "decode_scope": "First, middle and last frame per video; not a full-video pixel decode.",
        "mapping_scope": "Video IDs, timestamp metadata and shot counts; not per-shot semantic verification.",
        "media_or_annotations_modified": False,
        "inference_executed": False,
        "versions": {"decord": decord.__version__, "numpy": np.__version__, "scipy": scipy.__version__},
        "script_sha256": digest(Path(__file__)),
    }
    write_json(raw / "video_audit.json", result)


if __name__ == "__main__":
    main()
