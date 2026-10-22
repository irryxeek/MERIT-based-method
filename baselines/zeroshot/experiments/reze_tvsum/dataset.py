"""Audit author TVSum inputs; keep labels out of the GPU manifest."""

import ast
import json
import os
import zipfile
from pathlib import Path

import numpy as np
from shared import HERE, atomic_json, sha256, utc_now

DATA = Path((os.environ.get("REPRO_DATA_ROOT", "data") + "/TVSum"))
RELEASE = DATA / "official_release/ydata-tvsum50-v1_1"
UMT_DATA = Path((os.environ.get("ZEROSHOT_ARTIFACT_ROOT", "artifacts/zeroshot") + "/metric_sources/tvsum_umt_release"))
UMT_REVISION = "bde05e3de39049bf90d8807f6b2a4ed8b42acbb0"
ARCHIVE_SHA = "ec05ad4e26f2fb17cbe32b02bb16deb7745be200701e1569f5a18e20d0d47a23"


def author_splits():
    tree = ast.parse((HERE / "vendor/umt_static.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "TVSUM_SPLITS" for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError("Author split not found")


def audit_dataset():
    archive = UMT_DATA / "tvsum-ec05ad4e.zip"
    if sha256(archive) != ARCHIVE_SHA:
        raise ValueError("UMT author archive changed")
    receipt = json.loads((UMT_DATA / "download_receipt.json").read_text())
    labels_path = UMT_DATA / "tvsum_anno.json"
    if sha256(labels_path) != receipt["annotation_sha256"]:
        raise ValueError("UMT annotations changed")
    labels = json.loads(labels_path.read_text())
    raw = json.loads((DATA / "video_audit.json").read_text())
    native = {v["video_id"]: v for v in raw["videos"]}
    splits = author_splits()
    split_ids = [v for group in splits.values() for ids in group.values() for v in ids]
    if len(split_ids) != 50 or len(set(split_ids)) != 50 or set(split_ids) != set(labels) or set(labels) != set(native):
        raise ValueError("Expected the same 50 unique IDs in labels, media and author split")
    records = {}
    with zipfile.ZipFile(archive) as source:
        if source.read("tvsum/tvsum_anno.json") != labels_path.read_bytes():
            raise ValueError("Extracted annotation differs from archive")
        for vid, label in labels.items():
            values = np.asarray(label["anno"])
            if values.ndim != 2 or values.shape[1] != 20 or not np.isfinite(values).all():
                raise ValueError(f"Invalid 20-annotator labels: {vid}")
            if values.min() < 1 or values.max() > 5:
                raise ValueError("Annotation score outside [1, 5]")
            shapes = {}
            for kind, suffix in (("video", "_rgb"), ("video", "_opt"), ("audio", "")):
                with source.open(f"tvsum/{kind}_features/{vid}{suffix}.npy") as stream:
                    version = np.lib.format.read_magic(stream)
                    shape, _, _ = np.lib.format._read_array_header(stream, version)
                shapes[kind + suffix] = shape[0]
            n = len(values)
            video = native[vid]
            if min(shapes.values()) != n or int(video["duration_seconds"] / 2) != n:
                raise ValueError(f"Author feature/label/2-second time grids disagree: {vid}")
            if int(label["frames"]) != video["decoded_reader_frame_count"]:
                raise ValueError("Author metadata differs from official video frame count")
            category = label["domain"]
            if vid not in splits[category]["train"] + splits[category]["val"]:
                raise ValueError("Category and author split disagree")
            records[vid] = {
                "vid": vid,
                "category": category,
                "umt_split": "val" if vid in splits[category]["val"] else "train",
                "duration": video["duration_seconds"],
                "n_2s_clips": n,
                "feature_lengths": shapes,
                "video_path": video["path"],
                "media_sha256": video["sha256"],
                "media_size": video["bytes"],
                "native_annotation_frames": video["native_annotation_frames"],
                "media_frames": video["decoded_reader_frame_count"],
            }
    result = {
        "status": "passed",
        "checked_at": utc_now(),
        "videos": records,
        "n_videos": 50,
        "umt_val_ids": sorted(v for group in splits.values() for v in group["val"]),
        "total_seconds": sum(v["duration"] for v in records.values()),
        "total_2s_clips": sum(v["n_2s_clips"] for v in records.values()),
        "umt_revision": UMT_REVISION,
        "archive_sha256": ARCHIVE_SHA,
        "annotation_sha256": sha256(labels_path),
        "native_video_audit_sha256": sha256(DATA / "video_audit.json"),
        "policy": "Use author-released 2s labels unchanged; no frame-label reconstruction; no features used for inference",
        "frame_count_findings": raw["frame_alignment_findings"],
    }
    atomic_json(UMT_DATA / "dataset_audit.json", result)
    return result


if __name__ == "__main__":
    audit = audit_dataset()
    print(json.dumps({k: v for k, v in audit.items() if k != "videos"}))
