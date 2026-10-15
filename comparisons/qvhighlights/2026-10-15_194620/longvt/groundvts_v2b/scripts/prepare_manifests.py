#!/usr/bin/env python3
"""Freeze label-free interval-inference manifests from audited QVHighlights inputs."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[2]
DEV_SOURCE = REPO_ROOT / "reproduction/qvhighlights/lora_sft/data/pilot_dev.jsonl"
DEV_GT_SOURCE = REPO_ROOT / "reproduction/qvhighlights/lora_sft/data/pilot_dev_ground_truth.jsonl"
OVERLAP_AUDIT = REPO_ROOT / "reproduction/qvhighlights/lora_sft/data/overlap_audit.json"
VAL_SOURCE = REPO_ROOT / "reproduction/qvhighlights/manifests/full_validation.json"
VAL_GT_SOURCE = REPO_ROOT / "reproduction/qvhighlights/manifests/validation_ground_truth.jsonl"
ALLOWED_FIELDS = {
    "qid", "vid", "query", "source_video_id", "youtube_id", "source_start_seconds",
    "source_end_seconds", "duration", "num_clips", "video_path", "actual_duration",
    "media_timeline", "media_source",
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def project(row: dict) -> dict:
    return {key: row[key] for key in ALLOWED_FIELDS if key in row}


def validate(rows: list[dict], expected: int, label: str) -> None:
    qids = [row["qid"] for row in rows]
    if len(rows) != expected or len(set(qids)) != expected:
        raise ValueError(f"{label}: expected {expected} unique qids, got {len(rows)} rows/{len(set(qids))} qids")
    for row in rows:
        if not Path(row["video_path"]).is_file():
            raise FileNotFoundError(f"{label}: missing media for qid={row['qid']}: {row['video_path']}")
        if row["num_clips"] != int(row["duration"] / 2):
            raise ValueError(f"{label}: official clip count mismatch at qid={row['qid']}")
        leaked = {"target_scores", "target_text", "saliency_scores", "relevant_clip_ids", "relevant_windows"} & set(row)
        if leaked:
            raise ValueError(f"{label}: label fields leaked into inference manifest: {sorted(leaked)}")


def main() -> None:
    audit = json.loads(OVERLAP_AUDIT.read_text())
    if any(rows for comparison in audit.values() for rows in comparison.values()):
        raise ValueError("Existing split audit is not empty; inspect before using dev split")

    dev_rows = [project(row) for row in read_jsonl(DEV_SOURCE)]
    dev_rows.sort(key=lambda row: row["qid"])
    validate(dev_rows, 100, "dev")
    dev_gt = read_jsonl(DEV_GT_SOURCE)
    if len(dev_gt) != 100 or {row["qid"] for row in dev_rows} != {row["qid"] for row in dev_gt}:
        raise ValueError("Dev inference manifest/GT qid mismatch")
    val_rows = [project(row) for row in json.loads(VAL_SOURCE.read_text())]
    val_gt = read_jsonl(VAL_GT_SOURCE)
    validate(val_rows, 1550, "validation")
    if {row["qid"] for row in val_rows} != {row["qid"] for row in val_gt}:
        raise ValueError("Validation manifest/GT qid mismatch")
    if {row["qid"] for row in dev_rows} & {row["qid"] for row in val_rows}:
        raise ValueError("Dev/validation qid overlap")

    smoke = dev_rows[:8]
    for name, rows in (("pilot_dev.json", dev_rows), ("smoke.json", smoke), ("validation.json", val_rows)):
        (ROOT / "manifests" / name).write_text(json.dumps(rows, indent=2) + "\n")
    (ROOT / "manifests" / "pilot_dev_ground_truth.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in dev_gt)
    )
    (ROOT / "manifests" / "validation_ground_truth.jsonl").write_bytes(VAL_GT_SOURCE.read_bytes())

    source_manifest = {
        "sources": {
            str(path): {"sha256": digest(path), "bytes": path.stat().st_size}
            for path in (DEV_SOURCE, DEV_GT_SOURCE, OVERLAP_AUDIT, VAL_SOURCE, VAL_GT_SOURCE)
        },
        "dev_rows": len(dev_rows), "smoke_rows": len(smoke), "validation_rows": len(val_rows),
        "dev_val_source_overlap": [],
        "label_fields_removed": ["target_scores", "target_text", "saliency_scores", "relevant_clip_ids", "relevant_windows"],
    }
    (ROOT / "manifests" / "manifest_audit.json").write_text(json.dumps(source_manifest, indent=2) + "\n")
    print(json.dumps({"dev": len(dev_rows), "smoke": len(smoke), "validation": len(val_rows), "all_media_present": True}))


if __name__ == "__main__":
    main()
