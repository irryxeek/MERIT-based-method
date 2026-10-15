"""Project official annotations to a label-free whitelist, then audit actual media."""

import _bootstrap  # noqa: F401 -- establish the isolated implementation import
import argparse
import concurrent.futures
import json
import os
from collections import defaultdict
from pathlib import Path

from videoarm.qvhighlights.common import (
    ARTIFACTS,
    atomic_json,
    digest,
    file_hash,
    load_manifest,
    read_jsonl,
    source_id,
    write_jsonl,
)
from videoarm.qvhighlights.media import metadata, probe


def development_sample(train):
    groups = defaultdict(list)
    for row in train:
        groups[source_id(row["vid"])[0]].append(row)
    sources = sorted(groups, key=lambda s: digest([42, "source", s]))
    # One query per source in round one; later rounds only needed for a tiny test fixture.
    groups = {
        s: sorted(v, key=lambda r: digest([42, "query", r["qid"], r["vid"]]))
        for s, v in groups.items()
    }
    result = []
    for depth in range(max(map(len, groups.values()))):
        result.extend(groups[s][depth] for s in sources if depth < len(groups[s]))
        if len(result) >= 50:
            break
    return result[:50]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--media-root",
        default="/path/to/storage/longvt-repro/benchmark_data/QVHighlights/videos_full/videos",
    )
    ap.add_argument("--official-repo", default=str(ARTIFACTS / "external/moment_detr"))
    ap.add_argument(
        "--data-root",
        default=os.getenv("QVH_DATA_ROOT", "/path/to/storage/videoarm-repro/qvh_data"),
    )
    ap.add_argument(
        "--probe",
        choices=[
            "none",
            "metadata",
            "dev50",
            "all",
            "metadata_mapping",
            "dev50_mapping",
        ],
        default="metadata_mapping",
    )
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    root = Path(args.data_root)
    root.mkdir(parents=True, exist_ok=True)
    manifest_root = ARTIFACTS / "manifests"
    splits = {}
    hashes = {}
    for split in ("train", "val", "test"):
        annotation = (
            Path(args.official_repo) / "data" / f"highlight_{split}_release.jsonl"
        )
        hashes[split] = file_hash(annotation)
        # test_release has identifiers only; never read test_with_gt here.
        splits[split] = [
            {
                "qid": r["qid"],
                "query": r["query"],
                "vid": r["vid"],
                "duration": r["duration"],
                "media_path": str(
                    Path(args.media_root).resolve() / (r["vid"] + ".mp4")
                ),
                "timeline_id": digest(
                    [r["vid"], "dataset_segment_origin", r["duration"]]
                ),
                "num_clips": int(r["duration"] / 2),
                "split": split,
            }
            for r in read_jsonl(annotation)
        ]
        write_jsonl(manifest_root / f"{split}.jsonl", splits[split])
        load_manifest(manifest_root / f"{split}.jsonl")
    qid_sets = {k: {r["qid"] for r in v} for k, v in splits.items()}
    if any(
        qid_sets[a] & qid_sets[b]
        for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
    ):
        raise ValueError("official qids overlap across splits")
    dev = development_sample(splits["train"])
    for name, rows in (("dev50", dev), ("smoke8", dev[:8])):
        write_jsonl(manifest_root / f"{name}.jsonl", rows)
    sampling = {
        "seed": 42,
        "algorithm": "SHA256 of canonical JSON [42,source,source_id]; first query per source ranked by SHA256 [42,query,qid,vid]; first 50 sources; smoke first 8",
        "dev50_qids": [r["qid"] for r in dev],
        "smoke8_qids": [r["qid"] for r in dev[:8]],
        "train_sha256": hashes["train"],
    }
    sample_path = manifest_root / "development_selection.json"
    if sample_path.exists() and json.loads(sample_path.read_text()) != sampling:
        raise ValueError("previously registered sample changed")
    atomic_json(sample_path, sampling)
    totals = {}
    for split, rows in splits.items():
        vids = {r["vid"] for r in rows}
        missing = [r["qid"] for r in rows if not Path(r["media_path"]).is_file()]
        totals[split] = {
            "queries": len(rows),
            "segments": len(vids),
            "source_videos": len({source_id(v)[0] for v in vids}),
            "media_queries": len(rows) - len(missing),
            "missing_qids": missing,
        }
    audit = {
        "annotation_sha256": hashes,
        "splits": totals,
        "sampling": sampling,
        "media_root": args.media_root,
        "data_root": str(root),
        "probe_mode": args.probe,
    }
    atomic_json(ARTIFACTS / "checks/data_inventory.json", audit)
    all_rows = {r["vid"]: r for rows in splits.values() for r in rows}
    selected = (
        {r["vid"] for r in dev} if args.probe.startswith("dev50") else set(all_rows)
    )
    if args.probe != "none":
        full = args.probe.startswith("dev50") or args.probe == "all"

        def check(vid):
            row = all_rows[vid]
            try:
                p = (
                    probe(row["media_path"], row["duration"], root / "timelines")
                    if full
                    else metadata(row["media_path"])
                )
                gap = row["duration"] - p["video_duration"]
                if args.probe == "metadata" and abs(gap) > 0.25:
                    raise ValueError(
                        f"duration mismatch: {p['video_duration']} vs {row['duration']}"
                    )
                if not -1 <= gap < 2:
                    raise ValueError("duration requires unresolved mapping")
                return {
                    "vid": vid,
                    "status": "passed",
                    "annotation_duration": row["duration"],
                    "media_duration": p["video_duration"],
                    "origin": p["origin"],
                    "duration_gap_seconds": gap,
                    "partial_tail": gap > 0.25,
                    "full_pts_and_first_last_decode": full,
                    **(
                        {
                            "media_sha256": p["media_sha256"],
                            "timeline_sha256": p["timeline_sha256"],
                            "first_pts": p["pts"][0],
                            "last_pts": p["pts"][-1],
                            "vfr": p["vfr"],
                        }
                        if full
                        else {}
                    ),
                }
            except Exception as e:
                return {
                    "vid": vid,
                    "status": "failed",
                    "error": f"{type(e).__name__}: {e}",
                }

        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            for row in pool.map(check, sorted(selected)):
                results.append(row)
                if len(results) % 100 == 0 or len(results) == len(selected):
                    write_jsonl(
                        ARTIFACTS / "checks" / f"media_{args.probe}.jsonl", results
                    )
                    atomic_json(
                        ARTIFACTS / "checks" / f"media_{args.probe}_progress.json",
                        {
                            "planned": len(selected),
                            "completed": len(results),
                            "failed": sum(r["status"] == "failed" for r in results),
                        },
                    )
                    print(f"{args.probe}: {len(results)}/{len(selected)}", flush=True)
                    if args.probe == "all":
                        from refresh_progress import refresh

                        refresh(args.data_root)
        audit["probe"] = {
            "planned": len(selected),
            "passed": sum(r["status"] == "passed" for r in results),
            "failed": sum(r["status"] == "failed" for r in results),
        }
        atomic_json(ARTIFACTS / f"checks/data_audit_{args.probe}.json", audit)
    print(json.dumps(totals, ensure_ascii=False))


if __name__ == "__main__":
    main()
