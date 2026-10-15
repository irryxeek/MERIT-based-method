"""Audit and cache validation media with bounded CPU workers, without model inference."""

import argparse
import concurrent.futures
import json
import time
from pathlib import Path

from merit_qvh.cli import frozen_run
from merit_qvh.core import read_jsonl, validate_manifest, write_json
from merit_qvh.media import prepare_frames


def decode(job):
    row, config = job
    started = time.monotonic()
    try:
        media = prepare_frames(row, config["memory"], config["artifact_root"])
        return {
            "vid": row["vid"],
            "status": "ok",
            "identity": media["identity"],
            "decoded_duration": media["decoded_duration"],
            "annotation_duration": media["annotation_duration"],
            "unobserved_tail_seconds": media["unobserved_tail_seconds"],
            "clips": len(media["clips"]),
            "frames": sum(len(clip["frames"]) for clip in media["clips"]),
            "seconds": time.monotonic() - started,
        }
    except (OSError, RuntimeError, ValueError) as error:
        return {"vid": row["vid"], "status": "failed", "error": str(error)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--split", choices=("smoke", "val"), default="val")
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error("Use between 1 and 16 CPU workers")
    repo = Path(__file__).resolve().parents[1]
    config = json.loads((repo / "configs/main_v1.json").read_text())
    protocol_id, root = frozen_run(config)
    rows = validate_manifest(read_jsonl(root / "manifests" / (args.split + ".jsonl")))
    unique = {row["vid"]: row for row in rows}
    report = {"protocol_id": protocol_id, "workers": args.workers, "total_videos": len(unique), "results": []}
    started = time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(decode, (row, config)) for row in unique.values()]
        for future in concurrent.futures.as_completed(futures):
            record = future.result()
            report["results"].append(record)
            if len(report["results"]) % 50 == 0 or record["status"] != "ok":
                report["seconds"] = time.monotonic() - started
                write_json(root / (args.split + "_media_audit.json"), report)
                print(
                    json.dumps(
                        {
                            "completed": len(report["results"]),
                            "total": len(unique),
                            "last_status": record["status"],
                            "seconds": report["seconds"],
                        }
                    ),
                    flush=True,
                )
    report["seconds"] = time.monotonic() - started
    report["failures"] = sum(row["status"] != "ok" for row in report["results"])
    write_json(root / (args.split + "_media_audit.json"), report)
    print(
        json.dumps(
            {
                "completed": len(report["results"]),
                "failures": report["failures"],
                "seconds": report["seconds"],
            }
        ),
        flush=True,
    )
    if report["failures"]:
        raise RuntimeError("Some media failed; inspect the audit report")


if __name__ == "__main__":
    main()
