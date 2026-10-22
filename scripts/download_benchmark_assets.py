"""Download a frozen list of public dataset assets with bounded, safe resume."""

import argparse
import json
import os
import time
from pathlib import Path

import requests
from download_qfvs_videos import digest, now, write_json


def transfer(root, asset):
    destination = root / asset["path"]
    if not destination.resolve().is_relative_to(root.resolve()):
        raise ValueError("Asset destination escapes the data root")
    destination.parent.mkdir(parents=True, exist_ok=True)
    receipt_path = root / "receipts" / f"{asset['name']}.json"
    if receipt_path.exists() and destination.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt["sha256"] != digest(destination) or destination.stat().st_size != asset["expected_bytes"]:
            raise ValueError("Completed asset changed")
        return receipt
    partial = destination.with_name(destination.name + ".part")
    if partial.exists() and partial.stat().st_size == asset["expected_bytes"]:
        with partial.open("rb") as stream:
            if not stream.read(64).startswith(bytes.fromhex(asset["magic_hex"])):
                raise ValueError("Completed partial asset has an invalid signature")
        checksum = digest(partial)
        partial.replace(destination)
        receipt = {
            "status": "downloaded_size_and_signature_checked",
            "completed_at": now(),
            "source_url": asset["url"],
            "path": str(destination),
            "bytes": destination.stat().st_size,
            "sha256": checksum,
            "published_checksum_available": False,
            "completed_from_full_partial": True,
        }
        write_json(receipt_path, receipt)
        return receipt
    ledger_path = root / "attempts" / f"{asset['name']}.json"
    attempts = json.loads(ledger_path.read_text()) if ledger_path.exists() else []
    while len(attempts) < 4:
        offset = partial.stat().st_size if partial.exists() else 0
        attempt = {"started_at": now(), "partial_bytes": offset, "status": "started"}
        attempts.append(attempt)
        write_json(ledger_path, attempts)
        then, last = time.monotonic(), 0.0
        try:
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            with requests.get(asset["url"], headers=headers, stream=True, timeout=(30, 90)) as response:
                response.raise_for_status()
                resumed = offset > 0 and response.status_code == 206
                if resumed and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise ValueError("Wrong server resume range")
                if response.status_code not in (200, 206):
                    raise ValueError("Unexpected HTTP response")
                count = offset if resumed else 0
                if offset and not resumed:
                    attempt["restarted_from_zero"] = True
                with partial.open("ab" if resumed else "wb") as output:
                    for chunk in response.iter_content(1024 * 1024):
                        if not chunk:
                            continue
                        if count == 0 and not chunk.startswith(bytes.fromhex(asset["magic_hex"])):
                            raise ValueError("Response has wrong asset signature")
                        output.write(chunk)
                        count += len(chunk)
                        if count > asset["expected_bytes"]:
                            raise ValueError("Remote asset exceeds the frozen size")
                        elapsed = time.monotonic() - then
                        if elapsed - last >= 10:
                            last = elapsed
                            status = {
                                "status": "downloading",
                                "updated_at": now(),
                                "asset": asset["name"],
                                "bytes": count,
                                "expected_bytes": asset["expected_bytes"],
                                "bytes_per_second": (count - (offset if resumed else 0)) / elapsed,
                                "attempt": len(attempts),
                            }
                            write_json(root / "status.json", status)
                            print(json.dumps(status), flush=True)
                    output.flush()
                    os.fsync(output.fileno())
                if partial.stat().st_size != asset["expected_bytes"]:
                    raise ValueError("Incomplete download")
                attempt.update(
                    http_status=response.status_code,
                    resumed=resumed,
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
            with partial.open("rb") as stream:
                if not stream.read(64).startswith(bytes.fromhex(asset["magic_hex"])):
                    raise ValueError("Final asset signature mismatch")
            checksum = digest(partial)
            partial.replace(destination)
            attempt.update(status="complete", finished_at=now(), seconds=time.monotonic() - then)
            write_json(ledger_path, attempts)
            receipt = {
                "status": "downloaded_size_and_signature_checked",
                "completed_at": now(),
                "source_url": asset["url"],
                "path": str(destination),
                "bytes": destination.stat().st_size,
                "sha256": checksum,
                "published_checksum_available": False,
            }
            write_json(receipt_path, receipt)
            return receipt
        except Exception as exc:
            attempt.update(status="failed", finished_at=now(), error=f"{type(exc).__name__}: {exc}")
            write_json(ledger_path, attempts)
            write_json(
                root / "status.json",
                {
                    "status": "retry_pending",
                    "updated_at": now(),
                    "asset": asset["name"],
                    "error": attempt["error"],
                },
            )
            print(json.dumps(attempt), flush=True)
            if len(attempts) < 4:
                time.sleep(15)
    raise RuntimeError(f"Four download attempts exhausted: {asset['name']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    root = Path(manifest["root"]).resolve()
    receipts = [transfer(root, asset) for asset in manifest["assets"]]
    result = {
        "status": "official_assets_downloaded",
        "completed_at": now(),
        "manifest_sha256": digest(args.manifest),
        "assets": receipts,
        "dataset_validation": "pending_separate_check",
        "raw_video_coverage": manifest["raw_video_coverage"],
    }
    write_json(root / "assets_complete.json", result)
    write_json(
        root / "status.json", {"status": result["status"], "updated_at": now(), "assets": len(receipts)}
    )
    print(json.dumps({"status": result["status"], "assets": len(receipts)}), flush=True)


if __name__ == "__main__":
    main()
