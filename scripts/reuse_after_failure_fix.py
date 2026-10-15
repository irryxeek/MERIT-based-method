"""Audit and reuse completed work after the reviewed CLI failure-boundary fix.

All model settings and inference modules must match. Reconstruct exact original
request payloads from verified images before rekeying their caches. Failed requests
retain their exhausted retry state. Original response records remain available.
"""

import argparse
import json
import re
from pathlib import Path

from merit_qvh.cli import frozen_run
from merit_qvh.client import image_part
from merit_qvh.core import digest, file_sha256, read_jsonl, write_json
from merit_qvh.media import prepare_frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_run", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_run
    config = json.loads(args.config.read_text())
    protocol_id, target = frozen_run(config)
    old = json.loads((source / "protocol.json").read_text())
    new = json.loads((target / "protocol.json").read_text())
    changed = {
        name
        for name in old["source_sha256"].keys() | new["source_sha256"].keys()
        if old["source_sha256"].get(name) != new["source_sha256"].get(name)
    }
    if old["config"] != config or changed != {"src/merit_qvh/cli.py"}:
        raise ValueError("Reuse requires identical settings and only the reviewed CLI change")
    old_id = digest(old)
    rows = {}
    for split in ("smoke", "val"):
        original = read_jsonl(source / "manifests" / f"{split}.jsonl")
        if original != read_jsonl(target / "manifests" / f"{split}.jsonl"):
            raise ValueError("Inference inputs changed")
        rows.update({row["qid"]: row for row in original})
    by_vid = {row["vid"]: row for row in rows.values()}
    originals = {path.stem: json.loads(path.read_text()) for path in (source / "memories").glob("*.json")}
    used_vids = set(originals)
    for log in (source / "logs").glob("main_*.log"):
        used_vids.update(re.findall(r'"event": "start_video"[^\n]*"vid": "([^"]+)"', log.read_text()))
    media = {}
    images = {}
    for vid in sorted(used_vids):
        media[vid] = prepare_frames(by_vid[vid], config["memory"], config["artifact_root"])
        for clip in media[vid]["clips"]:
            for frame in clip["frames"]:
                if file_sha256(frame["path"]) != frame["sha256"]:
                    raise ValueError("Cached frame changed")
                url = image_part(frame["path"])["image_url"]["url"]
                images[digest(url)] = url

    def restore(value):
        if isinstance(value, dict):
            if set(value) == {"image_data_digest", "encoded_length"}:
                url = images[value["image_data_digest"]]
                if len(url) != value["encoded_length"]:
                    raise ValueError("Encoded image length differs")
                return url
            return {key: restore(item) for key, item in value.items()}
        if isinstance(value, list):
            return [restore(item) for item in value]
        return value

    generation_ids = {old_id} | {
        memory.get("generation_protocol_id", old_id) for memory in originals.values()
    }
    aliases = []
    failed = []
    for path in sorted((source / "requests").glob("*/*.json")):
        record = json.loads(path.read_text())
        payload = restore(record["payload"])
        matches = [
            generation_id
            for generation_id in generation_ids
            if digest({"protocol": generation_id, "stage": record["stage"], "payload": payload})
            == record["request_id"]
        ]
        if len(matches) != 1:
            raise ValueError(f"Original request identity did not verify: {path}")
        request_id = digest({"protocol": protocol_id, "stage": record["stage"], "payload": payload})
        write_json(target / "requests" / path.parent.name / path.name, record)
        alias = {
            **record,
            "request_id": request_id,
            "reused_from": str(path),
            "generation_request_id": record["request_id"],
            "generation_protocol_id": matches[0],
        }
        destination = target / "requests" / request_id[:2] / f"{request_id}.json"
        if destination.exists() and json.loads(destination.read_text()) != alias:
            raise ValueError("Conflicting cached responses")
        write_json(destination, alias)
        aliases.append({"source": record["request_id"], "target": request_id})
        if record["status"] != "ok":
            failed.append(request_id)
    for vid, memory in originals.items():
        inputs = [
            {key: clip[key] for key in ("clip_id", "start", "end", "frames")} for clip in memory["clips"]
        ]
        if inputs != media[vid]["clips"] or memory["source_sha256"] != media[vid]["source_sha256"]:
            raise ValueError("Memory generation inputs changed")
        if memory["identity"] != digest({"protocol": old_id, "media": media[vid]["identity"]}):
            raise ValueError("Original memory identity differs")
        copied = {
            **memory,
            "identity": digest({"protocol": protocol_id, "media": media[vid]["identity"]}),
            "protocol_id": protocol_id,
            "generation_protocol_id": memory.get("generation_protocol_id", old_id),
            "reused_from": str(source / "memories" / f"{vid}.json"),
        }
        write_json(target / "memories" / f"{vid}.json", copied)
    completed = []
    for path in sorted((source / "queries").glob("*.json")):
        if path.name.endswith(".partial.json"):
            continue
        query = json.loads(path.read_text())
        row = rows[query["qid"]]
        original_identity = digest(
            {"protocol": old_id, "memory": originals[row["vid"]]["identity"], "input": row}
        )
        if query["identity"] != original_identity or query["status"] != "ok":
            raise ValueError("Only verified completed queries can be reused")
        memory = json.loads((target / "memories" / f"{row['vid']}.json").read_text())
        copied = {
            **query,
            "identity": digest({"protocol": protocol_id, "memory": memory["identity"], "input": row}),
            "protocol_id": protocol_id,
            "inference_protocol_id": old_id,
            "reused_from": str(path),
        }
        write_json(target / "queries" / path.name, copied)
        completed.append(query["qid"])
    record = {
        "source_run": str(source),
        "source_protocol": old_id,
        "target_protocol": protocol_id,
        "reviewed_changed_files": sorted(changed),
        "reason": "Extend the existing whole-query fallback to memory/index failures; allow evaluation with missing memories.",
        "reused_memories": len(originals),
        "reused_query_ids": completed,
        "verified_images": len(images),
        "request_aliases": aliases,
        "preserved_failed_request_ids": failed,
    }
    write_json(target / "failure_fix_reuse.json", record)
    print(json.dumps({key: value for key, value in record.items() if key != "request_aliases"}))


if __name__ == "__main__":
    main()
