"""Preserve smoke captions when their model and exact generation inputs are unchanged.

Reuse is permitted only when the memory-generation config, prompts, client, and every
caption input are identical. The reviewed media/audit fixes and a separate embedding
device assignment do not invalidate identical captions. Original responses stay intact.
"""

import argparse
import json
from pathlib import Path

from merit_qvh.cli import frozen_run
from merit_qvh.core import digest, read_jsonl, write_json
from merit_qvh.media import prepare_frames

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("source_run", type=Path)
parser.add_argument("--config", type=Path)
args = parser.parse_args()
repo = Path(__file__).resolve().parents[1]
config = json.loads((args.config or repo / "configs/main_v1.json").read_text())
protocol_id, target = frozen_run(config)
old_protocol = json.loads((args.source_run / "protocol.json").read_text())
new_protocol = json.loads((target / "protocol.json").read_text())
if any(old_protocol["config"][key] != config[key] for key in ("vlm", "memory")):
    raise ValueError("Memory generation configuration changed; do not reuse memories")
changed = {
    name
    for name in old_protocol["source_sha256"]
    if old_protocol["source_sha256"][name] != new_protocol["source_sha256"].get(name)
}
if changed - {"src/merit_qvh/media.py", "src/merit_qvh/cli.py"}:
    raise ValueError("Changes exceed the reviewed media/audit fixes")
rows = read_jsonl(target / "manifests/smoke.jsonl")
if rows != read_jsonl(args.source_run / "manifests/smoke.jsonl"):
    raise ValueError("Smoke inputs changed")
records = []
for row in rows:
    source = args.source_run / "memories" / (row["vid"] + ".json")
    if not source.exists():
        continue
    memory = json.loads(source.read_text())
    media = prepare_frames(row, config["memory"], config["artifact_root"])
    caption_inputs = [
        {key: clip[key] for key in ("clip_id", "start", "end", "frames")} for clip in memory["clips"]
    ]
    if caption_inputs != media["clips"] or memory["source_sha256"] != media["source_sha256"]:
        raise ValueError(f"Caption inputs changed for {row['vid']}")
    for event in memory["requests"]:
        request_id = event["request_id"]
        relative = Path("requests") / request_id[:2] / (request_id + ".json")
        record = json.loads((args.source_run / relative).read_text())
        if record["status"] != "ok":
            raise ValueError("Cannot reuse incomplete generation")
        write_json(target / relative, record)
    identity = digest({"protocol": protocol_id, "media": media["identity"]})
    memory.update(
        identity=identity,
        protocol_id=protocol_id,
        reused_from=str(source),
        generation_protocol_id=memory.get("generation_protocol_id", digest(old_protocol)),
        generation_source=memory.get("generation_source", memory.get("reused_from", str(source))),
        caption_inputs_digest=digest(caption_inputs),
    )
    write_json(target / "memories" / source.name, memory)
    records.append(
        {
            "vid": row["vid"],
            "source": str(source),
            "identity": identity,
            "unchanged_caption_inputs_digest": digest(caption_inputs),
        }
    )
write_json(
    target / "memory_reuse.json",
    {
        "source_protocol": digest(old_protocol),
        "target_protocol": protocol_id,
        "records": records,
        "reviewed_changed_files": sorted(changed),
    },
)
print(json.dumps({"reused_videos": len(records), "target": str(target)}))
