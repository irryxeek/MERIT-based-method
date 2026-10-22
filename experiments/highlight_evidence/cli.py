"""CPU source-frame smoke: decode a bounded, query-bound view without model calls."""

import argparse
import base64
from pathlib import Path

from hd_adapter.media import probe

from merit_qvh.core import digest, write_json

from .media import decode_view
from .planner import Budget, Need, plan_views, unresolved


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--duration", type=float, required=True)
    p.add_argument("--start", type=float, required=True)
    p.add_argument("--end", type=float, required=True)
    p.add_argument("--query", required=True)
    p.add_argument("--question", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    info = probe(args.video, args.duration, args.output / "probes")
    need = Need("smoke_event", "visible_evidence", args.question, args.start, args.end)
    gaps = unresolved([need], [], [], args.duration)
    plan = plan_views(args.query, gaps, info["pts"], args.duration, Budget(max_views=1, max_new_frames=24))
    frames = decode_view(args.video, info, plan["views"][0])
    metadata = []
    for f in frames:
        encoded = f["url"].split(",", 1)[1]
        assert digest(encoded) == f["jpeg_sha256"]
        path = args.output / (f["image_id"] + ".jpg")
        path.write_bytes(base64.b64decode(encoded))
        metadata.append({**{k: v for k, v in f.items() if k != "url"}, "file": path.name})
    report = {
        "scope": "Real-video CPU decoding check only; no VLM observations or accuracy claims",
        "source_sha256": info["media_sha256"],
        "timeline_sha256": info["timeline_sha256"],
        "plan": plan,
        "frames": metadata,
        "model_calls": 0,
        "distinct_source_frames": len({f["pts"] for f in frames}),
    }
    write_json(args.output / "smoke.json", report)
    print(f"Decoded {len(frames)} distinct real frames; report: {args.output / 'smoke.json'}")


if __name__ == "__main__":
    main()
