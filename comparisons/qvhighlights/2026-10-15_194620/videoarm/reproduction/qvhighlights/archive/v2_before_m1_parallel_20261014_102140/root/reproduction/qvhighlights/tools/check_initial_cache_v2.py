"""Verify exact initializer request compatibility without making model requests."""

import _bootstrap  # noqa: F401
import base64
import json
import os
import sqlite3
from pathlib import Path

from videoarm.qvhighlights import prompts
from videoarm.qvhighlights.common import ARTIFACTS, atomic_json, load_manifest
from videoarm.qvhighlights.contracts import clip_scores
from videoarm.qvhighlights.pipeline import Engine, parse_events
from videoarm.qvhighlights.state import Store


class NoNetworkBackend:
    def complete(self, *args, **kwargs):
        raise RuntimeError("cache compatibility probe forbids any new model request")


root = Path(os.getenv("QVH_DATA_ROOT", str(ARTIFACTS.parents[1] / "qvh_data")))
manifest = {r["qid"]: r for r in load_manifest(ARTIFACTS / "manifests/dev50.jsonl")}
checks = []
for task, stage, role, instruction in (
    ("mr", "overview", "scene", prompts.OVERVIEW),
    ("hd", "initialize_0", "clip", prompts.HD),
):
    source = root / "runs" / f"dev_v1_{task}_smoke8_B0" / "state.sqlite"
    db = sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    saved = db.execute(
        "SELECT * FROM requests WHERE stage=? AND status='completed' ORDER BY qid LIMIT 1",
        (stage,),
    ).fetchone()
    request = json.loads(saved["request"])
    context = json.loads(request["text"].splitlines()[0])
    state = json.loads(
        db.execute(
            "SELECT snapshot FROM items WHERE qid=?", (saved["qid"],)
        ).fetchone()[0]
    )
    db.close()
    frames = [
        {
            **f,
            "url": "data:image/jpeg;base64,"
            + base64.b64encode(Path(f["path"]).read_bytes()).decode(),
        }
        for f in request["frames"]
    ]
    row = manifest[saved["qid"]]
    config = json.loads((ARTIFACTS / "configs/default_v2.json").read_text())
    config.update(task=task, method="B0")
    info = {
        "media_sha256": state["media_sha256"],
        "timeline_sha256": state["timeline_sha256"],
        "video_duration": context["available_media_duration"],
        "mapping_audit": context["mapping_audit"],
    }
    store = Store(
        root / "probes" / f"initial_cache_v2_{task}",
        {"config": config, "source_run": str(source)},
        [row],
    )
    engine = Engine(row, config, store, NoNetworkBackend(), None, info)
    if task == "hd":
        ids = [f["clip_id"] for f in frames]
        instruction += f"\nExpected IDs: {ids}"

        def parser(p):
            clip_scores(p, ids)
            return p
    else:
        parser = parse_events
    engine.request(stage, role, instruction, frames, parser)
    assert (
        engine.budget["actual_requests"] == 0
        and engine.budget["shared_cache_hits"] == 1
    )
    checks.append(
        {
            "task": task,
            "qid": row["qid"],
            "passed": True,
            "actual_new_requests": 0,
            "shared_hits": engine.budget["shared_cache_hits"],
            "initializer_protocol": config["initial_protocol"],
        }
    )
    engine.shared.close()
atomic_json(
    ARTIFACTS / "checks/initial_cache_compatibility_v2.json",
    {
        "passed": True,
        "checks": checks,
        "scope": "initializer pixels/messages/model/sampling/output protocol unchanged; controllers are not shared across versions",
    },
)
print(json.dumps(checks))
