"""Live image binding and forced interfaces; never an autonomous effectiveness result."""

import _bootstrap  # noqa: F401
import base64
import json
import os
from pathlib import Path

from PIL import Image

from videoarm.qvhighlights import prompts
from videoarm.qvhighlights.backend import LocalBackend
from videoarm.qvhighlights.common import (
    ARTIFACTS,
    atomic_json,
    code_identity,
    digest,
    load_manifest,
)
from videoarm.qvhighlights.contracts import windows
from videoarm.qvhighlights.media import FrameReader, probe
from videoarm.qvhighlights.pipeline import Engine
from videoarm.qvhighlights.state import Store


def main():
    root = Path(os.getenv("QVH_DATA_ROOT", str(ARTIFACTS.parents[1] / "qvh_data")))
    c = json.loads((ARTIFACTS / "configs/default_v2.json").read_text())
    c.update(task="mr", method="M1")
    backend = LocalBackend(c)
    backend.verify()
    colors = ["red", "green", "blue", "yellow", "black", "white"] * 4
    directory = root / "probes/image_binding_v2"
    directory.mkdir(parents=True, exist_ok=True)
    frames = []
    for i, color in enumerate(colors):
        p = directory / f"{i}.jpg"
        if not p.exists():
            Image.new("RGB", (672, 448), color).save(p, quality=90)
        encoded = base64.b64encode(p.read_bytes()).decode()
        frames.append(
            {
                "image_id": digest(["binding", i, color]),
                "pts": float(i),
                "position": i,
                "clip_id": None,
                "path": str(p),
                "jpeg_sha256": digest(encoded),
                "url": "data:image/jpeg;base64," + encoded,
            }
        )
    row = {
        "qid": -1,
        "query": "Read each picture color in the supplied order",
        "vid": "probe_0.0_24.0",
        "duration": 24,
    }
    store = Store(directory, {"config": c, "code": code_identity()}, [row])
    store.recover()
    engine = Engine(
        row,
        c,
        store,
        backend,
        None,
        {
            "video_duration": 24,
            "media_sha256": digest(colors),
            "timeline_sha256": digest(list(range(24))),
        },
    )

    def parser(p):
        if not isinstance(p, dict) or set(p) != {"colors"} or p["colors"] != colors:
            raise ValueError("24-picture color/position binding failed")
        return p

    result = engine.request(
        "binding",
        "scene",
        'Return {"colors":["red",...]} for all 24 pictures in order. Allowed colors: red, green, blue, yellow, black, white.',
        frames,
        parser,
    )
    atomic_json(
        ARTIFACTS / "checks/live_image_binding.json",
        {
            "passed": True,
            "logical_frames": 24,
            "budget": engine.budget,
            "result": result,
            "interpretation": "color and position fixture only; not semantic or temporal localization accuracy",
        },
    )
    engine.shared.close()
    real = load_manifest(ARTIFACTS / "manifests/smoke8.jsonl")[0]
    info = probe(real["media_path"], real["duration"], root / "timelines")
    directory = root / "probes/forced_real_interfaces_v2"
    store = Store(
        directory, {"config": c, "code": code_identity(), "row": real}, [real]
    )
    store.recover()
    reader = FrameReader(real["media_path"], info)
    engine = Engine(real, c, store, backend, reader, info)
    try:
        initial = engine.initialize()
        for i, tool in enumerate(("scene_snapper", "clip_analyzer")):
            engine.observe(
                {
                    "action": tool,
                    "start": 0,
                    "end": min(30, real["duration"]),
                    "question": "Check the query event, persistence, repetition and boundaries in this interval.",
                },
                i,
            )
        if engine.state.get("invalid_tools"):
            raise ValueError("forced real visual interface output failed")
        final = engine.request(
            "probe_final",
            "final",
            prompts.FINAL + "\n" + json.dumps(engine.state.get("memory", [])),
            engine.current_images(initial),
            lambda p: windows(
                p["pred_relevant_windows"], real["duration"], canonicalize=True
            ),
            final=True,
        )
        atomic_json(
            ARTIFACTS / "checks/live_forced_interfaces.json",
            {
                "passed": True,
                "qid": real["qid"],
                "budget": engine.budget,
                "candidates": final,
                "interpretation": "forced Scene/Clip/final interface probe only; excluded from method scores",
            },
        )
    finally:
        reader.close()
        engine.shared.close()
    print(
        "Live binding and forced real interfaces passed; autonomous smoke still required"
    )


if __name__ == "__main__":
    main()
