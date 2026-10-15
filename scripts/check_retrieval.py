"""Exercise the real embedding model on completed, label-free smoke memories."""

import json
import time
from pathlib import Path

from merit_qvh.cli import frozen_run
from merit_qvh.core import rank_clips, read_jsonl, write_json
from merit_qvh.embedding import Embedder
from merit_qvh.pipeline import Pipeline

repo = Path(__file__).resolve().parents[1]
config = json.loads((repo / "configs/main_v1.json").read_text())
protocol_id, root = frozen_run(config)
audit = json.loads((root / "model_audit.json").read_text())
if "embedding" not in audit:
    raise RuntimeError("Embedding weights must pass audit-models first")
embedder = Embedder(config["embedding"], config["artifact_root"])
pipeline = Pipeline(config, protocol_id, root, None, embedder)
records = []
for row in read_jsonl(root / "manifests/smoke.jsonl"):
    path = root / "memories" / (row["vid"] + ".json")
    if not path.exists():
        continue
    started = time.monotonic()
    memory = json.loads(path.read_text())
    key_vectors = pipeline.index(memory)
    query_vector = embedder.encode([row["query"]])[0]
    ranked = rank_clips(key_vectors, query_vector, set(), config["retrieval"]["top_k"])
    record = {
        "qid": row["qid"],
        "vid": row["vid"],
        "query": row["query"],
        "key_shape": list(key_vectors.shape),
        "seconds": time.monotonic() - started,
        "top3": [
            {
                "clip_id": index,
                "start": memory["clips"][index]["start"],
                "end": memory["clips"][index]["end"],
                "maxsim": score,
                "caption": memory["clips"][index]["caption"],
            }
            for index, score in ranked
        ],
    }
    records.append(record)
    write_json(
        root / "checks/retrieval_smoke.json",
        {"protocol_id": protocol_id, "completed_videos": len(records), "results": records},
    )
    print(json.dumps({key: record[key] for key in ("qid", "vid", "key_shape", "seconds")}), flush=True)
if not records:
    raise RuntimeError("No completed smoke memories available")
