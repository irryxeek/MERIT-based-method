"""D0 dense scoring and H1 rescoring around the unchanged MERIT M0 pipeline."""

import base64
import json
import math
import threading
import time
from pathlib import Path

from merit_qvh.cli import infer_query
from merit_qvh.core import digest, file_sha256, write_json
from merit_qvh.embedding import Embedder
from merit_qvh.pipeline import Pipeline

from .client import Client, locked
from .media import FrameReader, mapping, probe, visual_message
from .prompts import HD, SYSTEM


class SharedEmbedder:
    """One model and one encode at a time, even when query workers overlap."""

    def __init__(self, config, root):
        self.inner = Embedder(config, root)
        self.lock = threading.Lock()
        self.local = threading.local()

    def bind(self, qid):
        self.local.events = []
        self.local.qid = qid
        return self.local.events

    def encode(self, texts):
        started = time.monotonic()
        with self.lock:
            cache_id = digest({"config": self.inner.config, "texts": texts})
            cache_hit = (self.inner.root / (cache_id + ".npy")).exists()
            tick = time.monotonic()
            result = self.inner.encode(texts)
            if hasattr(self.local, "events"):
                self.local.events.append(
                    {
                        "qid": self.local.qid,
                        "identity": cache_id,
                        "cache_hit": cache_hit,
                        "texts": len(texts),
                        "queue_seconds": tick - started,
                        "encode_seconds": time.monotonic() - tick,
                    }
                )
            return result


class SharedPipeline(Pipeline):
    def memory(self, row):
        with locked(self.root / "locks" / (row["vid"] + ".lock")):
            return super().memory(row)


def validate_hd(value, ids):
    if not isinstance(value, dict) or set(value) != {"clip_scores", "events"}:
        raise ValueError("HD requires exactly clip_scores and events")
    if not isinstance(value["events"], str) or len(value["events"]) > 2000:
        raise ValueError("Invalid HD events")
    rows = value["clip_scores"]
    if not isinstance(rows, list) or len(rows) != len(ids):
        raise ValueError("HD coverage mismatch")
    for row, expected in zip(rows, ids, strict=True):
        if not isinstance(row, dict) or set(row) != {"clip_id", "score"}:
            raise ValueError("Invalid HD row")
        if type(row["clip_id"]) is not int or row["clip_id"] != expected:
            raise ValueError("HD IDs/order mismatch")
        score = row["score"]
        if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 4:
            raise ValueError("HD score must be finite and in [0,4]")
    return value


def local_batches(memory, selected, n_bins):
    """Union of intersecting bins, ascending time, each visited once, max 24/batch."""
    ids = sorted(
        {
            i
            for anchor in selected
            for i in range(n_bins)
            if 2 * i < memory["clips"][anchor]["end"] and 2 * (i + 1) > memory["clips"][anchor]["start"]
        }
    )
    return [ids[start : start + 24] for start in range(0, len(ids), 24)]


def dense_frames(row, root):
    with locked(root / "locks" / (row["vid"] + "-hd.lock")):
        info = probe(row["video_path"], row["duration"], root / "media_probes")
        n = int(row["duration"] / 2)
        identity = digest([info["media_sha256"], info["timeline_sha256"], n, 672, 90])
        directory = root / "hd_frames" / identity
        path = directory / "frames.json"
        if path.exists():
            saved = json.loads(path.read_text())
            frames = saved["frames"]
        else:
            reader = FrameReader(row["video_path"], info)
            try:
                frames = mapping(reader.sample(0, 2 * n, clip_ids=list(range(n)), output_dir=directory))
            finally:
                reader.close()
            write_json(path, {"identity": identity, "frames": frames})
        for frame in frames:
            encoded = base64.b64encode(Path(frame["path"]).read_bytes()).decode()
            if digest(encoded) != frame["jpeg_sha256"]:
                raise ValueError("Representative frame hash drift")
            frame["url"] = "data:image/jpeg;base64," + encoded
        return info, frames


def hd_content(row, info, frames, ids, *, local=False):
    context = {
        "qid": row["qid"],
        "query": row["query"],
        "vid": row["vid"],
        "duration": row["duration"],
        "task": "hd",
        "available_media_duration": info["video_duration"],
        "mapping_audit": info["mapping_audit"],
    }
    instruction = HD + f"\nExpected IDs: {ids}"
    if local:
        # No previous scores or captions: control the evidence to the same pixels.
        instruction += "\n" + json.dumps(
            {
                "requested_clip_ids": ids,
                "question": row["query"],
                "selection": "MERIT retrieved memory anchors; score only the supplied clip IDs",
            },
            ensure_ascii=False,
        )
    selected = [{**frames[i], "position": pos} for pos, i in enumerate(ids)]
    return visual_message(json.dumps(context, ensure_ascii=False) + "\n" + instruction, selected)["content"]


def score_stage(client, row, info, frames, batches, *, initial=None):
    scores = [0.0] * int(row["duration"] / 2) if initial is None else list(initial)
    records = []
    for ids in batches:
        result = client.complete(
            "hd_initialize" if initial is None else "hd_local",
            hd_content(row, info, frames, ids, local=initial is not None),
            2048,
            lambda value, expected=ids: validate_hd(value, expected),
            system=SYSTEM,
        )
        for item in result["clip_scores"]:
            scores[item["clip_id"]] = item["score"]
        records.append(
            {"ids": ids, "response": result, "frame_hashes": [frames[i]["jpeg_sha256"] for i in ids]}
        )
    return scores, records


def trace_complete(m0):
    # The original pipeline enters scoring only after the retrieval loop completes.
    # A scoring failure can still provide a valid shared retrieval trace for H1.
    return any(e["stage"] == "score" for e in m0.get("requests", []))


def run_query(config, protocol_id, root, endpoint, embedder, row):
    root = Path(root)
    output = root / "queries" / (str(row["qid"]) + ".json")
    with locked(root / "query_locks" / (str(row["qid"]) + ".lock")):
        identity = digest([protocol_id, row, file_sha256(row["video_path"])])
        if output.exists():
            saved = json.loads(output.read_text())
            if saved["identity"] != identity:
                raise ValueError("Query/source identity changed")
            return saved
        started = time.monotonic()
        embedding_events = embedder.bind(row["qid"]) if hasattr(embedder, "bind") else []
        client = Client(endpoint, config["vlm"], root / "requests", protocol_id)
        n = int(row["duration"] / 2)
        base = {"qid": row["qid"], "vid": row["vid"], "query": row["query"]}
        d0_path = root / "initial" / (str(row["qid"]) + ".json")
        info, frames = None, None
        if d0_path.exists():
            d0 = json.loads(d0_path.read_text())
            if d0["identity"] != identity:
                raise ValueError("Initialization identity changed")
        else:
            tick = time.monotonic()
            d0 = {**base, "identity": identity}
            try:
                info, frames = dense_frames(row, root)
                batches = [list(range(i, min(i + 24, n))) for i in range(0, n, 24)]
                scores, records = score_stage(client, row, info, frames, batches)
                d0.update(status="ok", pred_saliency_scores=scores, batches=records)
            except (RuntimeError, ValueError, OSError) as error:
                d0.update(status="failed", error=str(error), pred_saliency_scores=[0.0] * n)
            d0.update(seconds=time.monotonic() - tick, requests=list(client.events))
            write_json(d0_path, d0)
        tick = time.monotonic()
        legacy = SharedPipeline(config, protocol_id, root / "legacy", client, embedder)
        m0 = infer_query(legacy, row)
        legacy_seconds = time.monotonic() - tick
        tick = time.monotonic()
        first_event = len(client.events)
        h1 = dict(base)
        try:
            if d0["status"] != "ok":
                raise RuntimeError("D0 initialization failed")
            if not trace_complete(m0):
                raise RuntimeError("MERIT retrieval did not complete")
            memory = json.loads((root / "legacy" / "memories" / (row["vid"] + ".json")).read_text())
            if frames is None:
                info, frames = dense_frames(row, root)
            batches = local_batches(memory, m0["selected_clip_ids"], n)
            scores, records = score_stage(
                client, row, info, frames, batches, initial=d0["pred_saliency_scores"]
            )
            h1.update(
                status="ok",
                pred_saliency_scores=scores,
                batches=records,
                updated_clip_ids=[i for ids in batches for i in ids],
            )
        except (RuntimeError, ValueError, OSError) as error:
            h1.update(status="failed", error=str(error), pred_saliency_scores=[0.0] * n)
        h1.update(seconds=time.monotonic() - tick, requests=client.events[first_event:])
        result = {
            **base,
            "identity": identity,
            "protocol_id": protocol_id,
            "M0": m0,
            "D0": d0,
            "H1": h1,
            "retrieval_trace_sha256": digest(m0.get("rounds", [])),
            "legacy_memory_index_query_seconds_this_session": legacy_seconds,
            "wall_seconds_this_session": time.monotonic() - started,
            "finished_at": time.time(),
            "requests_this_session": client.events,
            "embedding_events_this_session": embedding_events,
        }
        write_json(output, result)
        return result
