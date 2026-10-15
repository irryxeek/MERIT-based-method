"""MERIT retrieval and the explicitly separate dense highlight scoring adapter."""

import json
import time
from pathlib import Path

from . import prompts
from .client import image_part, text_part
from .core import (
    digest,
    neighbors,
    rank_clips,
    validate_decision,
    validate_filter,
    validate_keys,
    validate_scores,
    write_json,
)
from .media import prepare_frames


def visual_content(clips):
    content = []
    for clip in sorted(clips, key=lambda c: c["clip_id"]):
        content.append(text_part(f"Memory clip {clip['clip_id']}: [{clip['start']}, {clip['end']}) seconds."))
        for frame in clip["frames"]:
            content.append(text_part(f"Frame at {frame['time']:.3f} seconds."))
            content.append(image_part(frame["path"]))
    return content


def captions_only(clips):
    return [{key: clip[key] for key in ("clip_id", "start", "end", "caption")} for clip in clips]


class Pipeline:
    def __init__(self, config, protocol_id, run_root, client, embedder):
        self.config = config
        self.protocol_id = protocol_id
        self.root = Path(run_root)
        self.client = client
        self.embedder = embedder

    def memory(self, row):
        path = self.root / "memories" / (row["vid"] + ".json")
        media = prepare_frames(row, self.config["memory"], self.config["artifact_root"])
        identity = digest({"protocol": self.protocol_id, "media": media["identity"]})
        if path.exists():
            result = json.loads(path.read_text())
            if result["identity"] != identity:
                raise ValueError("Memory identity changed; start a new frozen run")
            return result
        start_event = len(self.client.events)
        started = time.monotonic()
        clips = []
        for clip in media["clips"]:
            caption = self.client.complete(
                "caption",
                [text_part(prompts.CAPTION)] + visual_content([clip]),
                self.config["memory"]["caption_max_tokens"],
            )
            keys = self.client.complete(
                "keys",
                [text_part(prompts.KEYS + "\nDESCRIPTION:\n" + caption)],
                self.config["memory"]["keys_max_tokens"],
                validate_keys,
            )
            clips.append({**clip, "caption": caption, "keys": keys})
        result = {
            "identity": identity,
            "protocol_id": self.protocol_id,
            "vid": row["vid"],
            "source_sha256": media["source_sha256"],
            "clips": clips,
            "seconds": time.monotonic() - started,
            "requests": self.client.events[start_event:],
        }
        write_json(path, result)
        return result

    def index(self, memory):
        clips = memory["clips"]
        texts = [clip["keys"][key] for clip in clips for key in self.config["memory"]["key_types"]]
        vectors = self.embedder.encode(texts)
        return vectors.reshape(len(clips), len(self.config["memory"]["key_types"]), -1)

    def infer(self, row, memory, key_vectors):
        identity = digest({"protocol": self.protocol_id, "memory": memory["identity"], "input": row})
        path = self.root / "queries" / (str(row["qid"]) + ".json")
        if path.exists():
            result = json.loads(path.read_text())
            if result["identity"] != identity:
                raise ValueError("Query identity changed; start a new frozen run")
            return result
        started = time.monotonic()
        first_event = len(self.client.events)
        clips = memory["clips"]
        selected = set()
        evidence = []
        rounds = []
        batches = []
        search_query = row["query"]
        n_bins = int(row["duration"] / self.config["scoring"]["clip_seconds"])
        result = {
            "identity": identity,
            "protocol_id": self.protocol_id,
            "qid": row["qid"],
            "vid": row["vid"],
            "query": row["query"],
            "status": "pending",
            "rounds": rounds,
            "batches": batches,
        }
        try:
            for round_id in range(self.config["retrieval"]["max_rounds"]):
                query_vector = self.embedder.encode([search_query])[0]
                ranked = rank_clips(key_vectors, query_vector, selected, self.config["retrieval"]["top_k"])
                if not ranked:
                    break
                anchors = [index for index, _ in ranked]
                selected.update(anchors)
                neighbor_ids = neighbors(anchors, len(clips), self.config["retrieval"]["neighbor_radius"])
                filtered = self.client.complete(
                    "filter",
                    [
                        text_part(
                            prompts.FILTER
                            + "\nORIGINAL QUERY: "
                            + row["query"]
                            + "\nCAPTIONS:\n"
                            + json.dumps(captions_only([clips[i] for i in neighbor_ids]))
                        )
                    ],
                    self.config["retrieval"]["filter_max_tokens"],
                    lambda value, allowed=set(neighbor_ids): validate_filter(value, allowed),
                )
                evidence.extend(filtered)
                record = {
                    "round": round_id + 1,
                    "search_query": search_query,
                    "anchors": anchors,
                    "maxsim_scores": [score for _, score in ranked],
                    "neighbors": neighbor_ids,
                    "filtered_evidence": filtered,
                }
                rounds.append(record)
                decision = self.client.complete(
                    "decide",
                    [
                        text_part(
                            prompts.DECIDE
                            + "\nORIGINAL QUERY: "
                            + row["query"]
                            + "\nRETRIEVED CAPTIONS:\n"
                            + json.dumps(captions_only([clips[i] for i in sorted(selected)]))
                            + "\nFILTERED EVIDENCE:\n"
                            + json.dumps(evidence)
                        )
                    ]
                    + visual_content([clips[i] for i in sorted(selected)]),
                    self.config["retrieval"]["decision_max_tokens"],
                    validate_decision,
                )
                record["decision"] = decision
                if decision["sufficient"]:
                    break
                search_query = decision["search_query"]
            context = (
                prompts.SCORE
                + "\nORIGINAL QUERY: "
                + row["query"]
                + "\nWHOLE-VIDEO CAPTIONS:\n"
                + json.dumps(captions_only(clips))
                + "\nFILTERED EVIDENCE:\n"
                + json.dumps(evidence)
            )
            images = visual_content([clips[i] for i in sorted(selected)])
            scores = []
            for offset in range(0, n_bins, self.config["scoring"]["batch_clips"]):
                count = min(self.config["scoring"]["batch_clips"], n_bins - offset)
                intervals = [
                    {"bin_id": i, "start": 2 * i, "end": 2 * (i + 1)} for i in range(offset, offset + count)
                ]
                batch_scores = self.client.complete(
                    "score",
                    [
                        text_part(
                            context
                            + "\nREQUESTED INTERVALS:\n"
                            + json.dumps(intervals)
                            + f"\nReturn exactly {count} scores."
                        )
                    ]
                    + images,
                    self.config["scoring"]["max_tokens"],
                    lambda value, size=count: validate_scores(value, size),
                )
                scores.extend(batch_scores)
                batches.append({"offset": offset, "scores": batch_scores})
                write_json(path.with_suffix(".partial.json"), result)
            validate_scores({"scores": scores}, n_bins)
            result.update(status="ok", pred_saliency_scores=scores)
        except (RuntimeError, ValueError, OSError) as error:
            # Retain every qid in metrics. A failed generation cannot look like a valid all-zero answer.
            result.update(
                status="failed", error=f"{type(error).__name__}: {error}", pred_saliency_scores=[0.0] * n_bins
            )
        result.update(
            seconds=time.monotonic() - started,
            selected_clip_ids=sorted(selected),
            unique_selected_frames=sum(len(clips[i]["frames"]) for i in selected),
            requests=self.client.events[first_event:],
        )
        write_json(path, result)
        return result
