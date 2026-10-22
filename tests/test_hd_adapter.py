"""Behavioral checks for the isolated dense-HD adapter (CPU, no model service)."""

import copy
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from hd_adapter import pipeline
from hd_adapter.client import Client
from hd_adapter.media import FrameReader
from hd_adapter.pipeline import local_batches, score_stage, trace_complete, validate_hd
from hd_adapter.prepare import select_rows, source_id
from PIL import Image

from merit_qvh.client import CachedClient
from merit_qvh.core import write_json


def response(text):
    return {
        "choices": [{"finish_reason": "stop", "message": {"content": text}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }


def vlm_config():
    return {"model": "test", "temperature": 0, "format_retries": 0, "transport_retries": 0}


def test_hd_rejects_incomplete_misbound_or_out_of_range_scores():
    value = {"clip_scores": [{"clip_id": 2, "score": 3.0}], "events": "visible"}
    assert validate_hd(value, [2]) == value
    for score in (True, -1, 4.01, float("nan"), float("inf"), "3"):
        bad = copy.deepcopy(value)
        bad["clip_scores"][0]["score"] = score
        with pytest.raises(ValueError):
            validate_hd(bad, [2])
    with pytest.raises(ValueError):
        validate_hd(value, [3])
    with pytest.raises(ValueError):
        validate_hd(value, [2, 3])


def test_anchor_mapping_is_time_ordered_nonduplicating_and_tail_safe():
    memory = {"clips": [{"start": 0, "end": 10}, {"start": 8, "end": 20}, {"start": 20, "end": 23.4}]}
    assert local_batches(memory, [2, 0, 1], 12) == [list(range(12))]
    assert local_batches(memory, [2], 12) == [[10, 11]]
    assert local_batches(memory, [], 12) == []


def test_representative_frame_is_nearest_real_pts_within_each_bin():
    reader = FrameReader.__new__(FrameReader)
    reader.info = {"pts": [0, 0.9, 1.2, 2.01, 3.01, 4.01], "video_duration": 4.1, "media_sha256": "test"}
    reader.jpeg_cache = {}
    reader.at = lambda t: Image.new("RGB", (900, 600))
    frames = reader.sample(0, 6, clip_ids=[0, 1, 2])
    assert [f["pts"] for f in frames] == [0.9, 3.01, 4.01]
    assert [f["clip_id"] for f in frames] == [0, 1, 2]
    with pytest.raises(ValueError, match="no frame"):
        reader.sample(0, 8, clip_ids=[0, 3])


def test_local_rescoring_preserves_all_other_scores_and_pixels():
    observed = []

    class FakeClient:
        def complete(self, stage, content, tokens, validator, **kwargs):
            observed.extend(content)
            return validator({"clip_scores": [{"clip_id": 1, "score": 4}], "events": "local"})

    initial = [0.0, 1.0, 2.0]
    frames = [
        {"clip_id": i, "pts": i * 2 + 1, "position": i, "url": f"image-{i}", "jpeg_sha256": f"hash-{i}"}
        for i in range(3)
    ]
    row = {"qid": 1, "vid": "source_0_6", "duration": 6, "query": "query"}
    scores, records = score_stage(
        FakeClient(), row, {"video_duration": 6, "mapping_audit": {}}, frames, [[1]], initial=initial
    )
    assert scores == [0.0, 4, 2.0]
    assert initial == [0.0, 1.0, 2.0]
    assert [p["image_url"]["url"] for p in observed if p["type"] == "image_url"] == ["image-1"]
    assert records[0]["frame_hashes"] == ["hash-1"]


def test_request_lock_deduplicates_without_sharing_query_events(tmp_path, monkeypatch):
    calls = []
    barrier = threading.Barrier(2)

    def post(self, payload):
        calls.append(payload)
        time.sleep(0.03)
        return response("done")

    monkeypatch.setattr(Client, "_post", post)
    clients = [Client("http://localhost", vlm_config(), tmp_path, "protocol") for _ in range(2)]

    def run(client):
        barrier.wait()
        return client.complete("caption", [{"type": "text", "text": "same"}], 20)

    with ThreadPoolExecutor(2) as pool:
        assert list(pool.map(run, clients)) == ["done", "done"]
    assert len(calls) == 1
    assert clients[0].events is not clients[1].events
    assert sorted(c.events[0]["cache_hit"] for c in clients) == [False, True]


def test_legacy_request_payload_matches_original(tmp_path, monkeypatch):
    payloads = []

    def post(self, payload):
        payloads.append(copy.deepcopy(payload))
        return response("description")

    monkeypatch.setattr(CachedClient, "_post", post)
    for cls, name in ((CachedClient, "original"), (Client, "new")):
        client = cls("http://localhost", vlm_config(), tmp_path / name, "same")
        client.complete("caption", [{"type": "text", "text": "question"}], 384)
    assert payloads[0] == payloads[1]


def test_interrupted_response_recovered_and_terminal_failure_not_rerun(tmp_path, monkeypatch):
    client = Client("http://localhost", vlm_config(), tmp_path, "p")
    calls = []
    monkeypatch.setattr(client, "_post", lambda p: calls.append(p) or response("ok"))
    client.complete("stage", [], 3)
    path = next(tmp_path.glob("*/*.json"))
    record = json.loads(path.read_text())
    record["status"] = "pending"
    write_json(path, record)
    assert client.complete("stage", [], 3) == "ok"
    assert len(calls) == 1
    record["status"] = "failed"
    write_json(path, record)
    with pytest.raises(RuntimeError, match="Previously failed"):
        client.complete("stage", [], 3)
    assert len(calls) == 1


def test_sampling_is_label_independent_and_source_disjoint():
    train = [
        {"qid": i * 2 + j, "vid": f"source{i}_0_10", "label": i % 3} for i in range(12) for j in range(2)
    ]
    selected = select_rows(train, {"source0", "source1"}, count=4)
    a, b = ({source_id(r) for r in selected[s]} for s in ("explore", "confirm"))
    assert len(a) == len(b) == 4
    assert not a & b
    assert not (a | b) & {"source0", "source1"}
    flipped = [{**r, "label": 999} for r in train]
    again = select_rows(flipped, {"source0", "source1"}, count=4)
    assert {s: [r["qid"] for r in rs] for s, rs in selected.items()} == {
        s: [r["qid"] for r in rs] for s, rs in again.items()
    }


@pytest.mark.parametrize("failure", ["memory", "score", "initial", None])
def test_group_failures_do_not_erase_independent_results(tmp_path, monkeypatch, failure):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fixture")
    row = {"qid": 7, "vid": "source_0_10", "duration": 10, "query": "action", "video_path": str(video)}
    config = {"vlm": vlm_config()}
    frames = [{"jpeg_sha256": f"h{i}"} for i in range(5)]
    monkeypatch.setattr(pipeline, "dense_frames", lambda *a: ({}, frames))
    monkeypatch.setattr(pipeline, "SharedPipeline", lambda *a: None)

    def infer(*args):
        write_json(tmp_path / "legacy/memories/source_0_10.json", {"clips": [{"start": 0, "end": 4}]})
        return {
            **row,
            "status": "failed" if failure in ("memory", "score") else "ok",
            "pred_saliency_scores": [0.0] * 5,
            "selected_clip_ids": [0],
            "rounds": [],
            "requests": [] if failure == "memory" else [{"stage": "score"}],
        }

    def score(client, row, info, frames, batches, *, initial=None):
        if initial is None and failure == "initial":
            raise RuntimeError("initial failed")
        scores = [1.0] * 5 if initial is None else list(initial)
        if initial is not None:
            for ids in batches:
                for i in ids:
                    scores[i] = 4.0
        return scores, []

    monkeypatch.setattr(pipeline, "infer_query", infer)
    monkeypatch.setattr(pipeline, "score_stage", score)
    result = pipeline.run_query(config, "p", tmp_path, "unused", None, row)
    assert result["D0"]["status"] == ("failed" if failure == "initial" else "ok")
    assert result["H1"]["status"] == ("failed" if failure in ("initial", "memory") else "ok")
    if failure not in ("initial", "memory"):
        assert result["H1"]["pred_saliency_scores"] == [4.0, 4.0, 1.0, 1.0, 1.0]
    assert pipeline.run_query(config, "p", tmp_path, "unused", None, row) == result
    video.write_bytes(b"changed")
    with pytest.raises(ValueError, match="identity changed"):
        pipeline.run_query(config, "p", tmp_path, "unused", None, row)


def test_scoring_failure_can_supply_trace_but_partial_retrieval_cannot():
    assert trace_complete({"status": "failed", "requests": [{"stage": "score"}]})
    assert not trace_complete({"status": "failed", "requests": [{"stage": "decide"}]})


def test_gpu_guard_refuses_changed_or_occupied_assignment(monkeypatch):
    from hd_adapter import service

    config = {"hd": {"server": {"tensor_parallel_size": 2, "gpu_uuids": ["gpu-four", "gpu-five"]}}}
    monkeypatch.setattr(
        service.subprocess, "check_output", lambda *a, **k: "4, gpu-four, 15\n5, gpu-five, 15\n"
    )
    service.guard(config)
    monkeypatch.setattr(
        service.subprocess, "check_output", lambda *a, **k: "4, gpu-four, 9000\n5, gpu-five, 15\n"
    )
    with pytest.raises(RuntimeError, match="occupied"):
        service.guard(config)
    monkeypatch.setattr(service.subprocess, "check_output", lambda *a, **k: "4, wrong, 15\n5, gpu-five, 15\n")
    with pytest.raises(RuntimeError, match="absent"):
        service.guard(config)
