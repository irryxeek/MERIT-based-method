import json
import re
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from merit_qvh.cli import infer_query, inference_row, validate_submission
from merit_qvh.client import CachedClient
from merit_qvh.core import neighbors, rank_clips, validate_filter, validate_manifest, validate_scores, windows
from merit_qvh.media import frame_indices, observed_duration, sample_times
from merit_qvh.pipeline import Pipeline


def test_tail_and_vfr_sampling():
    assert windows(21.2, 10) == [(0, 10), (10, 20), (20, 21.2)]
    assert sample_times(20, 20.2, 1, 0.5) == [20.1]
    timestamps = [[0, 0.1], [0.1, 0.4], [0.4, 0.7], [0.7, 1.0]]
    assert frame_indices(timestamps, [0, 0.39, 0.4, 0.99]).tolist() == [0, 1, 2, 3]


def test_partial_terminal_bin_keeps_only_observed_media():
    assert observed_duration(128, 126.459) == 126.459
    assert windows(observed_duration(128, 126.459), 10)[-1] == (120, 126.459)
    assert sample_times(120, 126.459, 1, 0.5)[-1] == 125.5
    assert observed_duration(150, 150.016) == 150
    with pytest.raises(ValueError):
        observed_duration(128, 126)
    with pytest.raises(ValueError):
        observed_duration(150, 155)


def test_maxsim_distinct_temporal_ids_and_exclusion():
    vectors = np.array(
        [[[0, 1], [1, 0], [0, 1]], [[1, 0], [1, 0], [1, 0]], [[0, 1], [0, 1], [0, 1]]], dtype=float
    )
    # Identical highest similarities do not collapse different timestamps.
    assert rank_clips(vectors, np.array([1, 0]), set(), 3) == [(0, 1.0), (1, 1.0), (2, 0.0)]
    assert rank_clips(vectors, np.array([1, 0]), {0}, 1) == [(1, 1.0)]
    assert neighbors([0, 2], 3, 1) == [0, 1, 2]
    with pytest.raises(ValueError):
        rank_clips(vectors, np.array([np.nan, 0]), set(), 3)


@pytest.mark.parametrize(
    "scores", [[1], [1, True], [0, float("nan")], [0, float("inf")], [-1, 3], [101, 0], ["25", 1]]
)
def test_scores_reject_silent_coercion(scores):
    with pytest.raises(ValueError):
        validate_scores({"scores": scores}, 2)


def test_annotations_excluded_from_inference():
    truth = {
        "qid": 1,
        "vid": "video_0_20",
        "query": "a person opens a box",
        "duration": 20,
        "saliency_scores": [[4, 4, 4]],
        "relevant_clip_ids": [0],
        "relevant_windows": [[0, 2]],
    }
    row = inference_row(truth, "/videos")
    assert validate_manifest([row]) == [row]
    with pytest.raises(ValueError):
        validate_manifest([{**row, "saliency_scores": [[4, 4, 4]]}])
    with pytest.raises(ValueError):
        validate_manifest([row, row])


def test_evaluation_requires_full_denominator_and_lengths():
    truth = [{"qid": 1, "vid": "a", "duration": 4}, {"qid": 2, "vid": "b", "duration": 4}]
    predictions = [
        {"qid": 1, "vid": "a", "pred_saliency_scores": [3, 1]},
        {"qid": 2, "vid": "b", "pred_saliency_scores": [0, 0]},
    ]
    validate_submission(predictions, truth)
    for invalid in (
        predictions[:1],
        [predictions[0], predictions[0]],
        [predictions[0], {**predictions[1], "pred_saliency_scores": [0]}],
    ):
        with pytest.raises(ValueError):
            validate_submission(invalid, truth)


def test_evidence_cannot_invent_temporal_ids():
    with pytest.raises(ValueError):
        validate_filter({"evidence": [{"clip_ids": [9], "text": "opening"}]}, {0, 1})


def response(text):
    return {
        "choices": [{"finish_reason": "stop", "message": {"content": text}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
    }


def test_bounded_format_retry_and_persistent_cache(tmp_path, monkeypatch):
    config = {"model": "test", "temperature": 0, "format_retries": 1, "transport_retries": 1}
    client = CachedClient("http://unused/v1", config, tmp_path, "v1")
    outputs = iter([response('{"scores": [true]}'), response('{"scores": [25]}')])
    monkeypatch.setattr(client, "_post", lambda _: next(outputs))

    def validator(value):
        return validate_scores(value, 1)

    assert client.complete("score", [], 32, validator) == [25]
    assert client.complete("score", [], 32, validator) == [25]
    record = json.loads(next(tmp_path.glob("*/*.json")).read_text())
    assert len(record["attempts"]) == 2
    assert record["status"] == "ok"
    assert client.events[-1]["cache_hit"]


def test_exhausted_failure_is_not_repeated_on_resume(tmp_path, monkeypatch):
    config = {"model": "test", "temperature": 0, "format_retries": 1, "transport_retries": 0}
    client = CachedClient("http://unused/v1", config, tmp_path, "v1")
    calls = []

    def fail(_):
        calls.append(1)
        return response("invalid JSON")

    monkeypatch.setattr(client, "_post", fail)
    for _ in range(2):
        with pytest.raises(RuntimeError):
            client.complete("keys", [], 32, lambda value: value)
    assert len(calls) == 2


class FakeEmbedder:
    def encode(self, texts):
        return np.tile([1.0, 0.0], (len(texts), 1))


class FakeClient:
    def __init__(self, fail_batch=False):
        self.events = []
        self.fail_batch = fail_batch
        self.score_calls = 0

    def complete(self, stage, content, max_tokens, validator=None):
        if stage == "filter":
            return [{"clip_ids": [0], "text": "person opening box"}]
        if stage == "decide":
            return {"sufficient": False, "search_query": "box opened"}
        if stage == "score":
            self.score_calls += 1
            if self.fail_batch and self.score_calls == 2:
                raise RuntimeError("format attempts exhausted")
            count = int(re.search(r"Return exactly (\d+) scores", content[0]["text"])[1])
            return validator({"scores": [75.0] * count})
        raise AssertionError(stage)


@pytest.mark.parametrize("fail_batch", [False, True])
def test_dense_scoring_resume_and_whole_query_failure(tmp_path, fail_batch):
    config = json.loads((Path(__file__).parents[1] / "configs/main_v1.json").read_text())
    client = FakeClient(fail_batch)
    pipeline = Pipeline(config, "test_protocol", tmp_path, client, FakeEmbedder())
    row = {
        "qid": 7,
        "vid": "fake",
        "query": "opening a box",
        "duration": 41.5,
        "video_path": "/unused/fake.mp4",
    }
    memory = {
        "identity": "fake_memory",
        "clips": [
            {
                "clip_id": i,
                "start": i * 10,
                "end": min((i + 1) * 10, 41.5),
                "frames": [],
                "caption": "person opens box",
                "keys": {"event": "opens", "object": "box", "summary": "opening"},
            }
            for i in range(5)
        ],
    }
    result = pipeline.infer(row, memory, pipeline.index(memory))
    assert result["status"] == ("failed" if fail_batch else "ok")
    assert result["pred_saliency_scores"] == ([0.0] if fail_batch else [75.0]) * 20
    assert [r["anchors"] for r in result["rounds"]] == [[0, 1, 2], [3, 4]]
    assert len({i for r in result["rounds"] for i in r["anchors"]}) == 5
    assert pipeline.infer(row, memory, pipeline.index(memory)) == result
    assert client.score_calls == 2
    with pytest.raises(ValueError, match="identity changed"):
        pipeline.infer({**row, "query": "other query"}, memory, pipeline.index(memory))


@pytest.mark.parametrize("failure_stage", ["memory", "index"])
def test_preparation_failure_keeps_query_and_resume_does_not_retry(tmp_path, monkeypatch, failure_stage):
    config = {"scoring": {"clip_seconds": 2}}
    video = tmp_path / "input.mp4"
    video.write_bytes(b"fixed video input")
    row = {"qid": 1, "vid": "v", "query": "box", "duration": 10, "video_path": str(video)}
    client = FakeClient()
    pipeline = Pipeline(config, "test", tmp_path, client, FakeEmbedder())
    calls = []

    def fail(_):
        calls.append(1)
        client.events.append({"request_id": "failed_request", "stage": "caption"})
        raise RuntimeError("Incomplete generation: length")

    monkeypatch.setattr(pipeline, "memory", fail if failure_stage == "memory" else lambda _: {})
    monkeypatch.setattr(pipeline, "index", fail if failure_stage == "index" else lambda _: [])
    failed = infer_query(pipeline, row)
    assert failed["failure_stage"] == failure_stage
    assert failed["status"] == "failed"
    assert failed["pred_saliency_scores"] == [0.0] * 5
    assert failed["requests"][0]["request_id"] == "failed_request"
    assert infer_query(pipeline, row) == failed
    assert len(calls) == 1
    with pytest.raises(ValueError, match="identity changed"):
        infer_query(pipeline, {**row, "query": "changed"})
    monkeypatch.setattr(pipeline, "memory", lambda _: {})
    monkeypatch.setattr(pipeline, "index", lambda _: [])
    monkeypatch.setattr(pipeline, "infer", lambda row, memory, vectors: {"qid": row["qid"], "status": "ok"})
    assert infer_query(pipeline, {**row, "qid": 2}) == {"qid": 2, "status": "ok"}


def test_evaluation_includes_query_whose_memory_failed(tmp_path, monkeypatch):
    from merit_qvh import cli
    from merit_qvh.core import write_json, write_jsonl

    truth = [{"qid": 1, "vid": "failed", "duration": 4}, {"qid": 2, "vid": "ok", "duration": 4}]
    write_jsonl(tmp_path / "ground_truth/val.jsonl", truth)
    for row in truth:
        write_json(
            tmp_path / "queries" / f"{row['qid']}.json",
            {
                **row,
                "status": "failed" if row["qid"] == 1 else "ok",
                "failure_stage": "memory" if row["qid"] == 1 else None,
                "pred_saliency_scores": [0, 0] if row["qid"] == 1 else [20, 80],
                "rounds": [],
                "seconds": 1.0,
            },
        )
    write_json(tmp_path / "memories/ok.json", {"requests": []})
    evaluator = ModuleType("standalone_eval.eval")

    def check_denominator(predictions, labels, **kwargs):
        assert len(predictions) == len(labels) == 2
        assert predictions[0]["pred_saliency_scores"] == [0, 0]
        assert kwargs["match_number"] is True
        return {"test_only": True}

    evaluator.eval_submission = check_denominator
    monkeypatch.setitem(sys.modules, "standalone_eval.eval", evaluator)
    monkeypatch.setattr(cli, "check_revisions", lambda _: None)
    report = cli.evaluate({"evaluator_root": str(tmp_path)}, tmp_path, "val")
    assert report["queries"] == 2
    assert report["failed_qids"] == [1]
    assert report["videos_without_complete_memory"] == 1
    assert report["failure_stage_counts"] == {"memory": 1}
