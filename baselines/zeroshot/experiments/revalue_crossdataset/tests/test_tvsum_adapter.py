"""CPU regression tests for leakage, evidence reuse and native metric parity."""

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1] / "tvsum_adapter.py"
SPEC = importlib.util.spec_from_file_location("tested_tvsum_crossdataset_adapter", PATH)
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class FakeStream:
    calls = []

    def __init__(self, row):
        self.row = row
        self.segments = [[i, i + 3] for i in range(0, 12, 3)]

    def describe(self):
        return {"vid": self.row["vid"], "segments": self.segments}

    def read(self, start, size):
        self.calls.append((self.row["vid"], start, size))
        clips, indices, requested, timestamps, originals = [], [], [], [], []
        for window in range(start, start + size):
            frames = np.stack([np.full((2, 2, 3), 6 * window + i, dtype=np.uint8) for i in range(6)])
            clips.append({"video": frames, "rgb_sha256": hashlib.sha256(frames.tobytes()).hexdigest()})
            indices.append(list(range(6 * window, 6 * window + 6)))
            requested.append([3 * window + i / 2 for i in range(6)])
            timestamps.append([[t, t + 0.5] for t in requested[-1]])
            originals.append(frames)
        trace = {
            "frame_indices": indices,
            "requested_times": requested,
            "frame_timestamps": timestamps,
            "raw_decoder_frame_timestamps": timestamps,
            "original_height_width": [2, 2],
            "resized_height_width": [2, 2],
            "sampled_original_rgb_sha256": hashlib.sha256(np.concatenate(originals).tobytes()).hexdigest(),
            "model_rgb_sha256": [c["rgb_sha256"] for c in clips],
            "model_input_frames": size * 6,
        }
        return clips, trace


@pytest.fixture
def baseline(tmp_path, monkeypatch):
    root = tmp_path / "baseline"
    modules = adapter._modules()
    original_modules = adapter._modules
    monkeypatch.setattr(adapter, "_modules", lambda: SimpleNamespace(**{**vars(original_modules()), "media": SimpleNamespace(StreamVideo=FakeStream)}))
    ids = adapter.val_ids()
    plan = {"umt_val_ids": ids, "selected_ids": ids, "shards": []}
    aggregate = []
    for part in range(2):
        shard = root / f"shard_{part}"
        part_ids = ids[part::2]
        plan["shards"].append({"directory": shard.name})
        rows, timelines = [], {}
        for vid in part_ids:
            video = root / "media" / f"{vid}.mp4"
            video.parent.mkdir(parents=True, exist_ok=True)
            video.write_bytes(b"fake media fixture, decoder mocked")
            row = {
                "qid": vid, "vid": vid, "query": adapter.QUERY, "duration": 12.0,
                "n_seconds": 12, "video_path": str(video), "media_sha256": adapter.sha256(video),
            }
            rows.append(row)
            timelines[vid] = {"decoded_timeline": FakeStream(row).describe()}
        write(shard / "media_verification.json", timelines)
        inference = shard / "inference.jsonl"
        inference.write_text("".join(json.dumps(row) + "\n" for row in rows))
        labels = {vid: {"anno": [[score] * 20 for score in [1, 2, 3, 4, 5, 2]], "title": "SECRET TITLE"} for vid in part_ids}
        write(shard / "evaluation_inputs/tvsum_anno.json", labels)
        binding = {
            "selected_qids": part_ids,
            "inference_sha256": adapter.sha256(inference),
            "media_verification_sha256": adapter.sha256(shard / "media_verification.json"),
            "source_sha256": {n: adapter.sha256(adapter.TVSUM / n) for n in adapter.BASELINE_SOURCES},
            "labels_sha256": {"tvsum_anno.json": adapter.sha256(shard / "evaluation_inputs/tvsum_anno.json")},
            "batch_size": 2,
        }
        write(shard / "binding.json", binding)
        bh = adapter.sha256(shard / "binding.json")
        for row in rows:
            vid = row["vid"]
            stream = FakeStream(row)
            write(shard / "media" / f"{vid}.json", stream.describe())
            records = []
            for p in (0.1, 0.4, 0.7, 0.9):
                raw = {9454: math.log(p), 2753: math.log(1 - p)}
                probability, mode = modules.core.extract_pyes(raw, 9454, 2753)
                records.append({
                    "pyes": probability, "score_mode": mode, "prompt_tokens": 10,
                    "top20": [{"id": k, "logprob": v} for k, v in raw.items()],
                })
            for start in (0, 2):
                _, trace = stream.read(start, 2)
                write(shard / "chunks" / vid / f"{start:06d}.json", {
                    "binding_sha256": bh, "qid": vid, "start": start,
                    "segments": stream.segments[start:start + 2], "records": records[start:start + 2], "trace": trace,
                })
            curve = modules.shared.second_scores([r["pyes"] for r in records], stream.segments, 12.0, 12)
            write(shard / "predictions" / f"{vid}.json", {"binding_sha256": bh, "second_scores": curve})
            write(shard / "query_summaries" / f"{vid}.json", {
                "binding_sha256": bh, "model_requests": 4, "output_tokens": 4, "input_tokens": 40,
                "frames": 24, "inference_seconds": 0.1, "decode_preprocess_seconds": 0.2,
            })
            aggregate.append({"vid": vid, "scores": modules.metrics.two_second_scores(curve, 6)})
        write(shard / "evaluation.json", {})
        write(shard / "complete.json", {
            "status": "complete", "binding_sha256": bh,
            "evaluation_sha256": adapter.sha256(shard / "evaluation.json"),
        })
    write(root / "plan.json", plan)
    (root / "hd_predictions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in aggregate))
    write(root / "evaluation.json", {"hd_predictions_sha256": adapter.sha256(root / "hd_predictions.jsonl")})
    write(root / "complete.json", {
        "status": "complete", "plan_sha256": adapter.sha256(root / "plan.json"),
        "evaluation_sha256": adapter.sha256(root / "evaluation.json"),
    })
    FakeStream.calls.clear()
    return root


def test_pinned_val10_and_no_generic_module_pollution(monkeypatch):
    marker = object()
    for name in ("backend", "prompts", "shared", "media", "metrics", "reze_core"):
        monkeypatch.setitem(sys.modules, name, marker)
    before = list(sys.path)
    modules = adapter._modules()
    assert modules.ranking.candidates([0.1, 0.5, 0.5]) == [1, 2, 0]
    assert len(adapter.val_ids()) == 10
    assert "Bhxk-O1Y7Ho" in adapter.val_ids()
    assert sys.path == before
    for name in ("backend", "prompts", "shared", "media", "metrics", "reze_core"):
        assert sys.modules[name] is marker


def test_prepare_never_reads_annotation_values_and_reuses_only_seen_frames(baseline, tmp_path, monkeypatch):
    original = adapter._read

    def guard(path):
        assert "evaluation_inputs" not in str(path), "Labels opened during inference preparation"
        return original(path)

    monkeypatch.setattr(adapter, "_read", guard)
    output = tmp_path / "prepared"
    rows = adapter.prepare_inputs(baseline, output)
    assert [r["vid"] for r in rows] == adapter.val_ids()
    assert all(r["query"] == "Highlight of this video." for r in rows)
    assert "SECRET TITLE" not in json.dumps(rows)
    assert all("category" not in row and "anno" not in row and "title" not in row for row in rows)
    for row in rows:
        targets, traces = adapter.load_candidates(output, row)
        assert len(targets) == 5
        assert all(len(t["video"]) == 4 for t in targets.values())
        assert row["baseline_cost"]["frames"] == 24
        assert row["preparation"]["new_model_observations"] == 0
        assert sum(t["frames_observed"] for t in traces.values()) == 20
    assert all(size <= 2 for _, _, size in FakeStream.calls)
    before = list(FakeStream.calls)
    assert adapter.prepare_inputs(baseline, output) == rows
    assert FakeStream.calls == before  # Verified resume does not decode again.


def test_reject_rgb_change_against_baseline_trace(baseline, tmp_path, monkeypatch):
    original = FakeStream.read

    def broken(self, start, size):
        clips, trace = original(self, start, size)
        trace["sampled_original_rgb_sha256"] = "corrupt"
        return clips, trace

    monkeypatch.setattr(FakeStream, "read", broken)
    with pytest.raises(ValueError, match="Old frame evidence changed"):
        adapter.prepare_inputs(baseline, tmp_path / "prepared")


def test_reject_prediction_and_raw_probability_mismatch(baseline, tmp_path):
    vid = adapter.val_ids()[0]
    path = baseline / "shard_0/predictions" / f"{vid}.json"
    old = json.loads(path.read_text())
    old["second_scores"][0] += 0.01
    write(path, old)
    with pytest.raises(ValueError, match="Saved B0 curve"):
        adapter.prepare_inputs(baseline, tmp_path / "prepared")


def test_load_rejects_npz_corruption_and_path_escape(baseline, tmp_path):
    output = tmp_path / "prepared"
    row = adapter.prepare_inputs(baseline, output)[0]
    with pytest.raises(ValueError, match="escapes"):
        adapter.load_candidates(output, {**row, "candidate_payload": "../bad.npz"})
    with (output / row["candidate_payload"]).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="Parent artifact changed"):
        adapter.load_candidates(output, row)


def test_post_prediction_evaluation_matches_native_umt_and_rejects_bad_scope(baseline):
    predictions = {r["vid"]: r["scores"] for r in adapter._jsonl(baseline / "hd_predictions.jsonl")}
    result = adapter.evaluate_predictions(predictions, baseline)
    assert result["evaluated_videos"] == 10
    assert result["author_metric_parity_passed"]
    assert result["author_evaluator_rounded_percent"] == pytest.approx(result["metrics_percent"]["Top5_AP"]["mean"], abs=0.000501)
    assert len(result["per_video"]) == 10
    with pytest.raises(ValueError, match="exactly"):
        adapter.evaluate_predictions(dict(list(predictions.items())[:-1]), baseline)
    bad = {**predictions, adapter.val_ids()[0]: [0.1]}
    with pytest.raises(ValueError, match="length or score range"):
        adapter.evaluate_predictions(bad, baseline)


def test_evaluation_rejects_changed_annotation_file(baseline):
    predictions = {r["vid"]: r["scores"] for r in adapter._jsonl(baseline / "hd_predictions.jsonl")}
    with (baseline / "shard_0/evaluation_inputs/tvsum_anno.json").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="Parent artifact changed"):
        adapter.evaluate_predictions(predictions, baseline)
