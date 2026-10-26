"""CPU wiring tests: exact protocol call counts and dependency fallback behavior."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import candidate_frames
import io_utils  # noqa: F401
import observer
import worker
from backend import SemanticFailure
from schema import Comparison, Facts, MemoryDraft, ObservationFacts, TaskSpec


class Tokenizer:
    def encode(self, text, **kwargs):
        return list(text.split())


class FakeBackend:
    tokenizer = Tokenizer()

    def __init__(self, fail=None):
        self.calls = []
        self.fail = fail

    def call(self, key, text, videos, evidence, schema, cap, arms, validate, references=None, output_schema=None):
        self.calls.append(key)
        if self.fail and self.fail in key:
            raise SemanticFailure("Prespecified fixture invalid response")
        if schema is TaskSpec:
            value = {"request_type": "specific", "purpose": "show action", "constraints": []}
        elif schema in (Facts, ObservationFacts):
            trace = evidence[0]
            frame_id = trace["frames"][0].get("id", "s" + str(trace["frames"][0].get("frame_index")))
            value = {"facts": [{"kind": "event", "text": "Visible action", "frames": [frame_id]}], "unknown": []}
        elif schema is MemoryDraft:
            value = {"items": [{"text": "Actions", "fact_ids": ["c0_f0", "c1_f0", "c2_f0", "c3_f0", "c4_f0"]}]}
        elif schema is Comparison:
            a, b = [trace["frames"][0]["frame_index"] for trace in evidence]
            value = {
                "a_eligibility": "satisfied",
                "b_eligibility": "satisfied",
                "preference": "tie",
                "values": ["action"],
                "reason": "Both depict an action",
                "a_evidence": [f"s{a}"],
                "b_evidence": [f"s{b}"],
                "missing_question": None,
            }
        else:
            raise AssertionError(schema)
        return validate(value), {"attempt": len(self.calls) - 1}


def fixture(monkeypatch):
    traces = {
        str(i): {
            "start": i * 2.0,
            "end": i * 2.0 + 2,
            "frames": [{"frame_index": i, "frame_timestamps": [i * 2.0, i * 2.0 + 0.03]}],
        }
        for i in range(5)
    }
    targets = {i: {"video": np.zeros((4, 32, 32, 3), dtype=np.uint8), "metadata": {}} for i in range(5)}
    original = [0.9, 0.8, 0.7, 0.6, 0.5, 0.1]
    baseline = {
        "prediction": {"pred_saliency_scores": original},
        "media": {"frame_indices": [[0, 1, 2, 3, 4]], "resized_height_width": [32, 32]},
    }
    e0 = [0.8, 0.9, 0.7, 0.6, 0.5, 0.1]
    pair = {"candidate_ids": list(range(5)), "arms": {"P_pair": {"scores": e0}}}
    monkeypatch.setattr(worker, "parents", lambda *args: (baseline, pair))
    monkeypatch.setattr(candidate_frames, "prepare_candidates", lambda *args: (targets, traces, 0.0))
    observation = {
        "action": {"question": "Action?"},
        "frames": [{"id": f"s{i}", "pts": [float(i), float(i + 1)]} for i in range(24)],
    }
    monkeypatch.setattr(observer, "read_observation", lambda *args: (targets[0], observation))
    row = {"qid": 1, "vid": "video", "query": "show action", "duration": 20.0, "video_path": "unused"}
    return row, {"candidate_frames": traces, "observation": observation}, e0


def test_r0_wiring_uses_15_engineering_calls_no_hd_ranking(monkeypatch):
    row, frozen, _ = fixture(monkeypatch)
    backend = FakeBackend()
    output = worker.one_query(row, {}, "binding", backend, frozen, "R0")
    assert len(backend.calls) == 15
    assert all(v["status"] == "complete" and v["scores"] is None for v in output["arms"].values())
    assert "observation" in output and not output["errors"]


def test_r1_wiring_uses_68_calls_and_no_new_observation(monkeypatch):
    row, frozen, _ = fixture(monkeypatch)
    backend = FakeBackend()
    output = worker.one_query(row, {}, "binding", backend, frozen, "R1")
    assert len(backend.calls) == 68
    assert not any("observation" in key for key in backend.calls)
    assert all(len(v["comparisons"]) == 20 for v in output["arms"].values())
    assert output["memories"]["M_plain"]["fact_store_version"] == output["memories"]["M_decision"]["fact_store_version"]


def test_memory_failure_preserves_direct_and_valid_parent_full_scores(monkeypatch):
    row, frozen, e0 = fixture(monkeypatch)
    backend = FakeBackend(fail="/memory/M_plain")
    output = worker.one_query(row, {}, "binding", backend, frozen, "R1")
    assert output["arms"]["M_plain"]["status"] == "fallback"
    assert output["arms"]["M_plain"]["scores"] == e0
    assert output["arms"]["C_direct4"]["status"] == output["arms"]["M_decision"]["status"] == "complete"
    assert backend.calls.count("1/memory/M_plain") == 1
