"""Reader controls, provenance, bounded retries and full-query failure behavior."""

import copy
import json
import time

import pytest
from highlight_evidence import memory_reader as module
from highlight_evidence.memory_only import score_request as original_request
from highlight_evidence.robust_client import MemoryClient

from merit_qvh.core import write_json


def memory_for(count):
    return [
        {
            "id": f"g{i // 5}:e{i % 5}",
            "start": 2 * i,
            "end": 2 * (i + 1),
            "description": "Person near a door. Level: 2. Evidence: f0.",
            "evidence_ids": [f"f{i}"],
            "clip_id": i,
        }
        for i in range(count)
    ]


def output_for(targets):
    return {"clips": [{"clip_id": i, "evidence_ids": [f"f{i}"], "score": 2.5} for i in targets]}


def response(payload, finish="stop"):
    return {"choices": [{"finish_reason": finish, "message": {"content": json.dumps(payload)}}]}


def test_both_readers_keep_all_original_text_and_query_without_scores_or_tools():
    memory = memory_for(18)
    original = copy.deepcopy(memory)
    row = {
        "query": "Person opens a door",
        "duration": 36,
        "D0": "DO_NOT_READ",
        "saliency_scores": "NO_LABELS",
    }
    content0, _ = original_request(row, memory, "Memory_text")
    body0 = json.loads(content0[0]["text"].split("\n", 1)[1])
    views = []
    for targets in (list(range(18)), list(range(15)), list(range(15, 18))):
        content, _ = module.score_request(row, memory, targets)
        assert len(content) == 1 and content[0]["type"] == "text"
        instruction, raw = content[0]["text"].split("\n", 1)
        assert "DO_NOT_READ" not in raw and "NO_LABELS" not in raw
        body = json.loads(raw)
        assert body.pop("target_clip_ids") == targets
        assert body == body0
        views.append((instruction, body))
    assert views[0] == views[1] == views[2] and memory == original


@pytest.mark.parametrize(
    "count,arm,sizes", [(75, "R1", [75]), (75, "R2", [15] * 5), (74, "R2", [15] * 4 + [14])]
)
def test_target_partition_has_exact_full_coverage(count, arm, sizes):
    groups = module.targets_for(count, arm)
    assert list(map(len, groups)) == sizes
    assert [i for group in groups for i in group] == list(range(count))


@pytest.mark.parametrize("count", [15, 75])
def test_xgrammar_enforces_metadata_and_accepts_entire_output(count):
    import xgrammar
    from xgrammar.testing import _is_grammar_accept_string

    targets = list(range(count))
    contract = module.ClipContract(memory_for(count), targets)
    grammar = xgrammar.Grammar.from_json_schema(json.dumps(contract.schema), any_whitespace=False)
    valid = output_for(targets)
    assert _is_grammar_accept_string(grammar, json.dumps(valid))
    invalid = copy.deepcopy(valid)
    invalid["clips"][-1]["clip_id"] = 0
    assert not _is_grammar_accept_string(grammar, json.dumps(invalid))


@pytest.mark.parametrize(
    "violation",
    [
        "missing",
        "reordered",
        "duplicate",
        "neighbor_ref",
        "empty_ref",
        "duplicate_ref",
        "bool_id",
        "bool_score",
        "nan",
        "high",
        "extra_field",
    ],
)
def test_invalid_provenance_or_scores_cannot_be_degraded_to_success(violation):
    contract = module.ClipContract(memory_for(3), [0, 1, 2])
    candidate = output_for([0, 1, 2])
    if violation == "missing":
        candidate["clips"].pop()
    elif violation == "reordered":
        candidate["clips"].reverse()
    elif violation == "duplicate":
        candidate["clips"][1] = copy.deepcopy(candidate["clips"][0])
    else:
        item = candidate["clips"][0]
        key, value = {
            "neighbor_ref": ("evidence_ids", ["f1"]),
            "empty_ref": ("evidence_ids", []),
            "duplicate_ref": ("evidence_ids", ["f0", "f0"]),
            "bool_id": ("clip_id", False),
            "bool_score": ("score", True),
            "nan": ("score", float("nan")),
            "high": ("score", 5),
            "extra_field": ("extra", 1),
        }[violation]
        item[key] = value
    assert contract.problems(candidate)
    with pytest.raises(ValueError, match="Unusable"):
        contract.finalize(candidate, degrade=True)


def test_merging_keeps_scores_and_rejects_incomplete_or_duplicate_batches():
    memory = memory_for(18)
    batches = [output_for(group) for group in module.targets_for(18, "R2")]
    assert module.merge_batches(memory, batches)["scores"] == [2.5] * 18
    for invalid in (batches[:1], [batches[0], batches[0]], list(reversed(batches))):
        with pytest.raises(ValueError, match="Unusable"):
            module.merge_batches(memory, invalid)


def test_context_overflow_is_not_silently_truncated():
    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return [0] * 30000

    class Client:
        def complete_memory(self, *args, **kwargs):
            pytest.fail("No request on overflow")

    with pytest.raises(ValueError, match="no truncation"):
        module.score_batch(
            Client(), {"query": "door", "duration": 6}, memory_for(3), [0, 1, 2], "R1", Tokenizer(), 32768
        )


def test_one_failed_r2_batch_fails_whole_query_and_remains_cached(tmp_path, monkeypatch):
    row = {"qid": 1, "vid": "test", "query": "door", "duration": 36}
    write_json(
        tmp_path / "inputs/1.json",
        {"row": row, "memory": memory_for(18), "source_memory_status": {"partial_windows": [4]}},
    )
    write_json(tmp_path / "references/1.json", {"D0": {"pred_saliency_scores": [4] * 18}})
    read = module.read

    def guarded_read(path):
        if "references" in path.parts:
            pytest.fail("Inference must not read baseline vectors")
        return read(path)

    class FakeClient:
        def __init__(self, *args):
            self.events = []

    called = []

    def score(client, row, memory, targets, *args):
        called.append(targets)
        if targets[0] == 15:
            raise RuntimeError("terminal failure")
        return output_for(targets)

    monkeypatch.setattr(module, "read", guarded_read)
    monkeypatch.setattr(module, "MemoryClient", FakeClient)
    monkeypatch.setattr(module, "score_batch", score)
    c = {"endpoint": "unused", "vlm": {}, "server": {"max_model_len": 32768}}
    result = module.run_query(tmp_path, c, "protocol", row, None, "R2")
    assert result["status"] == "failed" and result["pred_saliency_scores"] == [2.0] * 18
    assert result["batches"][0]["status"] == "ok" and result["batches"][1]["status"] == "failed"
    assert result["source_memory_status"]["partial_windows"] == [4]
    assert module.run_query(tmp_path, c, "protocol", row, None, "R2") == result
    assert len(called) == 2


def test_real_client_patch_and_raw_replay_detect_saved_score_tampering(tmp_path, monkeypatch):
    config = {"model": "test", "temperature": 0, "format_retries": 1, "transport_retries": 1}
    client = MemoryClient("http://unused/v1", config, tmp_path / "requests", "protocol")
    content, contract = module.score_request({"query": "door", "duration": 6}, memory_for(3), [0, 1, 2])
    invalid = output_for([0, 1, 2])
    invalid["clips"][1]["evidence_ids"] = ["f0"]
    valid = output_for([0, 1, 2])
    responses = iter([response(invalid), response({"/": valid})])
    monkeypatch.setattr(client, "_post", lambda payload: next(responses))
    result = client.complete_memory(
        "R1_score", content, module.MAX_TOKENS, contract, system=module.SCORE_SYSTEM
    )
    request = module.read(next((tmp_path / "requests").glob("*/*.json")))
    expected = module.request_payload(config, content, module.MAX_TOKENS, contract, module.SCORE_SYSTEM)
    assert result["repair_attempted"] and len(request["attempts"]) == 2
    assert module.replay_request(request, expected, contract)["clips"] == valid["clips"]
    request["result"]["scores"][0] = 4
    with pytest.raises(ValueError, match="raw model response"):
        module.replay_request(request, expected, contract)


def test_development_gate_requires_quality_not_just_nonconstant_scores():
    records = {i: {"R1": {"status": "ok", "pred_saliency_scores": [0, 1, 2] * 25}} for i in range(20)}
    metrics = {i: {"R0": {"mAP": 20, "Hit1": 30}, "R1": {"mAP": 21, "Hit1": 30}} for i in records}
    assert module.development_gate(metrics, records, "R1")["passed"]
    metrics[0]["R1"]["mAP"] = 20.9
    assert not module.development_gate(metrics, records, "R1")["passed"]
    metrics[0]["R1"]["mAP"] = 21
    records[0]["R1"]["status"] = "failed"
    assert not module.development_gate(metrics, records, "R1")["passed"]


def test_first4_gate_stops_on_any_terminal_failure(tmp_path, monkeypatch):
    called = []

    def query(root, c, protocol, row, *args):
        called.append(row["qid"])
        return {"qid": row["qid"], "status": "failed" if row["qid"] == 2 else "ok"}

    monkeypatch.setattr(module, "run_query", query)
    rows = [{"qid": i} for i in range(20)]
    with pytest.raises(RuntimeError, match="First4"):
        module.run_stage(
            tmp_path, {"workers": 1, "max_wall_seconds": 60}, "test", rows, None, "R1", time.time()
        )
    assert called == [0, 1, 2, 3]
    assert module.read(tmp_path / "circuit_breaker.json")["stage"] == "R1"


@pytest.mark.parametrize(
    "r1_pass,r2_pass,expected",
    [(True, False, ["R1"]), (False, True, ["R1", "R2"]), (False, False, ["R1", "R2"])],
)
def test_preregistered_stage_decision_never_runs_r2_after_r1_pass(
    tmp_path, monkeypatch, r1_pass, r2_pass, expected
):
    from transformers import AutoTokenizer

    plan = tmp_path / "plan.md"
    plan.write_text("frozen plan")
    c = {
        "plan_sha256": module.file_sha256(plan),
        "endpoint": "unused",
        "vlm": {"weight_path": "unused"},
        "workers": 1,
    }
    monkeypatch.setattr(module, "verify", lambda *args, **kwargs: (c, "protocol"))
    monkeypatch.setattr(module, "read_jsonl", lambda path: [{"qid": i} for i in range(20)])
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: None)

    class FakeClient:
        def __init__(self, *args):
            pass

        def check_model(self):
            return "test"

    monkeypatch.setattr(module, "MemoryClient", FakeClient)
    called = []
    monkeypatch.setattr(module, "run_stage", lambda *args: called.append(args[-2]))
    monkeypatch.setattr(module, "audit", lambda *args: {"status": "passed"})

    def evaluate(root, arms):
        return {
            "development_gate": {"passed": r1_pass if arms[-1] == "R1" else r2_pass},
            "groups": {arm: {"metrics": {"HL-min-VeryGood": {"HL-mAP": 25}}} for arm in arms},
        }

    monkeypatch.setattr(module, "evaluate", evaluate)
    module.run(tmp_path)
    assert called == expected
    progress = module.read(tmp_path / "progress.json")
    assert progress["completed_arms"] == expected
    assert progress["selected_reader"] == (expected[-1] if r1_pass or r2_pass else None)
