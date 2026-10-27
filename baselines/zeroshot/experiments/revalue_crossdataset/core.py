"""Frozen-comparator transfer, with bounded calls and no additional observations."""

import importlib.util
import itertools
import sys
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/revalue"))
from backend import Backend, SemanticFailure  # noqa: E402
from budget import BudgetStop, Ledger  # noqa: E402
from comparison import allowed_references, rerank, validate_comparison  # noqa: E402
from io_utils import compact, digest, read_json, utc_now, write_json  # noqa: E402
from memory import FactStore, build_memory, retrieve  # noqa: E402
from prompts import CAPS, COMPARE, FACTS, MEMORY, TASK  # noqa: E402
from ranking import candidates, make_scores, ordered_pairs  # noqa: E402
from schema import Comparison, Facts, MemoryDraft, TaskSpec, validate_task  # noqa: E402

spec = importlib.util.spec_from_file_location("crossdataset_pair_backend", ROOT / "experiments/reze_evidence/backend.py")
PAIR = importlib.util.module_from_spec(spec)
spec.loader.exec_module(PAIR)

ARMS = ["P_pair", "C_direct4", "E_memory"]
MAX_CONTEXT = 6144
CALLS_PER_VIDEO = 67
MAX_SHARD_CALLS = 5 * CALLS_PER_VIDEO
MAX_SHARD_TOKENS = MAX_SHARD_CALLS * MAX_CONTEXT
LIMITS = {"requests": MAX_SHARD_CALLS, "extra_attempts": 0,
          "arm_incremental_tokens": {arm: MAX_SHARD_TOKENS for arm in ARMS}}


class TransferLedger(Ledger):
    def reserve(self, key, prompt_tokens, output_cap, arms, request_hash, frames):
        if set(arms) - set(ARMS):
            raise ValueError("Unknown cost owner")
        if prompt_tokens + output_cap > MAX_CONTEXT:
            raise BudgetStop("Exact request exceeds fixed context budget")
        if self.cost() + prompt_tokens + output_cap > MAX_SHARD_TOKENS:
            raise BudgetStop("Absolute shard token budget")
        return super().reserve(key, prompt_tokens, output_cap, arms, request_hash, frames)


class TransferBackend(Backend):
    def pair(self, key, query, a, b, traces):
        """Reuse the original Yes/No prompt, top20 parser and score rule verbatim."""
        from vllm import SamplingParams

        ids = {word: self.tokenizer.encode(word, add_special_tokens=False) for word in ("Yes", "No")}
        if ids != {"Yes": [9454], "No": [2753]}:
            raise ValueError("Answer tokenizer drift")
        request = PAIR.make_request(self.tokenizer, query, "P_pair", a, b)
        specification = {"rendered_prompt": request["prompt"], "evidence": traces,
                         "sampling": {"temperature": 0, "max_tokens": 1, "logprobs": 20}, "arm": "P_pair"}
        request_hash = digest(specification)
        prior = self.ledger.attempts_for(key)
        if prior:
            old = prior[-1]
            if old["request_sha256"] != request_hash:
                raise ValueError("Pair request changed")
            if old["status"] in {"complete", "invalid"}:
                record = read_json(self.directory / "attempts" / f"{old['number']:05d}.json")
                if digest(record) != old["record_sha256"]:
                    raise ValueError("Pair cache changed")
                if old["status"] == "invalid":
                    raise SemanticFailure(record["error"])
                return record
        processed = self.engine.llm_engine.input_processor.input_preprocessor.preprocess(deepcopy(request))
        number = self.ledger.reserve(key, len(processed["prompt_token_ids"]), 1, ["P_pair"], request_hash,
                                     len(a["video"]) + len(b["video"]))
        del processed
        record = {"key": key, "attempt": number, "request_sha256": request_hash,
                  "specification": specification, "started_at": utc_now()}
        write_json(self.status_path, {"status": "generating", "key": key, "at": utc_now()})
        before = time.monotonic()
        try:
            values, elapsed = PAIR.score_requests(
                self.engine, SamplingParams(temperature=0, max_tokens=1, logprobs=20, top_p=1, top_k=-1),
                [request], {k: v[0] for k, v in ids.items()})
            record.update(values[0], inference_seconds=elapsed)
            record["status"] = "invalid" if record.get("error") else "complete"
        except Exception as error:
            record.update(status="runtime_error", error=f"{type(error).__name__}: {error}",
                          inference_seconds=time.monotonic() - before)
            write_json(self.directory / "attempts" / f"{number:05d}.json", record)
            self.ledger.settle(number, record)
            raise
        write_json(self.directory / "attempts" / f"{number:05d}.json", record)
        self.ledger.settle(number, record)
        if record["status"] == "invalid":
            raise SemanticFailure(record["error"])
        return record


def verify_scores(original, selected, scores):
    if selected != candidates(original) or len(scores) != len(original):
        raise ValueError("Candidate selection or score length changed")
    if sorted(original) != sorted(scores):
        raise ValueError("Reranking must preserve the original scores")
    if any(scores[i] != original[i] for i in range(len(original)) if i not in selected):
        raise ValueError("Changed a noncandidate score")


def one_video(row, targets, traces, backend):
    qid, selected, original = row["qid"], row["candidate_ids"], row["scores"]
    result = {"qid": qid, "vid": row["vid"], "query": row["query"], "candidate_ids": selected,
              "candidate_frames": traces, "B0": original, "arms": {}, "errors": []}
    pair_records = []
    try:
        for a, b in ordered_pairs(selected):
            record = backend.pair(f"{qid}/P_pair/{a}/{b}", row["query"], targets[a], targets[b],
                                  [traces[str(a)], traces[str(b)]])
            pair_records.append({**record, "a": a, "b": b})
        scores, ranking = make_scores(original, selected, pair_records, "P_pair")
        result["arms"]["P_pair"] = {"status": "complete", "scores": scores, "ranking": ranking,
                                        "comparisons": pair_records}
    except SemanticFailure as error:
        result["arms"]["P_pair"] = {"status": "fallback", "scores": original, "error": str(error),
                                        "comparisons": pair_records, "fallback_parent": "B0"}
    parent = result["arms"]["P_pair"]["scores"]
    for arm in ("C_direct4", "E_memory"):
        result["arms"][arm] = {"status": "fallback", "scores": parent, "error": "not_executed",
                                "fallback_parent": "P_pair"}
    try:
        task, _ = backend.call(f"{qid}/task", TASK.format(query=row["query"]), [], [], TaskSpec, CAPS["task"],
                               ["C_direct4", "E_memory"], lambda value: validate_task(value, row["query"]))
    except SemanticFailure as error:
        for arm in ("C_direct4", "E_memory"):
            result["arms"][arm]["error"] = str(error)
        return result
    result["task"] = task
    maps = {c: [{"id": f"s{f['frame_index']}", "pts": f["frame_timestamps"]}
                for f in traces[str(c)]["frames"]] for c in selected}
    frame_ids = {c: [f["id"] for f in maps[c]] for c in selected}
    store, memory = FactStore(), None
    try:
        for c in selected:
            def validate_facts(value, candidate=c):
                store.extract(candidate, value, frame_ids[candidate])
                return value
            backend.call(f"{qid}/facts/{c}", FACTS.format(frames=compact(maps[c])), [targets[c]],
                         [traces[str(c)]], Facts, CAPS["facts"], ["E_memory"], validate_facts,
                         references={"FactItem.frames": frame_ids[c]})
        result["facts"] = store.dump()
        memory, _ = backend.call(
            f"{qid}/memory/M_decision", MEMORY["M_decision"].format(facts=compact(store.dump())), [],
            {"fact_store_sha256": store.version}, MemoryDraft, CAPS["memory"], ["E_memory"],
            lambda value: build_memory(value, store.dump(), backend.tokenizer),
            references={"MemoryItem.fact_ids": list(store.dump())})
        result["memory"] = memory
    except SemanticFailure as error:
        result["errors"].append({"component": "memory", "error": str(error)})
        result["arms"]["E_memory"]["error"] = str(error)
    for arm in ("C_direct4", "E_memory"):
        if arm == "E_memory" and memory is None:
            continue
        view = {} if arm == "C_direct4" else retrieve(memory, backend.tokenizer)
        comparisons = []
        try:
            for a, b in itertools.permutations(selected, 2):
                value, raw = backend.call(
                    f"{qid}/{arm}/{a}/{b}",
                    COMPARE.format(task=compact(task), a_frames=compact(maps[a]), b_frames=compact(maps[b]),
                                   memory=compact(view)), [targets[a], targets[b]], [traces[str(a)], traces[str(b)]],
                    Comparison, CAPS["comparison"], [arm],
                    lambda value, a=a, b=b: validate_comparison(value, a, b, frame_ids, view),
                    references={"a_evidence": allowed_references(a, frame_ids, view),
                                "b_evidence": allowed_references(b, frame_ids, view)})
                comparisons.append({"a": a, "b": b, "result": value, "attempt": raw["attempt"]})
            scores, ranking = rerank(original, selected, comparisons)
            result["arms"][arm] = {"status": "complete", "scores": scores, "ranking": ranking,
                                    "comparisons": comparisons,
                                    "preferences": dict(Counter(r["result"]["preference"] for r in comparisons))}
        except SemanticFailure as error:
            result["arms"][arm].update(error=str(error), comparisons=comparisons)
    for arm in ARMS:
        verify_scores(original, selected, result["arms"][arm]["scores"])
    result["finished_at"] = utc_now()
    return result
