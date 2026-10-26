"""R0 real engineering gate and R1 four-arm fixed-twenty experiment."""

import argparse
import fcntl
import itertools
import os
from pathlib import Path

from backend import MODEL, Backend, SemanticFailure
from budget import BudgetStop, Ledger
from comparison import allowed_references, rerank, validate_comparison
from exporters import benchmark_prediction
from io_utils import compact, legacy, load_run, parents, read_json, sha256, utc_now, write_json
from memory import FactStore, build_memory, judgment, retrieve
from prompts import CAPS, COMPARE, FACTS, MEMORY, OBSERVE, TASK
from schema import Comparison, Facts, MemoryDraft, ObservationFacts, TaskSpec, validate_task

ARMS = ["C_direct4", "M_plain", "M_decision"]


def frame_map(trace):
    return [{"id": f"s{x['frame_index']}", "pts": x["frame_timestamps"]} for x in trace["frames"]]


def fallback(row, binding, binding_hash, reason):
    baseline, pair = parents(binding, row)
    original = baseline["prediction"]["pred_saliency_scores"]
    # Each new arm's valid parent is the unchanged, audited E0 (P_pair).
    return {
        "qid": row["qid"],
        "vid": row["vid"],
        "query": row["query"],
        "binding_sha256": binding_hash,
        "candidate_ids": pair["candidate_ids"],
        "original": original,
        "E0": pair["arms"]["P_pair"]["scores"],
        "error": reason,
        "arms": {a: {"status": "fallback", "error": reason, "scores": pair["arms"]["P_pair"]["scores"]} for a in ARMS},
    }


def one_query(row, binding, binding_hash, backend, frozen, stage):
    from candidate_frames import prepare_candidates
    from observer import read_observation

    qid = str(row["qid"])
    baseline, pair = parents(binding, row)
    selected, original = pair["candidate_ids"], baseline["prediction"]["pred_saliency_scores"]
    targets, traces, decode_seconds = prepare_candidates(row, baseline, selected)
    if traces != frozen["candidate_frames"]:
        raise ValueError("Candidate pixels/PTS changed after preflight")
    maps = {c: frame_map(traces[str(c)]) for c in selected}
    frame_ids = {c: [f["id"] for f in maps[c]] for c in selected}
    result = fallback(row, binding, binding_hash, "not_executed")
    result.update(candidate_frames=traces, decode_seconds=decode_seconds, errors=[])
    task, _ = backend.call(
        qid + "/task",
        TASK.format(query=row["query"]),
        [],
        [],
        TaskSpec,
        CAPS["task"],
        ARMS,
        lambda x: validate_task(x, row["query"]),
    )
    result["task"] = task
    store = FactStore()
    memories = {}
    try:
        for c in selected:

            def validate_facts(value, candidate=c):
                store.extract(candidate, value, frame_ids[candidate])
                return value

            backend.call(
                f"{qid}/facts/{c}",
                FACTS.format(frames=compact(maps[c])),
                [targets[c]],
                [traces[str(c)]],
                Facts,
                CAPS["facts"],
                ["M_plain", "M_decision"],
                validate_facts,
                references={"FactItem.frames": frame_ids[c]},
            )
        result["facts"] = store.dump()
        for arm in ("M_plain", "M_decision"):
            try:
                memory, _ = backend.call(
                    f"{qid}/memory/{arm}",
                    MEMORY[arm].format(facts=compact(store.dump())),
                    [],
                    {"fact_store_sha256": store.version},
                    MemoryDraft,
                    CAPS["memory"],
                    [arm],
                    lambda x: build_memory(x, store.dump(), backend.tokenizer),
                    references={"MemoryItem.fact_ids": list(store.dump())},
                )
                memories[arm] = memory
            except SemanticFailure as error:
                result["errors"].append({"component": arm, "error": str(error)})
    except SemanticFailure as error:
        result["errors"].append({"component": "facts", "error": str(error)})
    result["memories"] = memories
    pairs = list(itertools.permutations(selected if stage == "R1" else selected[:2], 2))
    for arm in ARMS:
        if arm != "C_direct4" and arm not in memories:
            result["arms"][arm]["error"] = "Invalid shared facts or memory; no semantic retry"
            continue
        view = {} if arm == "C_direct4" else retrieve(memories[arm], backend.tokenizer)
        comparisons = []
        try:
            for a, b in pairs:
                value, raw = backend.call(
                    f"{qid}/{arm}/{a}/{b}",
                    COMPARE.format(
                        task=compact(task), a_frames=compact(maps[a]), b_frames=compact(maps[b]), memory=compact(view)
                    ),
                    [targets[a], targets[b]],
                    [traces[str(a)], traces[str(b)]],
                    Comparison,
                    CAPS["comparison"],
                    [arm],
                    lambda x, a=a, b=b: validate_comparison(x, a, b, frame_ids, view),
                    references={
                        "a_evidence": allowed_references(a, frame_ids, view),
                        "b_evidence": allowed_references(b, frame_ids, view),
                    },
                )
                comparisons.append(
                    {
                        "a": a,
                        "b": b,
                        "result": value,
                        "attempt": raw["attempt"],
                        "response_path": raw.get("response_path"),
                        "judgment": judgment(task, view, value),
                    }
                )
            scores, ranking = rerank(original, selected, comparisons) if stage == "R1" else (None, None)
            result["arms"][arm] = {
                "status": "complete",
                "comparisons": comparisons,
                "scores": scores,
                "ranking": ranking,
            }
            if stage == "R1":
                benchmark_prediction(row, scores, original, selected)
        except SemanticFailure as error:
            result["arms"][arm].update(error=str(error), comparisons=comparisons)
            result["errors"].append({"component": arm + "/comparison", "error": str(error)})
    if stage == "R0":
        seen = {i for group in baseline["media"]["frame_indices"] for i in group}
        target, trace = read_observation(
            row, frozen["observation"]["action"], seen, baseline["media"]["resized_height_width"]
        )
        if trace != frozen["observation"] or len(trace["frames"]) != 24:
            raise ValueError("24-frame real observation failed exact preflight replay")
        observed_ids = [f["id"] for f in trace["frames"]]

        def validate_observed(value):
            isolated = FactStore()
            isolated.extract("observation", value, observed_ids)
            return value

        try:
            observed, raw = backend.call(
                f"{qid}/observation",
                OBSERVE.format(
                    question=trace["action"]["question"],
                    frames=compact([{"id": f["id"], "pts": f["pts"]} for f in trace["frames"]]),
                ),
                [target],
                [trace],
                ObservationFacts,
                CAPS["observation"],
                ["engineering"],
                validate_observed,
                references={"ObservationFact.frames": observed_ids},
            )
            result["observation"] = {
                "trace": trace,
                "result": observed,
                "attempt": raw["attempt"],
                "response_path": raw.get("response_path"),
            }
        except SemanticFailure as error:
            result["errors"].append({"component": "observation", "error": str(error)})
    result.pop("error", None)
    result["completed_at"] = utc_now()
    return result


def finish(run, stage, binding, rows, binding_hash, ledger, stopped=None):
    directory = run / stage
    receipt = {}
    for row in rows:
        path = directory / "records" / f"{row['qid']}.json"
        if not path.exists():
            write_json(path, fallback(row, binding, binding_hash, stopped or "missing_record"))
        receipt[str(row["qid"])] = sha256(path)
    write_json(directory / "record_receipt.json", receipt)
    summary = {
        "stage": stage,
        "queries": len(rows),
        "binding_sha256": binding_hash,
        "finished_at": utc_now(),
        "record_receipt_sha256": sha256(directory / "record_receipt.json"),
        "actual_attempts": len(ledger.state["attempts"]),
        "actual_charged_tokens": ledger.cost(),
        "verified_reused_requests": len(ledger.state.get("reused", {})),
        "cumulative_attempts": (
            binding.get("historical_usage", {}).get(stage, {}).get("attempts", 0) + len(ledger.state["attempts"])
        ),
        "cumulative_charged_tokens": (
            binding.get("historical_usage", {}).get(stage, {}).get("charged_tokens", 0) + ledger.cost()
        ),
        "stopped_reason": stopped,
        "formal_metrics_evaluated": False,
    }
    if stage == "R0":
        records = [read_json(directory / "records" / f"{r['qid']}.json") for r in rows]
        summary["checks"] = {
            "all_twenty_retained": len(records) == 20,
            "all_shared_and_comparison_contracts_valid": all(
                not r.get("errors") and not r.get("error") and all(r["arms"][a]["status"] == "complete" for a in ARMS)
                for r in records
            ),
            "twenty_real_24_frame_observations": all(
                len(r.get("observation", {}).get("trace", {}).get("frames", [])) == 24 for r in records
            ),
            "nonempty_source_cited_observations": all(
                r.get("observation", {}).get("result", {}).get("facts")
                and all(
                    fact["frames"]
                    and set(fact["frames"]) <= {frame["id"] for frame in r["observation"]["trace"]["frames"]}
                    for fact in r["observation"]["result"]["facts"]
                )
                for r in records
            ),
            "cpu_behavior_tests_passed": read_json(run / "cpu_tests.json")["returncode"] == 0,
            "budget_not_stopped": stopped is None,
        }
        summary["status"] = "passed" if all(summary["checks"].values()) else "failed"
        write_json(directory / "gate.json", summary)
        if summary["status"] == "passed":
            write_json(
                run / "formal_protocol.json",
                {
                    "status": "frozen_after_engineering",
                    "frozen_at": utc_now(),
                    "binding_sha256": binding_hash,
                    "R0_gate_sha256": sha256(directory / "gate.json"),
                    "source_sha256": binding["source_sha256"],
                    "prompt_version": binding["protocol"],
                    "engineering_accuracy_used_for_prompt_selection": False,
                },
            )
    else:
        summary["status"] = "stopped_at_budget" if stopped else "inference_complete"
        write_json(directory / "inference_complete.json", summary)
    write_json(directory / "status.json", summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--stage", choices=["R0", "R1"], required=True)
    args = parser.parse_args()
    run, stage = args.run.resolve(), args.stage
    directory = run / stage
    directory.mkdir(exist_ok=True)
    with (directory / "worker.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        binding, rows, binding_hash = load_run(run)
        if stage == "R1":
            formal = read_json(run / "formal_protocol.json")
            if (
                read_json(run / "R0/gate.json")["status"] != "passed"
                or formal["binding_sha256"] != binding_hash
                or (formal["R0_gate_sha256"] != sha256(run / "R0/gate.json"))
            ):
                raise ValueError("R1 requires the passed frozen R0 gate")
        import torch

        torch.set_num_threads(2)
        actual = "GPU-" + str(torch.cuda.get_device_properties(0).uuid).removeprefix("GPU-")
        if torch.cuda.device_count() != 1 or actual != os.environ["REZE_EXPECTED_GPU_UUID"]:
            raise ValueError("Worker resolved the wrong GPU UUID")
        model = read_json(run / "model_verification.json")
        if model["audit"]["model_path"] != MODEL:
            raise ValueError("Model location drift")
        for name, expected in model["stats"].items():
            stat = (Path(MODEL) / name).stat()
            if stat.st_size != expected["size"] or stat.st_mtime_ns != expected["mtime_ns"]:
                raise ValueError("Verified model files changed")
        ledger = Ledger(directory / "ledger.json", binding["budgets"][stage])
        reuse_index = (
            read_json(run / "reuse_index.json") if stage == "R0" and (run / "reuse_index.json").exists() else {}
        )
        backend = Backend(directory, ledger, reuse_index)
        frozen = {r["qid"]: r for r in legacy.read_jsonl(run / "candidate_inputs.jsonl")}
        stopped = None
        for index, row in enumerate(rows):
            path = directory / "records" / f"{row['qid']}.json"
            if path.exists():
                record = read_json(path)
                if record["binding_sha256"] != binding_hash or record["qid"] != row["qid"]:
                    raise ValueError("Resume record drift")
                continue
            try:
                record = one_query(row, binding, binding_hash, backend, frozen[row["qid"]], stage)
            except SemanticFailure as error:
                record = fallback(row, binding, binding_hash, str(error))
            except BudgetStop as error:
                stopped = str(error)
                break
            write_json(path, record)
            write_json(
                directory / "status.json",
                {
                    "status": "running",
                    "stage": stage,
                    "completed_queries": index + 1,
                    "expected_queries": len(rows),
                    "updated_at": utc_now(),
                    "gpu_uuid": actual,
                },
            )
            print(f"{stage} {index + 1}/{len(rows)} qid={row['qid']}", flush=True)
        finish(run, stage, binding, rows, binding_hash, ledger, stopped)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError:
        import traceback

        traceback.print_exc()
        raise SystemExit(75)  # Only runtime failures are eligible for bounded recovery.
