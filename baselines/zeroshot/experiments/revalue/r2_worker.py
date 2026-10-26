"""R2: identical problems and frame budgets, uniform versus targeted source observation."""

import argparse
import itertools
import os
from copy import deepcopy
from pathlib import Path

from backend import MODEL, Backend, SemanticFailure
from budget import BudgetStop
from comparison import allowed_references, rerank, validate_comparison
from io_utils import compact, digest, legacy, parents, read_json, sha256, utc_now, write_json
from memory import FactStore, build_memory, retrieve
from prompts import CAPS, COMPARE, FACTS, MEMORY, OBSERVE
from r2_observer import pair_actions, read_frames, source_timeline, timeline_hash
from r2_planner import PLAN, LocalPlan, actionable_gaps, candidate_ranges, resolve_plan, update_memory, validate_plan
from r2_protocol import ARMS, OBSERVERS, R1_ARMS, SharedLedger, claim_query, load_run
from schema import Comparison, Facts, MemoryDraft, ObservationFacts
from task_grounding import extract_task
from worker import frame_map


def fallback(row, binding, binding_hash, reason):
    baseline, pair = parents(binding, row)
    return {
        **{k: row[k] for k in ("qid", "vid", "query")},
        "binding_sha256": binding_hash,
        "candidate_ids": pair["candidate_ids"],
        "original": baseline["prediction"]["pred_saliency_scores"],
        "E0": pair["arms"]["P_pair"]["scores"],
        "errors": [{"component": "query", "error": reason}],
        "arms": {
            arm: {"status": "fallback", "error": reason, "parent_arm": "E0", "scores": pair["arms"]["P_pair"]["scores"]}
            for arm in ARMS
        },
    }


def compare(row, backend, name, billing, task, targets, traces, view, original, selected, parent_scores, parent_arm):
    maps = {c: frame_map(traces[str(c)]) for c in selected}
    ids = {c: [f["id"] for f in maps[c]] for c in selected}
    comparisons, errors = [], []
    for a, b in itertools.permutations(selected, 2):
        try:
            value, raw = backend.call(
                f"{row['qid']}/{name}/{a}/{b}",
                COMPARE.format(
                    task=compact(task), a_frames=compact(maps[a]), b_frames=compact(maps[b]), memory=compact(view)
                ),
                [targets[a], targets[b]],
                [traces[str(a)], traces[str(b)]],
                Comparison,
                CAPS["comparison"],
                billing,
                lambda x, a=a, b=b: validate_comparison(x, a, b, ids, view),
                references={
                    "a_evidence": allowed_references(a, ids, view),
                    "b_evidence": allowed_references(b, ids, view),
                },
            )
            comparisons.append(
                {"a": a, "b": b, "result": value, "attempt": raw["attempt"], "response_path": raw.get("response_path")}
            )
        except SemanticFailure as error:
            errors.append({"a": a, "b": b, "error": str(error)})
    if errors:
        return {
            "status": "fallback",
            "error": str(errors),
            "parent_arm": parent_arm,
            "scores": parent_scores,
            "comparisons": comparisons,
            "invalid_comparisons": errors,
        }
    scores, ranking = rerank(original, selected, comparisons)
    return {"status": "complete", "scores": scores, "ranking": ranking, "comparisons": comparisons}


def memory_dependencies(row, backend, targets, traces, selected):
    """Query-independent requests shared by live inference and exact cache preflight."""
    qid = str(row["qid"])
    store = FactStore()
    for c in selected:
        ids = [f["id"] for f in frame_map(traces[str(c)])]

        def validate_facts(value, candidate=c, refs=ids):
            store.extract(candidate, value, refs)
            return value

        backend.call(
            f"{qid}/facts/{c}",
            FACTS.format(frames=compact(frame_map(traces[str(c)]))),
            [targets[c]],
            [traces[str(c)]],
            Facts,
            CAPS["facts"],
            ["M_plain", "M_decision"],
            validate_facts,
            references={"FactItem.frames": ids},
        )
    memory, raw = backend.call(
        f"{qid}/memory/M_decision",
        MEMORY["M_decision"].format(facts=compact(store.dump())),
        [],
        {"fact_store_sha256": store.version},
        MemoryDraft,
        CAPS["memory"],
        ["M_decision"],
        lambda x: build_memory(x, store.dump(), backend.tokenizer),
        references={"MemoryItem.fact_ids": list(store.dump())},
    )
    return store.dump(), memory, raw


def one_query(row, binding, binding_hash, backend, frozen, prepared=None):
    from candidate_frames import prepare_candidates

    baseline, pair = parents(binding, row)
    selected = pair["candidate_ids"]
    original = baseline["prediction"]["pred_saliency_scores"]
    targets, traces, seconds = prepared if prepared is not None else prepare_candidates(row, baseline, selected)
    if traces != frozen["candidate_frames"]:
        raise ValueError("Frozen R2 candidate pixels/PTS changed")
    result = fallback(row, binding, binding_hash, "dependency_not_completed")
    result.update(candidate_frames=traces, decode_seconds=seconds, errors=[], memories={}, observations={})
    task, raw = extract_task(row, backend, R1_ARMS, CAPS["task"])
    result.update(task=task, task_response_path=raw.get("response_path"))
    memory = None
    try:
        facts, memory, raw = memory_dependencies(row, backend, targets, traces, selected)
        result["facts"] = facts
        result["memories"]["E_memory"] = memory
        result["memory_response_path"] = raw.get("response_path")
    except SemanticFailure as error:
        result["errors"].append({"component": "shared_memory", "error": str(error)})
    result["arms"]["C_direct4"] = compare(
        row,
        backend,
        "C_direct4",
        ["C_direct4"],
        task,
        targets,
        traces,
        {},
        original,
        selected,
        result["E0"],
        "E0",
    )
    if memory is None:
        return result
    view = retrieve(memory, backend.tokenizer)
    base = compare(
        row,
        backend,
        "M_decision",
        ["M_decision"],
        task,
        targets,
        traces,
        view,
        original,
        selected,
        result["E0"],
        "E0",
    )
    result["arms"]["E_memory"] = base
    for arm in OBSERVERS:
        result["arms"][arm] = {**deepcopy(base), "origin": "E_memory_without_observation"}
        result["memories"][arm] = deepcopy(memory)
    if base["status"] != "complete":
        return result
    interface = binding.get("evidence_interface", "legacy")
    if interface not in {"legacy", "temporal_v1"}:
        raise ValueError("Unknown observation evidence interface")
    return observe_and_compare(
        row, backend, frozen, result, task, targets, traces, memory, baseline,
        ground_observations=interface == "temporal_v1",
    )


def observe_and_compare(row, backend, frozen, result, task, targets, traces, memory, baseline, *, ground_observations=False):
    """Shared production/engineering path; engineering replay is never a formal prediction."""
    qid = str(row["qid"])
    base = result["arms"]["E_memory"]
    selected, original = result["candidate_ids"], result["original"]
    view = retrieve(memory, backend.tokenizer)
    gaps = actionable_gaps(base)
    result["planning"] = {"gaps": gaps, "called": False, "actions": [], "no_action_reason": "no_actionable_gap"}
    if not gaps:
        return result
    candidates = candidate_ranges(selected, float(row["duration"]))
    try:
        plan, raw = backend.call(
            f"{qid}/plan",
            PLAN.format(task=compact(task), candidates=compact(candidates), memory=compact(view), gaps=compact(gaps)),
            [],
            {"active_memory_sha256": digest(view), "gaps": gaps, "candidates": candidates},
            LocalPlan,
            384,
            OBSERVERS,
            lambda x: validate_plan(x, gaps, candidates),
        )
        resolution = resolve_plan(plan, gaps, candidates, task)
        result["planning"].update(
            **resolution,
            model_actions=plan["actions"],
            called=True,
            no_action_reason=None,
            response_path=raw.get("response_path"),
        )
        plan = resolution
    except SemanticFailure as error:
        for arm in OBSERVERS:
            result["arms"][arm] = {
                "status": "fallback",
                "parent_arm": "E_memory",
                "scores": base["scores"],
                "error": "planner: " + str(error),
            }
        result["planning"].update(called=True, error=str(error))
        return result
    if not plan["actions"]:
        result["planning"]["no_action_reason"] = "planner_no_executable_action"
        return result
    reader, timestamps, seen = source_timeline(row, baseline)
    if timeline_hash(timestamps) != frozen["source_timeline_sha256"] or sorted(seen) != frozen["B0_seen_indices"]:
        raise ValueError("Source timeline or complete parent seen-frame list drifted")
    paired = pair_actions(timestamps, plan["actions"], seen)
    result["paired_actions"] = paired
    observations = {arm: [] for arm in OBSERVERS}
    failed = []
    for action_pair in paired:
        if action_pair["cancelled"]:
            continue
        payloads = {}
        try:
            for arm, key in zip(OBSERVERS, ("uniform", "targeted"), strict=True):
                payloads[arm] = read_frames(
                    reader,
                    timestamps,
                    action_pair[key],
                    row,
                    action_pair["action"],
                    baseline["media"]["resized_height_width"],
                    arm,
                )
        except Exception as error:
            action_pair.update(cancelled=True, reason="paired_decode_failure: " + str(error))
            failed.append(action_pair["reason"])
            break
        for arm in OBSERVERS:
            target, trace = payloads[arm]
            ids = [f["id"] for f in trace["frames"]]

            def validate_observation(value, ids=ids):
                ObservationFacts.model_validate(value)
                isolated = FactStore()
                isolated.extract("observation", value, ids)
                return value

            try:
                value, raw = backend.call(
                    f"{qid}/{arm}/observe/{action_pair['number']}",
                    OBSERVE.format(
                        question=action_pair["action"]["question"],
                        frames=compact([{"id": f["id"], "pts": f["pts"]} for f in trace["frames"]]),
                    ),
                    [target],
                    [trace],
                    ObservationFacts,
                    CAPS["observation"],
                    [arm],
                    validate_observation,
                    references={"ObservationFact.frames": ids},
                )
                observations[arm].append(
                    {
                        "action": action_pair["action"],
                        "trace": trace,
                        "result": value,
                        "response_path": raw.get("response_path"),
                        "attempt": raw["attempt"],
                    }
                )
            except SemanticFailure as error:
                failed.append(f"{arm} observation: {error}")
        if failed:
            action_pair.update(cancelled=True, reason="paired_observation_contract_failure")
            break
    del reader
    result["observations"] = observations
    if failed:
        for arm in OBSERVERS:
            result["arms"][arm] = {
                "status": "fallback",
                "parent_arm": "E_memory",
                "scores": base["scores"],
                "error": str(failed),
            }
        result["errors"].append({"component": "paired_observation", "error": str(failed)})
        return result
    if not observations["E_uniform"]:
        result["planning"]["no_action_reason"] = "all_actions_cancelled_for_frame_budget"
        return result
    for arm in OBSERVERS:
        target_intervals = {item["candidate"]: item["target"] for item in candidates} if ground_observations else None
        updated = update_memory(memory, observations[arm], backend.tokenizer, target_intervals=target_intervals)
        result["memories"][arm] = updated
        result["arms"][arm] = compare(
            row,
            backend,
            arm,
            [arm],
            task,
            targets,
            traces,
            retrieve(updated, backend.tokenizer),
            original,
            selected,
            base["scores"],
            "E_memory",
        )
    return result


def charge_fallbacks(row, binding, record, ledger):
    _, pair = parents(binding, row)
    tokens = sum(pair["arms"]["P_pair"]["cost"][k] for k in ("input_tokens", "output_tokens"))
    for arm in ARMS:
        value = record["arms"][arm]
        if value["status"] == "fallback" and value["parent_arm"] == "E0":
            ledger.fallback_cost(row["qid"], arm, tokens)


def finish(run, binding, rows, binding_hash, stopped=None):
    directory = run / "R2"
    ledger = SharedLedger(directory, binding["budgets"])
    receipts = {}
    for row in rows:
        path = directory / "records" / f"{row['qid']}.json"
        if not path.exists():
            record = fallback(row, binding, binding_hash, stopped or "missing_record")
            charge_fallbacks(row, binding, record, ledger)
            write_json(path, record)
        receipts[str(row["qid"])] = sha256(path)
    write_json(directory / "record_receipt.json", receipts)
    write_json(
        directory / "inference_complete.json",
        {
            "status": "partial_stopped" if stopped else "inference_complete",
            "stopped_reason": stopped,
            "binding_sha256": binding_hash,
            "queries": len(rows),
            "finished_at": utc_now(),
            "record_receipt_sha256": sha256(directory / "record_receipt.json"),
            "ledger_sha256": sha256(directory / "ledger.json"),
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--worker-id", required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    binding, rows, binding_hash = load_run(run)
    import torch

    torch.set_num_threads(2)
    actual = "GPU-" + str(torch.cuda.get_device_properties(0).uuid).removeprefix("GPU-")
    if torch.cuda.device_count() != 1 or actual != os.environ["REZE_EXPECTED_GPU_UUID"]:
        raise ValueError("Wrong worker GPU")
    model = read_json(run / "model_verification.json")
    for name, expected in model["stats"].items():
        stat = (Path(MODEL) / name).stat()
        if stat.st_size != expected["size"] or stat.st_mtime_ns != expected["mtime_ns"]:
            raise ValueError("Frozen model files changed")
    directory = run / "R2"
    ledger = SharedLedger(directory, binding["budgets"])
    status_path = directory / "workers" / f"{args.worker_id}.status.json"
    backend = Backend(directory, ledger, read_json(run / "reuse_index.json"), status_path=status_path)
    frozen = {r["qid"]: r for r in legacy.read_jsonl(run / "candidate_inputs.jsonl")}
    for row in rows:
        if (directory / "stop_request.json").exists():
            break
        with claim_query(directory, row["qid"]) as claimed:
            if not claimed:
                continue
            try:
                record = one_query(row, binding, binding_hash, backend, frozen[row["qid"]])
            except SemanticFailure as error:
                record = fallback(row, binding, binding_hash, str(error))
            except BudgetStop as error:
                write_json(directory / "stop_request.json", {"reason": str(error), "at": utc_now()})
                break
            charge_fallbacks(row, binding, record, ledger)
            record.update(completed_at=utc_now(), worker_id=args.worker_id, gpu_uuid=actual)
            write_json(directory / "records" / f"{row['qid']}.json", record)
            write_json(status_path, {"status": "query_complete", "qid": row["qid"], "updated_at": utc_now()})
            print(f"R2 worker={args.worker_id} qid={row['qid']} complete", flush=True)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError:
        import traceback

        traceback.print_exc()
        raise SystemExit(75)
