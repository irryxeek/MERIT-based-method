"""Engineering replay of the seven parser failures and the first empty conflict plan.

The old comparison state is an explicitly labeled engineering fixture. This
script produces NO benchmark prediction or accuracy metric.
"""

import argparse
import os
import time
from copy import deepcopy
from pathlib import Path

from backend import MODEL, Backend, SemanticFailure
from io_utils import ROOT, digest, legacy, parents, read_json, sha256, utc_now, write_json
from r2_evaluate import audit_observation_request
from r2_observer import source_timeline
from r2_protocol import ARMS, OBSERVERS, R1_ARMS, SharedLedger
from r2_worker import observe_and_compare
from task_grounding import extract_task


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    manifest = read_json(run / "manifest.json")
    for name, expected in manifest["source_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError("Frozen smoke source changed")
    for name, expected in manifest["parent_receipts"].items():
        if sha256(name) != expected:
            raise ValueError("Frozen smoke parent changed")
    started = time.monotonic()
    import torch
    from candidate_frames import prepare_candidates

    torch.set_num_threads(2)
    actual = "GPU-" + str(torch.cuda.get_device_properties(0).uuid).removeprefix("GPU-")
    if torch.cuda.device_count() != 1 or actual != os.environ["REZE_EXPECTED_GPU_UUID"]:
        raise ValueError("Unexpected smoke GPU")
    parent = Path(manifest["parent_run"])
    model = read_json(parent / "model_verification.json")
    for name, expected in model["stats"].items():
        stat = (Path(MODEL) / name).stat()
        if stat.st_size != expected["size"] or stat.st_mtime_ns != expected["mtime_ns"]:
            raise ValueError("Frozen smoke model changed")
    limits = {"requests": 53, "extra_attempts": 0, "arm_incremental_tokens": {a: 1000000 for a in ARMS}}
    ledger = SharedLedger(run / "smoke", limits)
    backend = Backend(run / "smoke", ledger)
    rows = {r["qid"]: r for r in legacy.read_jsonl(parent / "inference.jsonl")}
    tasks, failures = {}, {}
    for qid in manifest["task_qids"]:
        try:
            task, raw = extract_task(rows[qid], backend, R1_ARMS)
            tasks[qid] = task
            write_json(run / "tasks" / f"{qid}.json", {"task": task, "response_path": raw["response_path"]})
        except SemanticFailure as error:
            failures[qid] = str(error)
        print(f"Engineering task qid={qid} valid={qid in tasks}", flush=True)
    checks = {"all_eight_tasks_valid": len(tasks) == 8 and not failures}
    qid = manifest["observation_qid"]
    if qid in tasks:
        row = rows[qid]
        binding = read_json(parent / "binding.json")
        baseline, pair = parents(binding, row)
        targets, traces, _ = prepare_candidates(row, baseline, pair["candidate_ids"])
        frozen = next(r for r in legacy.read_jsonl(parent / "candidate_inputs.jsonl") if r["qid"] == qid)
        if traces != frozen["candidate_frames"]:
            raise ValueError("Smoke input evidence drift")
        record = deepcopy(read_json(parent / "R2/records" / f"{qid}.json"))
        record.update(
            task=tasks[qid],
            engineering_replay=True,
            fixture_description="Frozen old E_memory comparison state; new parser; real source observations",
        )
        record = observe_and_compare(
            row, backend, frozen, record, tasks[qid], targets, traces, record["memories"]["E_memory"], baseline
        )
        write_json(run / "smoke_record.json", record)
        observations = record.get("observations", {})
        checks["two_real_observations_each"] = all(len(observations.get(a, [])) == 2 for a in OBSERVERS)
        checks["both_comparisons_complete"] = all(record["arms"][a]["status"] == "complete" for a in OBSERVERS)
        checks["observations_have_visible_facts"] = (
            all(
                any(f["kind"] != "unknown" for f in o["result"]["facts"])
                for a in OBSERVERS
                for o in observations.get(a, [])
            )
            and checks["two_real_observations_each"]
        )
        reader, timestamps, seen = source_timeline(row, baseline)
        for a in OBSERVERS:
            for i, observation in enumerate(observations.get(a, [])):
                planned = record["paired_actions"][i]
                audit_observation_request(
                    read_json(observation["response_path"]),
                    planned["action"],
                    planned["uniform" if a == "E_uniform" else "targeted"],
                    timestamps,
                    seen,
                )
        del reader
        checks["paired_question_and_frames"] = (
            all(
                u["action"] == t["action"] and len(u["trace"]["frames"]) == len(t["trace"]["frames"]) == 24
                for u, t in zip(observations.get("E_uniform", []), observations.get("E_targeted", []), strict=True)
            )
            and checks["two_real_observations_each"]
        )
    else:
        checks["real_observation_completed"] = False
    seconds = time.monotonic() - started
    prior_seconds = manifest.get("historical_gpu_seconds", 0)
    checks["within_one_gpu_hour"] = seconds + prior_seconds <= 3600
    gate = {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "failures": failures,
        "actual_new_attempts": len(ledger.state["attempts"]),
        "actual_compute_tokens": ledger.cost(),
        "gpu_seconds_until_gate": seconds,
        "historical_gpu_seconds": prior_seconds,
        "cumulative_gpu_seconds_until_gate": seconds + prior_seconds,
        "historical_new_attempts": manifest.get("historical_new_attempts", 0),
        "finished_at": utc_now(),
        "source_manifest_sha256": sha256(run / "manifest.json"),
        "ledger_sha256": sha256(run / "smoke/ledger.json"),
        "accuracy_evaluated": False,
    }
    write_json(run / "gate.json", gate)
    print(digest(gate), gate, flush=True)
    if gate["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
