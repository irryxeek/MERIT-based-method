"""Post-completion audit of raw scores, evidence lineage, paired inputs and actual cost."""

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/reze_evidence"))
import protocol  # noqa: E402
from candidate_frames import eligible_observations  # noqa: E402


def main():
    store = protocol.STORE
    base = store / "runs/stage_a_b0"
    pairs = store / "runs/stage_a_pairs"
    binding, rows = protocol.load_run(pairs)
    report = json.loads((store / "stage_a_results/evaluation.json").read_text())
    assert len(rows) == report["queries"] == 100
    summaries = {}
    for run in (base, pairs):
        run_binding, _ = protocol.load_run(run)
        completion = json.loads((run / "completion.json").read_text())
        assert completion["queries"] == 100 and completion["status"] == "complete"
        assert completion["binding_sha256"] == protocol.sha256(run / "binding.json")
        assert completion["record_receipt_sha256"] == protocol.sha256(run / "record_receipt.json")
        receipt = json.loads((run / "record_receipt.json").read_text())
        assert set(receipt) == {str(row["qid"]) for row in rows}
        for qid, expected in receipt.items():
            assert protocol.sha256(run / "records" / f"{qid}.json") == expected
        start = datetime.fromisoformat(run_binding["created_at"])
        end = datetime.fromisoformat(completion["completed_at"])
        summaries[run.name] = {
            "wall_seconds": (end - start).total_seconds(),
            "allocated_gpu_hours": completion["allocated_gpu_hours"],
            "attempted_requests": completion["attempted_requests"],
            "worker_restarts": completion["worker_restarts"],
            "physical_gpus": [w["physical_index"] for w in run_binding["workers"]],
        }
    matched, source_frames, candidate_frames = 0, 0, 0
    input_tokens = {a: 0 for a in ("B0", "P_abs", "P_pair")}
    output_tokens = dict(input_tokens)
    fallback = {a: [] for a in ("P_abs", "P_pair")}
    for row in rows:
        baseline = protocol.read_baseline(base, row)
        record = json.loads((pairs / "records" / f"{row['qid']}.json").read_text())
        protocol.validate_pair_record(record, row, binding, protocol.sha256(pairs / "binding.json"), baseline)
        source_frames += baseline["media"]["model_input_frames"]
        input_tokens["B0"] += sum(x["prompt_tokens"] for x in baseline["clips"])
        output_tokens["B0"] += len(baseline["clips"])
        for candidate in record["candidate_ids"]:
            trace = record["candidate_frames"].get(str(candidate))
            if trace is None:
                continue
            expected = eligible_observations(baseline["media"], candidate)
            observed = [(f["requested_time"], f["baseline_window"], f["baseline_position"]) for f in trace["frames"]]
            assert observed == expected
            assert 1 <= trace["frames_observed"] <= 4
            assert trace["native_temporal_padding_frames"] == trace["frames_observed"] % 2
            assert len(trace["metadata"]["frames_indices"]) == trace["frames_observed"]
            for frame, index in zip(trace["frames"], trace["metadata"]["frames_indices"], strict=True):
                w, p = frame["baseline_window"], frame["baseline_position"]
                assert frame["frame_index"] == baseline["media"]["frame_indices"][w][p]
                assert frame["frame_timestamps"] == baseline["media"]["frame_timestamps"][w][p]
                assert abs(index / trace["metadata"]["fps"] - (frame["requested_time"] - trace["start"])) <= 0.00000051
            candidate_frames += trace["frames_observed"]
        arm_requests = {}
        for arm in ("P_abs", "P_pair"):
            result = record["arms"][arm]
            arm_requests[arm] = {(r["a"], r["b"]): r for r in result["requests"]}
            input_tokens[arm] += sum(x["prompt_tokens"] for x in result["requests"])
            output_tokens[arm] += sum(x["output_tokens"] for x in result["requests"])
            if result["status"] == "fallback":
                fallback[arm].append({"qid": row["qid"], "reason": result["error"]})
        assert set(arm_requests["P_abs"]) == set(arm_requests["P_pair"])
        for key, first in arm_requests["P_abs"].items():
            second = arm_requests["P_pair"][key]
            assert first["evidence_sha256"] == second["evidence_sha256"]
            assert first["model_frames"] == second["model_frames"]
            matched += 1
    for arm in input_tokens:
        assert report["results"][arm]["incremental_cost"]["input_tokens"] == input_tokens[arm]
        assert report["results"][arm]["incremental_cost"]["output_tokens"] == output_tokens[arm]
    engineering = json.loads((store / "engineering_gate.json").read_text())
    result = {
        "audited_at": protocol.utc_now(),
        "queries": 100,
        "passed": True,
        "raw_logits_and_rankings_replay": True,
        "source_receipts_valid": True,
        "paired_requests_with_identical_evidence": matched,
        "all_target_frames_previously_seen_by_B0": True,
        "time_encoding_max_error_seconds": 0.00000051,
        "run_costs": summaries,
        "engineering": engineering,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "fallbacks": fallback,
        "frame_accounting": {
            "baseline_decoded_frame_occurrences": source_frames,
            "pair_preparation_redecodes_the_baseline_samples": source_frames,
            "candidate_observations_per_query_summed_once": candidate_frames,
            "note": "Pair preparation decodes the already observed baseline samples again to verify byte-identical input. "
            "Model frame counts are presentations, not unique encoder executions; GPU encoder reuse may occur.",
        },
        "total_reserved_requests_including_engineering_failures": engineering["all_engineering_attempted_requests"]
        + sum(s["attempted_requests"] for s in summaries.values()),
        "audit_script_sha256": protocol.sha256(__file__),
    }
    protocol.atomic_json(store / "stage_a_results/execution_accounting.json", result)
    protocol.atomic_json(store / "audit/stage_a_integrity.json", result)
    destination = ROOT / "results/reze_evidence_v1"
    shutil.copytree(store / "stage_a_results", destination / "stage_a", dirs_exist_ok=True)
    shutil.copytree(store / "audit", destination / "audit", dirs_exist_ok=True)
    for run in sorted((store / "runs").iterdir()):
        if not (run / "binding.json").exists():
            continue
        archived = destination / "runs" / run.name
        archived.mkdir(parents=True, exist_ok=True)
        for name in ("binding.json", "completion.json", "inference.jsonl", "record_receipt.json", "supervisor.json"):
            if (run / name).exists():
                shutil.copy2(run / name, archived / name)
        for path in run.glob("attempts_*.jsonl"):
            shutil.copy2(path, archived / path.name)
        shutil.copytree(
            run / "source_snapshot",
            archived / "source_snapshot",
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    print(
        json.dumps(
            {
                "passed": True,
                "matched_requests": matched,
                "run_costs": summaries,
                "total_reserved_requests": result["total_reserved_requests_including_engineering_failures"],
            }
        )
    )


if __name__ == "__main__":
    main()
