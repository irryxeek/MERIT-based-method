import _bootstrap  # noqa: F401 -- establish the isolated implementation import
import argparse
import bisect
import json
import sqlite3
from pathlib import Path

import numpy as np
from videoarm.qvhighlights.common import (
    ARTIFACTS,
    atomic_json,
    load_manifest,
    read_jsonl,
    source_id,
)
from score_qvh import metric_keys, score


def cluster_bootstrap(rows, n=10000):
    """Resample sources, retaining all queries per selected source; query-weighted mean."""
    groups = {}
    for source, diff in rows:
        groups.setdefault(source, []).append(diff)
    if not groups:
        return {"estimate": None, "ci95": None, "clusters": 0}
    total = np.array([sum(v) for v in groups.values()])
    counts = np.array([len(v) for v in groups.values()])
    rng = np.random.default_rng(42)
    estimates = []
    for start in range(0, n, 500):
        ids = rng.integers(0, len(groups), size=(min(500, n - start), len(groups)))
        estimates.extend((total[ids].sum(1) / counts[ids].sum(1)).tolist())
    return {
        "estimate": float(total.sum() / counts.sum()),
        "ci95": np.quantile(estimates, [0.025, 0.975]).tolist(),
        "clusters": len(groups),
        "queries": sum(counts).item(),
        "resamples": n,
        "seed": 42,
        "unit": "source YouTube video; all its queries stay together",
        "warning": "small development set; CI excludes systematic model and annotation bias"
        if len(groups) <= 50
        else None,
    }


def costs(directory):
    db = sqlite3.connect(Path(directory) / "state.sqlite")
    db.row_factory = sqlite3.Row
    items = list(db.execute("SELECT * FROM items"))
    budgets = [json.loads(r["budget"]) for r in items]
    reqs = list(db.execute("SELECT * FROM requests"))
    responses = [json.loads(r["response"]) for r in reqs if r["response"]]
    times = [b["active_seconds"] for b in budgets if b["active_seconds"] > 0]
    failures = [
        {"qid": r["qid"], "status": r["status"], "error": r["error"]}
        for r in items
        if r["status"] != "succeeded"
    ]
    frequency = {}
    stop = 0
    for r in items:
        s = json.loads(r["snapshot"])
        for a in s.get("actions", []):
            frequency[a["action"]] = frequency.get(a["action"], 0) + 1
        stop += s.get("stop_reason") == "model_stop"
    output = {
        "planned": len(items),
        "succeeded": len(items) - len(failures),
        "failures": failures,
        "totals": {
            k: sum(b.get(k, 0) for b in budgets)
            for k in (
                "requests",
                "controllers",
                "images",
                "input_tokens",
                "output_tokens",
                "local_frames",
                "active_seconds",
                "retries",
                "cache_hits",
                "shared_cache_hits",
                "actual_requests",
                "actual_images",
                "actual_input_tokens",
                "actual_output_tokens",
                "uncertain_requests",
            )
        },
        "unique_frames_sum_over_queries": sum(len(b["frame_ids"]) for b in budgets),
        "elapsed_p50_p95_seconds": np.quantile(times, [0.5, 0.95]).tolist()
        if times
        else None,
        "tool_frequency": frequency,
        "early_stop_queries": stop,
        "requests_with_uncertain_usage": sum(
            r["status"] in {"error", "uncertain", "inflight"} for r in reqs
        ),
        "format_or_contract_failures": sum(
            r.get("parse_status") == "failed" for r in responses
        ),
        "empty_mr_responses": sum(
            "empty MR candidates" in r.get("parse_error", "") for r in responses
        ),
        "truncated_responses": sum(
            r.get("finish_reason") == "length" for r in responses
        ),
        "budget_interrupted_queries": sum(
            json.loads(r["snapshot"]).get("budget_interrupted", False) for r in items
        ),
        "mr_candidates_without_direct_sample_pts": sum(
            not c["direct_sample_pts"]
            for r in items
            for c in json.loads(r["snapshot"]).get("candidate_evidence_audit", [])
        ),
        "cold_equivalent_model_request_seconds": sum(
            r.get("model_elapsed_seconds", 0) for r in responses
        ),
        "peak_gpu_memory_mib": None,
        "gpu_hours": None,
        "gpu_telemetry_note": "requires recorded QVH-owned service telemetry; cannot derive peak or GPU hours from token counts",
    }
    execution = Path(directory) / "execution.json"
    actual_wall = json.loads(execution.read_text()).get("total_wall_seconds") if execution.exists() else None
    output["runner_wall_seconds"] = actual_wall
    output["query_workers"] = json.loads(execution.read_text()).get("query_workers", 1) if execution.exists() else 1
    telemetry_path = ARTIFACTS / "logs/qvh_gpu_telemetry.jsonl"
    if telemetry_path.exists():
        telemetry = read_jsonl(telemetry_path)
        timestamps = [r["time"] for r in telemetry]
        observed = []
        for r in items:
            for interval in json.loads(r["snapshot"]).get("executions", []):
                if interval["end_unix"] is not None:
                    observed += telemetry[
                        bisect.bisect_left(
                            timestamps, interval["start_unix"]
                        ) : bisect.bisect_right(timestamps, interval["end_unix"])
                    ]
        output["peak_gpu_memory_mib"] = max(
            (sum(g["memory_used_mib"] for g in r["gpus"]) for r in observed),
            default=None,
        )
        output["gpu_hours"] = 2 * (actual_wall if actual_wall is not None else output["totals"]["active_seconds"]) / 3600
        output["gpu_telemetry_note"] = (
            "TP2 allocated GPU hours from recorded runner wall time for parallel runs (query latency sums are not GPU time); legacy serial runs use active query time; excludes service startup/idle; 5-second peak samples may miss short spikes"
        )
    db.close()
    return output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", default=[])
    ap.add_argument("--b1-score")
    ap.add_argument("--m1-score")
    ap.add_argument("--manifest")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    report = {"costs": {p: costs(p) for p in a.run}}
    if a.b1_score and a.m1_score:
        b1_result = json.loads(Path(a.b1_score).read_text())
        m1_result = json.loads(Path(a.m1_score).read_text())
        if (
            b1_result["manifest_sha256"] != m1_result["manifest_sha256"]
            or b1_result["task"] != m1_result["task"]
        ):
            raise ValueError("paired runs have different manifests/tasks")
        b1 = {
            r["qid"]: r
            for r in read_jsonl(Path(a.b1_score).with_suffix(".per_query.jsonl"))
        }
        m1 = {
            r["qid"]: r
            for r in read_jsonl(Path(a.m1_score).with_suffix(".per_query.jsonl"))
        }
        common = set(b1) & set(m1)
        manifest = load_manifest(a.manifest)
        metrics = (
            list(next(iter(b1.values()))["metrics"])
            if b1
            else list(next(iter(m1.values()))["metrics"])
            if m1
            else metric_keys(b1_result["task"])
        )
        report["paired"] = {
            "planned": len(manifest),
            "common_success": len(common),
            "common_coverage": len(common) / len(manifest),
            "common_success_ci": {},
            "custom_full_plan_failure_zero_ci": {},
        }
        if common:
            common_manifest = [r for r in manifest if r["qid"] in common]
            gt = read_jsonl(b1_result["gt_path"])
            report["paired"]["official_common_success_scores"] = {}
            for name, results in (("B1", b1_result), ("M1", m1_result)):
                predictions = [
                    r
                    for r in read_jsonl(results["predictions_path"])
                    if r["qid"] in common
                ]
                common_result, _ = score(
                    common_manifest, predictions, results["task"], gt
                )
                report["paired"]["official_common_success_scores"][name] = (
                    common_result["official_metrics"]
                )
        for k in metrics:
            report["paired"]["common_success_ci"][k] = cluster_bootstrap(
                [
                    (
                        m1[q]["source_id"],
                        100 * (m1[q]["metrics"][k] - b1[q]["metrics"][k]),
                    )
                    for q in sorted(common)
                ]
            )
            report["paired"]["custom_full_plan_failure_zero_ci"][k] = cluster_bootstrap(
                [
                    (
                        source_id(r["vid"])[0],
                        100
                        * (
                            m1.get(r["qid"], {"metrics": {k: 0}})["metrics"][k]
                            - b1.get(r["qid"], {"metrics": {k: 0}})["metrics"][k]
                        ),
                    )
                    for r in manifest
                ]
            )
    telemetry = ARTIFACTS / "checks/gpu_telemetry_summary.json"
    report["owned_service_gpu_telemetry"] = (
        json.loads(telemetry.read_text()) if telemetry.exists() else None
    )
    atomic_json(a.out, report)
    print(
        json.dumps({"saved": a.out, "runs": len(a.run), "paired": "paired" in report})
    )


if __name__ == "__main__":
    main()
