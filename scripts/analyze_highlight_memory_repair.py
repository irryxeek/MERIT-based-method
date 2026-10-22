"""Read-only engineering and metric comparison of completed frozen runs."""

import argparse
from collections import Counter
from pathlib import Path

from highlight_evidence.granularity import ARMS, paired, verify
from highlight_evidence.pilot import read

from merit_qvh.core import file_sha256, write_json


def analyze(root, before, witness, output):
    if output.exists():
        raise FileExistsError("Keep previous analyses; use a new output")
    config, protocol = verify(root, labels=True)
    verify(before, labels=True)
    if any(read(r / "progress.json")["status"] != "complete" for r in (root, before)):
        raise ValueError("Both runs must be complete")
    protected = read(witness)
    if Path(protected["old_run"]) != before:
        raise ValueError("Wrong prior-run witness")
    for name, expected in protected["files"].items():
        if file_sha256(before / name) != expected:
            raise ValueError("Previous artifact changed: " + name)
    inputs = sorted((root / "inputs").glob("*.json"))
    for path in inputs:
        if file_sha256(path) != file_sha256(before / "inputs" / path.name):
            raise ValueError("Input differs from the matched run: " + path.name)
    records = {read(p)["qid"]: read(p) for p in (root / "queries").glob("*.json")}
    old_records = {read(p)["qid"]: read(p) for p in (before / "queries").glob("*.json")}
    metrics = {}
    for label, run in (("before", before), ("after", root)):
        for row in read(run / "per_query.json"):
            metrics.setdefault(row["qid"], {}).update({f"{a}_{label}": row["metrics"][a] for a in ARMS})
    ledger = [read(p) for p in (root / "requests").glob("*/*.json")]
    attempts = [a for r in ledger for a in r["attempts"]]
    if any(r["status"] == "pending" for r in ledger):
        raise ValueError("Pending request in completed run")
    result = {
        "protocol_id": protocol,
        "previous_protocol_id": read(before / "protocol.json")["protocol_id"],
        "unchanged_input_files": len(inputs),
        "unchanged_previous_artifacts": len(protected["files"]),
        "groups": {},
        "request_statuses": dict(Counter(r["status"] for r in ledger)),
        "attempt_statuses": dict(Counter(a["status"] for a in attempts)),
        "finish_reasons": dict(
            Counter(a["response"]["choices"][0]["finish_reason"] for a in attempts if "response" in a)
        ),
        "worker_elapsed_seconds": read(root / "progress.json")["elapsed_seconds"],
        "cost": read(root / "report.json")["cost"],
        "note": "Joint engineering repair on explored train20. Completion is not semantic accuracy. "
        "Paired source-video bootstrap uses 10000 resamples, seed42. No new holdout evaluation.",
    }
    for arm in ARMS:
        requests = [r for r in ledger if r["stage"] == arm + "_memory"]
        chunks = [read(p) for p in (root / "memories" / arm).glob("*/*.json")]
        qids = config["active_qids"]
        old_failed = [q for q in qids if old_records[q][arm]["status"] == "fallback"]
        first_pass = sum(
            r["status"] == "ok" and len(r["attempts"]) == 1 and "validation_error" not in r["attempts"][0]
            for r in requests
        )
        result["groups"][arm] = {
            "active_queries": len(qids),
            "memory_requests": len(requests),
            "memory_first_attempt_valid": first_pass,
            "memory_terminal_statuses": dict(Counter(r["status"] for r in requests)),
            "memory_with_format_retry": sum(
                any(a["format_attempt"] for a in r["attempts"]) for r in requests
            ),
            "field_patch_attempts": sum(
                a.get("mode") == "field_patch" for r in requests for a in r["attempts"]
            ),
            "window_quality": dict(Counter(x["quality"] for x in chunks)),
            "downgraded_cues": sum(len(x["downgraded_cues"]) for x in chunks),
            "rejected_links": sum(len(x["rejected_links"]) for x in chunks),
            "queries_with_complete_memory": sum(records[q][arm]["memory_coverage_complete"] for q in qids),
            "queries_completing_E1": sum(records[q][arm]["status"] in {"ok", "partial"} for q in qids),
            "old_failed_queries": old_failed,
            "old_failed_now_complete": [
                q for q in old_failed if records[q][arm]["status"] in {"ok", "partial"}
            ],
            "full20_before_metrics": read(before / "report.json")["groups"][arm]["metrics"][
                "HL-min-VeryGood"
            ],
            "full20_after_metrics": read(root / "report.json")["groups"][arm]["metrics"]["HL-min-VeryGood"],
            "paired_full20_after_minus_before": paired(
                metrics, records, arm + "_after", arm + "_before", sorted(records)
            ),
            "validation_errors": [
                {
                    "request_id": r["request_id"],
                    "format_attempt": a["format_attempt"],
                    "error": a["validation_error"],
                }
                for r in requests
                for a in r["attempts"]
                if "validation_error" in a
            ],
        }
    write_json(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--witness", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(
        args.root.resolve(), args.before.resolve(), args.witness.resolve(), args.output.resolve()
    )
    print(
        {
            k: result[k]
            for k in ("request_statuses", "attempt_statuses", "finish_reasons", "worker_elapsed_seconds")
        }
    )
