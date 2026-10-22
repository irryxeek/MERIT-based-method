"""Read-only pixel, prompt, candidate, and provenance audit of the matched run."""

import argparse
import copy
import json
from collections import Counter
from pathlib import Path

from highlight_evidence.granularity import ARMS, units_for, verify
from highlight_evidence.inference import visual_content
from highlight_evidence.pilot import load_frames, read

from merit_qvh.client import redact_images
from merit_qvh.core import digest, file_sha256, write_json


def audit_repairs(request, contract, counters):
    """Replay patches read-only; verify persisted results and valid-field freeze."""
    candidate, issues = None, []
    for attempt in request["attempts"]:
        content = attempt["payload"]["messages"][1]["content"]
        if content != request["payload"]["messages"][1]["content"]:
            raise ValueError("A retry changed the frozen memory prompt or pictures")
        for key in ("model", "seed", "temperature", "max_tokens"):
            if attempt["payload"][key] != request["payload"][key]:
                raise ValueError("A retry changed generation settings")
        if attempt["mode"] == "field_patch":
            schema, instruction = contract.repair_spec(candidate, issues)
            if attempt["payload"]["response_format"]["json_schema"]["schema"] != schema:
                raise ValueError("Retry schema does not restrict edits to invalid fields")
            if attempt["payload"]["messages"][-1]["content"] != instruction:
                raise ValueError("Retry lost precise validation feedback")
            counters["audited_field_patch_attempts"] += 1
        elif attempt["payload"]["response_format"]["json_schema"]["schema"] != contract.schema:
            raise ValueError("Memory attempt changed its output contract")
        response = attempt.get("response", {}).get("choices", [{}])[0]
        if response.get("finish_reason") != "stop":
            continue
        try:
            decoded = json.loads(response["message"]["content"])
            candidate = (
                contract.apply_patch(candidate, decoded, issues)
                if attempt["mode"] == "field_patch"
                else decoded
            )
            issues = contract.problems(candidate)
        except (ValueError, TypeError, KeyError):
            if "validation_error" not in attempt:
                raise ValueError("Malformed memory response lacks recorded validation error") from None
    if request["status"] == "ok":
        expected = contract.finalize(candidate, degrade=True)
        expected["request_id"] = request["request_id"]
        expected["repair_attempted"] = any(a["format_attempt"] > 0 for a in request["attempts"])
        expected["quality"] = (
            "degraded" if expected["downgraded_cues"] or expected["rejected_links"] else "validated"
        )
        if request["result"] != expected:
            raise ValueError("Persisted memory result differs from validated/explicitly downgraded response")
        counters["audited_memory_results"] += 1


def normalized_payload(record):
    payload = copy.deepcopy(record["payload"])
    content = payload["messages"][-1]["content"]
    instruction, raw = content[0]["text"].split("\n", 1)
    data = json.loads(raw)
    if record["stage"].endswith("_memory"):
        data.pop("observation_units")
        data.pop("schema_example_first_unit_only", None)
        # v2 schema is derived from the treatment's unit IDs and their pixels.
        # The full schema is independently rebuilt and verified below.
        if payload.get("response_format", {}).get("type") == "json_schema":
            payload.pop("response_format")
    else:
        data.pop("memory")
    content[0]["text"] = instruction + "\n" + json.dumps(data, sort_keys=True)
    return digest(payload)


def audit(root, output):
    if output.exists() or root == output or root in output.parents:
        raise ValueError("Use a new audit directory outside the frozen run")
    c, protocol_id = verify(root)
    if read(root / "progress.json")["status"] != "complete":
        raise ValueError("Wait for the completed experiment before auditing")
    source = Path(c["source_run"])
    for name, expected in c["source_artifact_sha256"].items():
        if file_sha256(source / name) != expected:
            raise ValueError("An original experiment artifact changed: " + name)
    requests = {p.stem: read(p) for p in (root / "requests").glob("*/*.json")}
    counters = Counter()
    referenced = set()
    details = []
    for path in sorted((root / "inputs").glob("*.json")):
        data = read(path)
        q = data["row"]["qid"]
        record = read(root / "queries" / (str(q) + ".json"))
        if (
            record["selected_windows"] != data["selected_windows"]
            or record["selected_bin_ids"] != data["selected_bin_ids"]
        ):
            raise ValueError("Candidates changed after freezing")
        frames = load_frames(data["frames"]) if data["frames"] else []
        by_clip = {f["clip_id"]: f for f in frames}
        signatures = {a: {} for a in ARMS}
        for arm in ARMS:
            arm_record = record[arm]
            if not frames and (
                arm_record["requests"]
                or arm_record["pred_saliency_scores"] != data["D0"]["pred_saliency_scores"]
            ):
                raise ValueError("Inherited no-candidate sample was changed")
            for event in arm_record["requests"]:
                request = requests[event["request_id"]]
                referenced.add(event["request_id"])
                content = request["payload"]["messages"][-1]["content"]
                body = json.loads(content[0]["text"].split("\n", 1)[1])
                if request["stage"] == arm + "_memory":
                    interval = body["current_interval"]
                    window = next(w for w in data["windows"] if [w["start"], w["end"]] == interval)
                    expected_frames = [by_clip[i] for i in window["bin_ids"]]
                    if (
                        body["prior_observations"] != window["context"]
                        or body["value_definitions"] != c["values"]
                    ):
                        raise ValueError("Prior context/value definitions differ from the frozen input")
                    if c.get("memory_contract"):
                        from highlight_evidence.robust_memory import MemoryContract

                        contract = MemoryContract(
                            units_for(window, arm),
                            expected_frames,
                            c["values"],
                            data["row"]["duration"],
                            window["context"],
                        )
                        if request["payload"]["response_format"]["json_schema"]["schema"] != contract.schema:
                            raise ValueError("Structured output schema differs from the frozen unit contract")
                        audit_repairs(request, contract, counters)
                    key = ("memory", tuple(interval))
                    counters[arm + "_memory_requests"] += 1
                elif request["stage"] == arm + "_E1":
                    expected_frames = frames
                    if (
                        body["original_query"] != data["row"]["query"]
                        or [b["clip_id"] for b in body["requested_bins"]] != data["selected_bin_ids"]
                    ):
                        raise ValueError("Scorer changed query or requested bins")
                    key = ("E1",)
                    counters[arm + "_E1_requests"] += 1
                else:
                    raise ValueError("Unexpected stage in this E1-only experiment")
                expected_content = redact_images(visual_content("unused", expected_frames))[1:]
                if content[1:] != expected_content:
                    raise ValueError(
                        "Model request did not use the exact frozen pixels, timestamps, and order"
                    )
                signatures[arm][key] = normalized_payload(request)
                counters[arm + "_initial_image_presentations"] += len(expected_frames)
            changed = {
                i
                for i, (a, b) in enumerate(
                    zip(arm_record["pred_saliency_scores"], data["D0"]["pred_saliency_scores"], strict=True)
                )
                if a != b
            }
            if not changed <= set(data["selected_bin_ids"]):
                raise ValueError("An unselected score was changed")
            if c.get("memory_contract") and frames and not changed <= set(arm_record["valid_memory_bin_ids"]):
                raise ValueError("A failed memory window changed a D0 score")
        common = set(signatures["C10"]) & set(signatures["F2"])
        if any(signatures["C10"][k] != signatures["F2"][k] for k in common):
            raise ValueError("Paired requests differ beyond the memory granularity treatment")
        counters["matched_memory_request_pairs"] += sum(k[0] == "memory" for k in common)
        counters["matched_E1_request_pairs"] += sum(k[0] == "E1" for k in common)
        details.append(
            {
                "qid": q,
                "selected_windows": data["selected_windows"],
                "status": {a: record[a]["status"] for a in ARMS},
                "unpaired_requests": {a: [list(k) for k in signatures[a] if k not in common] for a in ARMS},
            }
        )
    unresolved = [p for p, r in requests.items() if r["status"] == "pending"]
    if unresolved:
        raise ValueError("Completed run contains pending requests")
    if referenced != set(requests):
        raise ValueError("Request ledger contains orphan requests missing from final query records")
    counters["actual_attempts"] = sum(len(r["attempts"]) for r in requests.values())
    counters["transport_errors"] = sum(
        a["status"] == "transport_error" for r in requests.values() for a in r["attempts"]
    )
    counters["length_finishes"] = sum(
        a.get("response", {}).get("choices", [{}])[0].get("finish_reason") == "length"
        for r in requests.values()
        for a in r["attempts"]
    )
    result = {
        "protocol_id": protocol_id,
        "status": "passed",
        "queries": len(details),
        "counters": dict(counters),
        "request_statuses": dict(Counter(r["status"] for r in requests.values())),
        "original_artifacts_unchanged": len(c["source_artifact_sha256"]),
        "queries_detail": details,
        "audit_source_sha256": file_sha256(__file__),
        "run_report_sha256": file_sha256(root / "report.json"),
        "note": "Initial requests match except current observation units/examples, their derived schema, "
        "and resulting score memory. v2 patches and persisted memory are replayed without generation. "
        "Retries and token costs can differ and remain separately recorded. This checks provenance, not semantic truth.",
    }
    write_json(output / "audit.json", result)
    print(
        json.dumps(
            {k: result[k] for k in ("status", "queries", "counters", "request_statuses")}, ensure_ascii=False
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.root.resolve(), args.output.resolve())
