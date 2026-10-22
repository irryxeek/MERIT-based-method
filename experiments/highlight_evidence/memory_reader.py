"""Frozen-memory R1/R2 reader experiment, with sequential preregistered gates."""

import argparse
import copy
import importlib.metadata
import json
import os
import shutil
import sys
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import numpy as np
from hd_adapter.client import locked

from merit_qvh.cli import validate_submission
from merit_qvh.core import digest, file_sha256, read_jsonl, write_json, write_jsonl

from .granularity import paired, verify
from .memory import number
from .memory_only import FALLBACK_SCORE, SCORE_SYSTEM, compact, costs, memory_view, request_payload
from .memory_only import score_request as original_request
from .pilot import read
from .robust_client import MemoryClient
from .robust_memory import ids_schema, object_schema

SOURCE_PROTOCOL = "aee598ce24ae75813982266aa1cef708ae7944b09a84ff9d3b85a251fd975eaa"
ARMS = ("R1", "R2")
REFERENCES = ("D0", "R0", "Memory_cue")
MAX_TOKENS = 4096
GPU_UUIDS = [uuid.strip() for uuid in os.environ.get("REPRO_GPU_UUIDS", "").split(",") if uuid.strip()]


def targets_for(count, arm):
    if arm not in ARMS or count < 1:
        raise ValueError("Unknown reader or empty timeline")
    size = count if arm == "R1" else 15
    return [list(range(i, min(i + size, count))) for i in range(0, count, size)]


class ClipContract:
    """IDs and citations are known metadata; their validity is not reasoning accuracy."""

    def __init__(self, memory, targets):
        if not targets or any(type(i) is not int for i in targets) or targets != sorted(set(targets)):
            raise ValueError("Targets must be unique ascending integer clip IDs")
        if any(i < 0 or i >= len(memory) for i in targets):
            raise ValueError("Target outside memory")
        self.targets = list(targets)
        self.allowed = {i: list(memory[i]["evidence_ids"]) for i in targets}
        if any(not refs or len(refs) != len(set(refs)) for refs in self.allowed.values()):
            raise ValueError("Each target must have valid structured citations")
        self.schema = object_schema(
            {
                "clips": {
                    "type": "array",
                    "prefixItems": [
                        object_schema(
                            {
                                "clip_id": {"type": "integer", "const": i},
                                "evidence_ids": ids_schema(self.allowed[i], 1),
                                "score": {"type": "number", "minimum": 0, "maximum": 4},
                            }
                        )
                        for i in targets
                    ],
                    "items": False,
                    "minItems": len(targets),
                    "maxItems": len(targets),
                }
            }
        )

    def problems(self, payload):
        reason = "Exactly one ordered clip_id, nonempty own evidence_ids and finite score in [0,4] per target"
        if not isinstance(payload, dict) or set(payload) != {"clips"}:
            return [{"path": [], "reason": reason}]
        records = payload["clips"]
        if not isinstance(records, list) or len(records) != len(self.targets):
            return [{"path": ["clips"], "reason": reason}]
        for item, i in zip(records, self.targets, strict=True):
            if not isinstance(item, dict) or set(item) != {"clip_id", "evidence_ids", "score"}:
                return [{"path": ["clips"], "reason": reason}]
            refs = item["evidence_ids"]
            if (
                type(item["clip_id"]) is not int
                or item["clip_id"] != i
                or not isinstance(refs, list)
                or not refs
                or any(not isinstance(ref, str) or ref not in self.allowed[i] for ref in refs)
                or len(refs) != len(set(refs))
                or not number(item["score"])
                or not 0 <= item["score"] <= 4
            ):
                return [{"path": ["clips"], "reason": reason}]
        return []

    def repair_spec(self, payload, issues):
        return object_schema({"/": self.schema}), (
            "Repair the complete clip score object under the single key '/'. Preserve the original task. "
            + compact({"validation_errors": issues, "previous_candidate": payload})
        )

    def apply_patch(self, original, patch, issues):
        if not isinstance(patch, dict) or set(patch) != {"/"}:
            raise ValueError("Score repair requires the single root path /")
        return copy.deepcopy(patch["/"])

    def finalize(self, payload, *, degrade=False):
        if self.problems(payload):
            raise ValueError("Unusable clip score records; no partial score degradation")
        return {
            "clips": copy.deepcopy(payload["clips"]),
            "scores": [float(item["score"]) for item in payload["clips"]],
            "downgraded_cues": [],
            "rejected_links": [],
        }


def score_request(row, memory, targets):
    content, _ = original_request(row, memory, "Memory_text")
    instruction, raw = content[0]["text"].split("\n", 1)
    body = json.loads(raw)
    if body["memory"] != memory:
        raise ValueError("Memory must equal the frozen R0 description view")
    old_output = "Return exactly {scores:[...]}, one number per clip_id in ascending order, no prose."
    if instruction.count(old_output) != 1:
        raise ValueError("Unexpected original scorer instruction")
    instruction = instruction.replace("Score EVERY 2-second clip", "Score EVERY target 2-second clip", 1)
    instruction = instruction.replace(
        old_output,
        "Return exactly {clips:[{clip_id,evidence_ids,score},...]}, one record per target_clip_id in "
        "ascending order, no prose. Cite one or more of that clip's supplied structured evidence_ids, "
        "including for low scores. All other memory records are context only; do not output their scores.",
    )
    body["target_clip_ids"] = list(targets)
    return [{"type": "text", "text": instruction + "\n" + compact(body)}], ClipContract(memory, targets)


def score_batch(client, row, memory, targets, arm, tokenizer, max_context):
    content, contract = score_request(row, memory, targets)
    messages = [{"role": "system", "content": SCORE_SYSTEM}, {"role": "user", "content": content}]
    tokens = len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
    if tokens + MAX_TOKENS > max_context:
        raise ValueError(f"Full memory exceeds context: {tokens}+{MAX_TOKENS}>{max_context}; no truncation")
    result = client.complete_memory(arm + "_score", content, MAX_TOKENS, contract, system=SCORE_SYSTEM)
    return {**result, "input_tokens_preflight": tokens, "input_memory_sha256": digest(memory)}


def merge_batches(memory, batches):
    records = [item for batch in batches for item in batch["clips"]]
    return ClipContract(memory, list(range(len(memory)))).finalize({"clips": records})


def shape(scores):
    counts = Counter(scores)
    suffix = 1
    for score in reversed(scores[:-1]):
        if score != scores[-1]:
            break
        suffix += 1
    return {
        "constant": len(counts) == 1,
        "constant_suffix_length": suffix,
        "top_ties": counts[max(scores)],
        "tied_pair_fraction": sum(n * (n - 1) for n in counts.values()) / (len(scores) * (len(scores) - 1)),
    }


def development_gate(metrics, records, arm):
    qids = sorted(records)
    delta = float(np.mean([metrics[q][arm]["mAP"] - metrics[q]["R0"]["mAP"] for q in qids]))
    hit_delta = float(np.mean([metrics[q][arm]["Hit1"] - metrics[q]["R0"]["Hit1"] for q in qids]))
    repeats = sum(
        shape(records[q][arm]["pred_saliency_scores"])["constant_suffix_length"] >= 45 for q in qids
    )
    checks = {
        "all20_complete": len(qids) == 20 and all(records[q][arm]["status"] == "ok" for q in qids),
        "map_gain_at_least_1pp": delta >= 1.0,
        "hit1_not_lower": hit_delta >= -1e-9,
        "constant_suffix45_at_most8": repeats <= 8,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "map_delta_vs_R0": delta,
        "hit1_delta_vs_R0": hit_delta,
        "suffix45_queries": repeats,
    }


def prepare(source, root, repo):
    if not GPU_UUIDS:
        raise ValueError("Set REPRO_GPU_UUIDS before preparing a GPU experiment")
    if root.exists():
        raise FileExistsError("Preserve prior runs; use a new directory")
    old, old_id = verify(source, labels=True)
    if old_id != SOURCE_PROTOCOL or read(source / "progress.json")["status"] != "complete":
        raise ValueError("Expected the frozen complete full20 memory-only run")
    rows = read_jsonl(source / "data/manifests/pilot.jsonl")
    if len(rows) != 20 or len({r["vid"] for r in rows}) != 20:
        raise ValueError("Expected all20 queries/videos")
    root.mkdir(parents=True)
    units = 0
    for row in rows:
        original = read(source / "queries" / f"{row['qid']}.json")
        chunks = [read(p) for p in sorted((source / "memories" / row["vid"]).glob("*.json"))]
        events = [e for chunk in chunks for e in chunk["events"]]
        if digest(events) != original["memory"]["events_sha256"] or any(
            c["status"] == "failed" for c in chunks
        ):
            raise ValueError("Source memory incomplete or changed")
        memory = memory_view(events, "Memory_text")
        expected_content, _ = original_request(row, events, "Memory_text")
        rid = original["Memory_text"]["request_id"]
        request = read(source / "requests" / rid[:2] / f"{rid}.json")
        if request["payload"]["messages"][1]["content"] != expected_content:
            raise ValueError("Input differs from actual R0 request")
        if original["Memory_text"]["pred_saliency_scores"] != request["result"]["scores"]:
            raise ValueError("R0 scores changed")
        safe_row = {k: row[k] for k in ("qid", "vid", "query", "duration")}
        write_json(
            root / "inputs" / f"{row['qid']}.json",
            {
                "row": safe_row,
                "memory": memory,
                "source_memory_status": {"partial_windows": original["memory"]["partial_windows"]},
            },
        )
        refs = {"D0": read(source / "references" / f"{row['qid']}.json")["D0"]}
        for target, source_arm in (("R0", "Memory_text"), ("Memory_cue", "Memory_cue")):
            refs[target] = {k: original[source_arm][k] for k in ("status", "pred_saliency_scores")}
        write_json(root / "references" / f"{row['qid']}.json", refs)
        score_request(safe_row, memory, list(range(len(memory))))
        units += len(memory)
    if units != 1499:
        raise ValueError("Frozen timeline changed")
    for name, expected in old["model_audit"]["files"].items():
        path = Path(old["vlm"]["weight_path"]) / name
        if path.stat().st_size != expected["size"] or file_sha256(path) != expected["sha256"]:
            raise ValueError("Frozen model changed: " + name)
    source_hashes = {}
    for folder in ("src", "experiments/hd_adapter", "experiments/highlight_evidence"):
        for path in sorted((repo / folder).rglob("*.py")):
            name = str(path.relative_to(repo))
            target = root / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            source_hashes[name] = file_sha256(path)
    for folder in ("evaluator", "data"):
        shutil.copytree(source / folder, root / folder, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copyfile(repo / "docs/highlight_memory_reader_plan_2026-10-17.md", root / "plan.md")
    witness = {
        str(p.relative_to(source)): file_sha256(p)
        for p in source.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.suffix not in {".lock", ".pyc"}
    }
    server = {**copy.deepcopy(old["server"]), "port": 8041, "gpu_uuids": GPU_UUIDS}
    settings = {
        "experiment": "full_memory_reader_R1_R2_v1",
        "source_run": str(source),
        "source_protocol_id": old_id,
        "source_artifact_sha256": witness,
        "vlm": old["vlm"],
        "model_audit": old["model_audit"],
        "server": server,
        "endpoint": "http://127.0.0.1:8041/v1",
        "gpu_indices": [4, 5],
        "workers": 4,
        "source_sha256": source_hashes,
        "input_sha256": {
            str(p.relative_to(root)): file_sha256(p)
            for folder in ("inputs", "references", "data", "evaluator")
            for p in (root / folder).rglob("*")
            if p.is_file()
        },
        "plan_sha256": file_sha256(root / "plan.md"),
        "score_max_tokens": MAX_TOKENS,
        "planned": {
            "queries": 20,
            "memory_units_reused": units,
            "new_memory_calls": 0,
            "new_frames": 0,
            "R1_logical_requests": 20,
            "R2_conditional_logical_requests": 100,
            "max_actual_attempts": 480,
        },
        "gates": {
            "map_gain_vs_R0_pp": 1.0,
            "hit1_delta_min_pp": 0.0,
            "max_suffix45_queries": 8,
            "required_scoring_successes": 20,
            "first4_engineering": "all four must complete",
        },
        "failure": "A failed batch fails its whole query with flat2.0; no D0/R0 blending; preserve attempts",
        "scope": "Only full frozen text memory, original query and target IDs; no tools/images/new memory/labels",
        "output_metadata": "Known clip IDs constrained by prefixItems; citations use own structured frame IDs",
        "max_wall_seconds": 3600,
        "packages": {n: importlib.metadata.version(n) for n in ("vllm", "torch", "transformers", "numpy")},
    }
    write_json(root / "protocol.json", {"protocol_id": digest(settings), "settings": settings})
    write_json(root / "prepared.json", {"at": time.time(), "weights_verified": True, **settings["planned"]})
    return {"root": str(root), "protocol_id": digest(settings), **settings["planned"]}


def run_query(root, c, protocol_id, row, tokenizer, arm):
    qid = row["qid"]
    destination = root / "arms" / arm / f"{qid}.json"
    with locked(root / "locks" / f"{arm}_{qid}.lock"):
        if destination.exists():
            return read(destination)
        data = read(root / "inputs" / f"{qid}.json")
        memory = data["memory"]
        started, batches = time.time(), []
        for index, targets in enumerate(targets_for(len(memory), arm)):
            path = root / "batches" / arm / str(qid) / f"{index:03}.json"
            if path.exists():
                batch = read(path)
            else:
                client = MemoryClient(c["endpoint"], c["vlm"], root / "requests", protocol_id)
                batch = {"index": index, "target_clip_ids": targets}
                try:
                    scored = score_batch(
                        client, data["row"], memory, targets, arm, tokenizer, c["server"]["max_model_len"]
                    )
                    batch.update(status="ok", **scored)
                except (RuntimeError, ValueError, TypeError, KeyError, OSError) as error:
                    batch.update(status="failed", error=str(error))
                batch["requests"] = client.events
                write_json(path, batch)
            batches.append(batch)
            if batch["status"] == "failed":
                break
        out = {
            "qid": qid,
            "vid": row["vid"],
            "protocol_id": protocol_id,
            "arm": arm,
            "status": "failed",
            "pred_saliency_scores": [FALLBACK_SCORE] * len(memory),
            "fallback": "flat2.0",
            "batches": batches,
            "source_memory_status": data["source_memory_status"],
            "input_memory_sha256": digest(memory),
            "wall_seconds": time.time() - started,
        }
        if all(b["status"] == "ok" for b in batches):
            merged = merge_batches(memory, batches)
            out.update(
                status="ok", pred_saliency_scores=merged["scores"], clips=merged["clips"], fallback=None
            )
        write_json(destination, out)
        return out


def evaluate(root, arms):
    _, protocol_id = verify(root, labels=True)
    truth = read_jsonl(root / "data/ground_truth/pilot.jsonl")
    records = {
        t["qid"]: {
            "vid": t["vid"],
            **read(root / "references" / f"{t['qid']}.json"),
            **{a: read(root / "arms" / a / f"{t['qid']}.json") for a in arms},
        }
        for t in truth
    }
    groups = (*REFERENCES, *arms)
    sys.path.insert(0, str(root / "evaluator"))
    from standalone_eval.eval import compute_ap_from_tuple, eval_submission, mk_gt_scores

    metrics, details = {}, []
    for row in truth:
        qid = row["qid"]
        labels = (mk_gt_scores(row) >= 4).astype(float)
        metrics[qid] = {}
        item = {
            "qid": qid,
            "vid": row["vid"],
            "query": row["query"],
            "metrics": metrics[qid],
            "shape": {},
            "status": {},
        }
        for arm in groups:
            scores = np.asarray(records[qid][arm]["pred_saliency_scores"])
            if len(scores) != len(labels) or not np.isfinite(scores).all():
                raise ValueError("Invalid score coverage")
            metrics[qid][arm] = {
                "mAP": 100
                * float(np.mean([compute_ap_from_tuple((0, i, labels[:, i], scores))[2] for i in range(3)])),
                "Hit1": float(100 * labels[int(scores.argmax())].max()),
            }
            item["shape"][arm] = shape(scores.tolist())
            item["status"][arm] = records[qid][arm]["status"]
        details.append(item)
    common = [q for q in records if all(records[q][a]["status"] in {"ok", "partial"} for a in groups)]
    report = {
        "protocol_id": protocol_id,
        "queries": len(truth),
        "arms": list(arms),
        "groups": {},
        "cost": {k: v for k, v in costs(root).items() if k.removesuffix("_score") in arms},
        "common_success_qids": common,
    }
    target = root / "evaluations" / ("after_" + arms[-1])
    for arm in groups:
        predictions = [
            {
                "qid": t["qid"],
                "vid": t["vid"],
                "pred_saliency_scores": records[t["qid"]][arm]["pred_saliency_scores"],
            }
            for t in truth
        ]
        validate_submission(predictions, truth)
        official = eval_submission(predictions, truth, verbose=False, match_number=True)
        for key, name in (("HL-mAP", "mAP"), ("HL-Hit1", "Hit1")):
            if official["HL-min-VeryGood"][key] != round(
                float(np.mean([m[arm][name] for m in metrics.values()])), 2
            ):
                raise ValueError("Per-query metrics disagree with official evaluator")
        report["groups"][arm] = {
            "metrics": official,
            "status_counts": dict(Counter(r[arm]["status"] for r in records.values())),
            "constant_queries": sum(d["shape"][arm]["constant"] for d in details),
            "suffix45_queries": sum(d["shape"][arm]["constant_suffix_length"] >= 45 for d in details),
            "median_suffix": float(np.median([d["shape"][arm]["constant_suffix_length"] for d in details])),
            "median_top_ties": float(np.median([d["shape"][arm]["top_ties"] for d in details])),
            "common_success_metrics": eval_submission(
                [p for p in predictions if p["qid"] in common],
                [t for t in truth if t["qid"] in common],
                verbose=False,
                match_number=True,
            )
            if common
            else None,
        }
        pred_path = target / f"{arm}_predictions.jsonl"
        if pred_path.exists() and read_jsonl(pred_path) != predictions:
            raise ValueError("Completed predictions changed")
        if not pred_path.exists():
            write_jsonl(pred_path, predictions)
    contrasts = [(a, ref) for a in arms for ref in ("R0", "D0")]
    if "R2" in arms:
        contrasts.append(("R2", "R1"))
    report["paired"] = {
        a + "_minus_" + b: {
            "full20": paired(metrics, records, a, b, sorted(records)),
            "common_success": paired(metrics, records, a, b, common),
        }
        for a, b in contrasts
    }
    report["development_gate"] = development_gate(metrics, records, arms[-1])
    report["limitations"] = [
        "Repeatedly explored training20; thresholds are development heuristics, not independent confirmation",
        "R1 changes output structure and max_tokens; R2 additionally limits target outputs while keeping full memory",
        "Known IDs/citations are grammar-constrained; valid provenance is not evidence of semantic reasoning",
        "Descriptions are unchanged and may contain cue levels or inconsistent inline frame references",
        "Cached R0/D0 differ in serving session; physical GPUs changed from6/7 to4/5, both TP2",
        "No new memory construction; full-method cost must include historical memory building",
    ]
    write_json(target / "per_query.json", details)
    write_json(target / "report.json", report)
    return report


def audit(root, arms):
    c, protocol_id = verify(root, labels=True)
    if file_sha256(root / "plan.md") != c["plan_sha256"]:
        raise ValueError("Frozen plan changed")
    for name, expected in c["source_artifact_sha256"].items():
        if file_sha256(Path(c["source_run"]) / name) != expected:
            raise ValueError("Previous experiment changed: " + name)
    used, counts = set(), Counter()
    for row in read_jsonl(root / "data/manifests/pilot.jsonl"):
        data = read(root / "inputs" / f"{row['qid']}.json")
        memory = data["memory"]
        for arm in arms:
            saved = read(root / "arms" / arm / f"{row['qid']}.json")
            expected_targets = targets_for(len(memory), arm)
            batches = saved["batches"]
            for index, batch in enumerate(batches):
                targets = expected_targets[index]
                if batch["target_clip_ids"] != targets:
                    raise ValueError("Target partition changed")
                content, contract = score_request(data["row"], memory, targets)
                payload = request_payload(c["vlm"], content, MAX_TOKENS, contract, SCORE_SYSTEM)
                stage = arm + "_score"
                rid = digest({"protocol": protocol_id, "stage": stage, "payload": payload})
                if batch["requests"]:
                    if [e["request_id"] for e in batch["requests"]] != [rid]:
                        raise ValueError("Request ID mismatch")
                    request = read(root / "requests" / rid[:2] / f"{rid}.json")
                    used.add(rid)
                    replayed = replay_request(request, payload, contract)
                    if batch["status"] == "ok":
                        if replayed is None or any(batch[k] != v for k, v in replayed.items()):
                            raise ValueError("Raw response differs from saved batch")
                    elif request["status"] != "failed":
                        raise ValueError("Failed batch has valid request")
                    counts["logical_requests_verified"] += 1
                    counts["actual_attempts_verified"] += len(request["attempts"])
                elif batch["status"] != "failed":
                    raise ValueError("Completed batch lacks request")
            if saved["status"] == "ok":
                if len(batches) != len(expected_targets):
                    raise ValueError("Incomplete batch partition")
                merged = merge_batches(memory, batches)
                if saved["clips"] != merged["clips"] or saved["pred_saliency_scores"] != merged["scores"]:
                    raise ValueError("Merged scores changed")
                counts["complete_query_arms"] += 1
                counts["validated_clip_records"] += len(memory)
            elif saved["pred_saliency_scores"] != [FALLBACK_SCORE] * len(memory):
                raise ValueError("Wrong failure policy")
    if used != {p.stem for p in (root / "requests").glob("*/*.json")}:
        raise ValueError("Missing or orphan request")
    return {
        "status": "passed",
        "protocol_id": protocol_id,
        **counts,
        "source_files_unchanged": len(c["source_artifact_sha256"]),
        "no_images_or_tools": True,
        "no_D0_or_label_inputs": True,
        "full_frozen_memory_per_request": True,
    }


def replay_request(request, payload, contract):
    if request["payload"] != payload or request["status"] not in {"ok", "failed"}:
        raise ValueError("Request payload/status mismatch")
    candidate, issues, slots, valid = None, [], set(), None
    for attempt in request["attempts"]:
        slot = (attempt["format_attempt"], attempt["transport_attempt"])
        if slot in slots or slot[0] not in {0, 1} or slot[1] not in {0, 1} or valid is not None:
            raise ValueError("Retry budget or sequence mismatch")
        slots.add(slot)
        expected = copy.deepcopy(payload)
        if candidate is not None:
            schema, prompt = contract.repair_spec(candidate, issues)
            expected["response_format"] = MemoryClient.response_format(schema)
            expected["messages"].append({"role": "user", "content": prompt})
        elif slot[0]:
            expected["messages"].append(
                {"role": "user", "content": "Previous generation was incomplete. Return compact JSON."}
            )
        if attempt["payload"] != expected:
            raise ValueError("Retry payload drift")
        if "response" not in attempt:
            continue
        try:
            choice = attempt["response"]["choices"][0]
            if choice["finish_reason"] != "stop":
                raise ValueError("Incomplete generation")
            decoded = json.loads(choice["message"]["content"])
            candidate = contract.apply_patch(candidate, decoded, issues) if candidate is not None else decoded
            issues = contract.problems(candidate)
            valid = contract.finalize(candidate)
        except (ValueError, TypeError, KeyError, IndexError):
            valid = None
    if request["status"] == "ok":
        if valid is None or any(request["result"][k] != v for k, v in valid.items()):
            raise ValueError("Saved result does not match raw model response")
        if json.loads(request["text"]) != candidate:
            raise ValueError("Saved candidate differs from replay")
        return valid
    if valid is not None:
        raise ValueError("Valid response stored as failure")
    return None


def run_stage(root, c, protocol_id, rows, tokenizer, arm, started):
    completed = []
    for segment in (rows[:4], rows[4:]):
        with ThreadPoolExecutor(max_workers=c["workers"]) as pool:
            futures = {
                pool.submit(run_query, root, c, protocol_id, row, tokenizer, arm): row for row in segment
            }
            while futures:
                done, _ = wait(futures, timeout=10, return_when=FIRST_COMPLETED)
                for future in done:
                    completed.append(future.result())
                    futures.pop(future)
                progress = {
                    "status": "running",
                    "stage": arm,
                    "completed": len(completed),
                    "total": 20,
                    "active_qids": [r["qid"] for r in futures.values()],
                    "updated_at": time.time(),
                    "elapsed_seconds": time.time() - started,
                    "status_counts": dict(Counter(r["status"] for r in completed)),
                }
                write_json(root / "progress.json", progress)
                if done:
                    print(json.dumps(progress), flush=True)
                if time.time() - started > c["max_wall_seconds"]:
                    raise RuntimeError("Wall budget exhausted")
        if len(completed) == 4 and any(r["status"] != "ok" for r in completed):
            write_json(
                root / "circuit_breaker.json",
                {
                    "at": time.time(),
                    "stage": arm,
                    "reason": "At least one terminal engineering failure in first4",
                },
            )
            raise RuntimeError("First4 engineering gate failed; protocol incomplete")


def run(root):
    from transformers import AutoTokenizer

    c, protocol_id = verify(root)
    if (root / "circuit_breaker.json").exists():
        raise RuntimeError("Preserved engineering gate failure")
    if file_sha256(root / "plan.md") != c["plan_sha256"]:
        raise ValueError("Frozen plan changed")
    served = MemoryClient(c["endpoint"], c["vlm"], root / "requests", protocol_id).check_model()
    tokenizer = AutoTokenizer.from_pretrained(c["vlm"]["weight_path"], local_files_only=True)
    rows = read_jsonl(root / "data/manifests/pilot.jsonl")
    started = time.time()
    session_path = root / "sessions" / f"{time.time_ns()}.json"
    session = {"started_at": started, "model": served, "workers": c["workers"]}
    write_json(session_path, session)
    completed_arms, report = [], None
    for arm in ARMS:
        completed_arms.append(arm)
        decision_path = root / "decisions" / f"{arm}.json"
        if decision_path.exists():
            report = read(root / "evaluations" / ("after_" + arm) / "report.json")
            decision = read(decision_path)
            if decision != report["development_gate"]:
                raise ValueError("Saved decision changed")
        else:
            run_stage(root, c, protocol_id, rows, tokenizer, arm, started)
            checked = audit(root, completed_arms)
            write_json(root / "audits" / f"after_{arm}.json", checked)
            report = evaluate(root, completed_arms)
            decision = report["development_gate"]
            write_json(decision_path, decision)
        print(json.dumps({"stage": arm, "decision": decision}), flush=True)
        if decision["passed"]:
            break
    write_json(root / "report.json", report)
    session.update(status="complete", finished_at=time.time())
    write_json(session_path, session)
    write_json(
        root / "progress.json",
        {
            "status": "complete",
            "completed": 20,
            "total": 20,
            "completed_arms": completed_arms,
            "elapsed_seconds": time.time() - started,
            "selected_reader": completed_arms[-1] if report["development_gate"]["passed"] else None,
            "updated_at": time.time(),
        },
    )
    return {a: report["groups"][a]["metrics"]["HL-min-VeryGood"] for a in report["groups"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "audit"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == "prepare":
        result = prepare(args.source.resolve(), root, args.repo.resolve())
    elif args.action == "audit":
        result = audit(root, read(root / "progress.json")["completed_arms"])
    else:
        with locked(root / "runner.lock", blocking=False):
            try:
                result = run(root)
            except Exception as error:
                old = read(root / "progress.json") if (root / "progress.json").exists() else {}
                write_json(
                    root / "progress.json",
                    {**old, "status": "stopped_with_error", "error": str(error), "updated_at": time.time()},
                )
                raise
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
