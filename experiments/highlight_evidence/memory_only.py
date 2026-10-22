"""Full-video 2s memory followed by text-only HD scoring on the fixed train20.

No D0 input, candidate selection, tool action, source revisit, graph inference,
or pairwise comparison. The shared visual memory is built without the query.
"""

import argparse
import copy
import importlib.metadata
import json
import math
import shutil
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import numpy as np
from hd_adapter.client import locked

from merit_qvh.cli import validate_submission
from merit_qvh.client import redact_images
from merit_qvh.core import digest, file_sha256, read_jsonl, write_json, write_jsonl

from .granularity import namespace_result, paired, units_for, verify
from .inference import visual_content
from .memory import SYSTEM, number
from .pilot import load_frames, read
from .robust_client import MemoryClient
from .robust_memory import MemoryContract, object_schema

ARMS = ("Memory_text", "Memory_cue")
GROUPS = ("D0", *ARMS)
SCORE_SYSTEM = "Judge query-specific video highlights using only the supplied textual observations."
SCORE_MAX_TOKENS = 2048
FALLBACK_SCORE = 2.0


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


class FullMemoryContract(MemoryContract):
    """Reuse validated observation fields, but do not generate optional graph edges."""

    def __init__(self, *args):
        super().__init__(*args)
        self.schema["properties"]["links"]["maxItems"] = 0

    def problems(self, payload):
        issues = super().problems(payload)
        if isinstance(payload, dict) and payload.get("links") != []:
            issues.append({"path": ["links"], "reason": "This experiment requires links:[]"})
        return issues


def memory_request(frames, values, duration, chunk, context):
    units = units_for(chunk, "F2")
    contract = FullMemoryContract(units, frames, values, duration, context)
    # Previous descriptions give concrete context without recursively copying cues.
    previous = [{k: e[k] for k in ("id", "start", "end", "description", "evidence_ids")} for e in context]
    prompt = (
        "Build query-independent visual evidence memory for the CURRENT window. Return one observation "
        "for EACH observation_unit, keyed by its supplied ID. The controller supplies IDs and times. "
        "Each 2-second unit has one sampled picture: it is observation support, not a true action boundary. "
        "Describe only facts visible in that unit's own picture; other pictures provide context only. "
        "A still picture cannot prove motion between samples. Record subject, visible action or state, "
        "and visible result when supported. Description <=240 characters, each cue <=160 characters. "
        "Each description and observed cue must cite the unit's own evidence_ids. "
        "Include all four cue dimensions. Status observed/unknown/not_applicable; observed level is "
        "0 absent,1 mild,2 moderate,3 strong; otherwise level=null. Unknown is not observed absence. "
        "Observed novelty/importance MUST cite another supplied observation and explain a concrete "
        "contrast or event-goal/outcome. Without that evidence use unknown, null, and empty citations. "
        "Previous descriptions are imperfect model observations, not ground truth. Do not copy "
        "references to events that are not supplied. Return links:[]; no graph construction. "
        "Do not generate query relevance or highlight scores. Return compact JSON matching the schema.\n"
        + compact(
            {
                "video_duration": duration,
                "current_interval": [chunk["start"], chunk["end"]],
                "value_definitions": values,
                "prior_observations": previous,
                "observation_units": units,
            }
        )
    )
    return visual_content(prompt, frames), contract, units


def extract_memory(client, frames, values, duration, chunk, context):
    content, contract, units = memory_request(frames, values, duration, chunk, context)
    result = client.complete_memory("F2_full_memory", content, 8192, contract, system=SYSTEM)
    return namespace_result(result, chunk, units)


def memory_view(events, arm):
    if arm not in ARMS:
        raise ValueError("Unknown memory-only condition")
    result = []
    for clip_id, event in enumerate(events):
        item = {k: copy.deepcopy(event[k]) for k in ("id", "start", "end", "description", "evidence_ids")}
        item["clip_id"] = clip_id
        if arm == "Memory_cue":
            item["values"] = {
                v["dimension"]: {k: copy.deepcopy(x) for k, x in v.items() if k != "dimension"}
                for v in event["values"]
            }
        result.append(item)
    return result


class ScoreContract:
    """Compact full-vector output; malformed scores cannot be downgraded into facts."""

    def __init__(self, count):
        self.count = count
        self.schema = object_schema(
            {
                "scores": {
                    "type": "array",
                    "minItems": count,
                    "maxItems": count,
                    "items": {"type": "number", "minimum": 0, "maximum": 4},
                }
            }
        )

    def problems(self, payload):
        valid = (
            isinstance(payload, dict)
            and set(payload) == {"scores"}
            and isinstance(payload["scores"], list)
            and len(payload["scores"]) == self.count
            and all(number(s) and 0 <= s <= 4 for s in payload["scores"])
        )
        return (
            [] if valid else [{"path": [], "reason": f"Exactly {self.count} finite scores in [0,4] required"}]
        )

    def repair_spec(self, payload, issues):
        return object_schema({"/": self.schema}), (
            "Repair the complete score object under the single key '/'. Preserve the original task. "
            + compact({"validation_errors": issues, "previous_candidate": payload})
        )

    def apply_patch(self, original, patch, issues):
        if not isinstance(patch, dict) or set(patch) != {"/"}:
            raise ValueError("Score repair requires the single root path /")
        return copy.deepcopy(patch["/"])

    def finalize(self, payload, *, degrade=False):
        if self.problems(payload):
            raise ValueError("Unusable full score vector")
        # These two empty fields satisfy the existing replayable client's contract.
        return {"scores": [float(s) for s in payload["scores"]], "downgraded_cues": [], "rejected_links": []}


def score_request(row, events, arm):
    memory = memory_view(events, arm)
    count = math.ceil(row["duration"] / 2)
    if len(memory) != count or any(e["start"] != 2 * i for i, e in enumerate(memory)):
        raise ValueError("Scoring requires complete ordered full-video memory")
    prompt = (
        "Score EVERY 2-second clip of this video for relevance and highlight suitability for the ORIGINAL QUERY. "
        "Only the supplied textual memory is available; do not assume access to pictures, audio, or tools. "
        "Use the same 0-4 scale throughout the whole video: 0 very bad,1 bad,2 fair,3 good,4 very good. "
        "Decimals are allowed. Higher means more query-relevant and salient. Do not force a winner, "
        "normalize scores, or add tiny differences merely to break ties. Compare supplied observations "
        "across the full timeline when useful. Descriptions and cues are model observations, not labels. "
        "Unknown/inapplicable cues do not imply low saliency. Never sum cue levels. "
        "Judge the query first: unrelated excitement is not a query highlight. Context may explain an "
        "event without making every context clip a highlight. Do not invent unobserved actions. "
        "Return exactly {scores:[...]}, one number per clip_id in ascending order, no prose.\n"
        + compact({"original_query": row["query"], "duration": row["duration"], "memory": memory})
    )
    return [{"type": "text", "text": prompt}], ScoreContract(count)


def request_payload(config, content, max_tokens, contract, system):
    return {
        "model": config["model"],
        "temperature": config["temperature"],
        "seed": 42,
        "max_tokens": max_tokens,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
        "response_format": MemoryClient.response_format(contract.schema),
    }


def score_memory(client, row, events, arm, tokenizer, max_context):
    content, contract = score_request(row, events, arm)
    messages = [{"role": "system", "content": SCORE_SYSTEM}, {"role": "user", "content": content}]
    tokens = len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
    if tokens + SCORE_MAX_TOKENS > max_context:
        raise ValueError(
            f"Full memory exceeds context: {tokens}+{SCORE_MAX_TOKENS}>{max_context}; no truncation"
        )
    result = client.complete_memory(arm + "_score", content, SCORE_MAX_TOKENS, contract, system=SCORE_SYSTEM)
    return {
        **result,
        "input_tokens_preflight": tokens,
        "input_memory_sha256": digest(memory_view(events, arm)),
    }


def prepare(source, root, repo):
    if root.exists():
        raise FileExistsError("Use a new directory; preserve previous runs")
    old, old_id = verify(source, labels=True)
    if read(source / "progress.json")["status"] != "complete":
        raise ValueError("Completed source run required")
    pilot = Path(old["source_run"])
    rows = read_jsonl(source / "data/manifests/pilot.jsonl")
    if len(rows) != 20 or len({r["vid"] for r in rows}) != 20:
        raise ValueError("Expected original20 queries on20 distinct videos")
    root.mkdir(parents=True)
    counts = Counter()
    source_witness = {}
    for row in rows:
        origin = pilot / "baseline" / f"{row['qid']}.json"
        source_witness[str(origin)] = file_sha256(origin)
        saved = read(origin)
        count = math.ceil(row["duration"] / 2)
        frames = saved["frames"]
        if len(frames) != count or [f["clip_id"] for f in frames] != list(range(count)):
            raise ValueError("Incomplete independently defined frame grid")
        if any(
            f["image_id"] != f"f{i}" or not 2 * i <= f["pts"] < min(2 * i + 2, row["duration"])
            for i, f in enumerate(frames)
        ):
            raise ValueError("Cached pictures do not match the fixed 2s grid")
        load_frames(frames)  # Verify every actual JPEG, including prior no-candidate/failure queries.
        if file_sha256(row["video_path"]) != saved["media"]["media_sha256"]:
            raise ValueError("Source video changed")
        windows = [
            {
                "window_id": start // 5,
                "start": 2 * start,
                "end": min(2 * (start + 5), row["duration"]),
                "bin_ids": list(range(start, min(start + 5, count))),
            }
            for start in range(0, count, 5)
        ]
        # D0 is kept outside inference inputs and only read by evaluation.
        write_json(root / "inputs" / f"{row['qid']}.json", {"row": row, "frames": frames, "windows": windows})
        baseline = {k: saved["D0"][k] for k in ("status", "pred_saliency_scores")}
        if len(baseline["pred_saliency_scores"]) != count:
            raise ValueError("Baseline grid mismatch")
        write_json(root / "references" / f"{row['qid']}.json", {"D0": baseline})
        counts.update(memory_calls=len(windows), memory_units=count, scoring_calls=len(ARMS))
    for name, expected in old["model_audit"]["files"].items():
        path = Path(old["vlm"]["weight_path"]) / name
        if path.stat().st_size != expected["size"] or file_sha256(path) != expected["sha256"]:
            raise ValueError("Frozen model artifact changed: " + name)
    sources = {}
    for folder in ("src", "experiments/hd_adapter", "experiments/highlight_evidence"):
        for path in sorted((repo / folder).rglob("*.py")):
            name = str(path.relative_to(repo))
            target = root / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            sources[name] = file_sha256(path)
    for name in ("evaluator", "data"):
        shutil.copytree(source / name, root / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    source_witness.update(
        {
            str(p): file_sha256(p)
            for p in source.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and p.suffix not in {".lock", ".pyc"}
        }
    )
    c = {
        "experiment": "full_video_F2_memory_only_v1",
        "source_run": str(source),
        "source_protocol_id": old_id,
        "source_artifact_sha256": source_witness,
        "vlm": old["vlm"],
        "model_audit": old["model_audit"],
        "server": old["server"],
        "endpoint": old["endpoint"],
        "values": old["values"],
        "workers": 4,
        "source_sha256": sources,
        "input_sha256": {
            str(p.relative_to(root)): file_sha256(p)
            for folder in ("inputs", "references", "data", "evaluator")
            for p in (root / folder).rglob("*")
            if p.is_file()
        },
        "planned": dict(counts),
        "arms": list(ARMS),
        "memory_max_tokens": 8192,
        "score_max_tokens": SCORE_MAX_TOKENS,
        "context": "Previous two window slots' successfully generated descriptions; no previous cue fields or graphs",
        "memory": "Fresh query-independent F2 for all windows; same cached midpoint JPEG pixels as D0",
        "scoring": "One full-video text-only request per condition/query; no truncation, tools, images, retrieval or D0",
        "failure": "Record failed windows and continue building; any missing window disables both scorers. "
        "A failed scorer disables only its arm. Failed predictions use flat2.0, never D0. "
        "Keep all20 in primary metrics and additionally report common-success subset.",
        "max_wall_seconds": 5400,
        "packages": {n: importlib.metadata.version(n) for n in ("vllm", "torch", "transformers", "numpy")},
    }
    write_json(root / "protocol.json", {"protocol_id": digest(c), "settings": c})
    write_json(root / "prepared.json", {"at": time.time(), "queries": 20, "weights_verified": True, **counts})
    return {"root": str(root), "protocol_id": digest(c), **counts}


def run_query(root, c, protocol_id, row, tokenizer):
    destination = root / "queries" / f"{row['qid']}.json"
    with locked(root / "locks" / f"{row['qid']}.lock"):
        if destination.exists():
            return read(destination)
        data = read(root / "inputs" / f"{row['qid']}.json")
        started = time.time()
        result = {**row, "protocol_id": protocol_id, "errors": []}
        chunks = []
        memory_client = MemoryClient(c["endpoint"], c["vlm"], root / "requests", protocol_id)
        frames = load_frames(data["frames"])
        for window in data["windows"]:
            path = root / "memories" / row["vid"] / f"{window['window_id']:03}.json"
            if path.exists():
                chunk = read(path)
            else:
                context = [e for prev in chunks[-2:] for e in prev.get("events", [])]
                try:
                    chunk = extract_memory(
                        memory_client,
                        [frames[i] for i in window["bin_ids"]],
                        c["values"],
                        row["duration"],
                        window,
                        context,
                    )
                    chunk["status"] = "partial" if chunk["quality"] != "validated" else "ok"
                except Exception as error:
                    chunk = {**window, "status": "failed", "events": [], "error": str(error)}
                chunk["context_event_ids"] = [e["id"] for e in context]
                write_json(path, chunk)
            chunks.append(chunk)
            write_json(
                root / "query_progress" / f"{row['qid']}.json",
                {
                    "qid": row["qid"],
                    "stage": "memory",
                    "completed_windows": len(chunks),
                    "total_windows": len(data["windows"]),
                    "updated_at": time.time(),
                },
            )
        events = [e for chunk in chunks for e in chunk["events"]]
        failures = [c0 for c0 in chunks if c0["status"] == "failed"]
        result["memory"] = {
            "windows": len(chunks),
            "units": len(events),
            "failed_windows": [x["window_id"] for x in failures],
            "partial_windows": [x["window_id"] for x in chunks if x["status"] == "partial"],
            "requests": memory_client.events,
            "events_sha256": digest(events),
        }
        result["errors"].extend(
            {"stage": "memory", "window_id": x["window_id"], "error": x["error"]} for x in failures
        )
        order = sorted(ARMS, key=lambda a: digest([42, row["qid"], a]))
        result["arm_execution_order"] = order
        for arm in order:
            path = root / "arms" / arm / f"{row['qid']}.json"
            if path.exists():
                result[arm] = read(path)
                continue
            out = {
                "status": "failed",
                "pred_saliency_scores": [FALLBACK_SCORE] * len(frames),
                "requests": [],
                "fallback": "flat2.0",
                "errors": [],
            }
            if failures:
                out["errors"].append("Incomplete full-video memory; no scorer request")
            else:
                client = MemoryClient(c["endpoint"], c["vlm"], root / "requests", protocol_id)
                try:
                    scored = score_memory(client, row, events, arm, tokenizer, c["server"]["max_model_len"])
                    out.update(
                        status="partial" if result["memory"]["partial_windows"] else "ok",
                        pred_saliency_scores=scored["scores"],
                        fallback=None,
                        request_id=scored["request_id"],
                        input_tokens_preflight=scored["input_tokens_preflight"],
                        input_memory_sha256=scored["input_memory_sha256"],
                    )
                except Exception as error:
                    out["errors"].append(str(error))
                out["requests"] = client.events
            write_json(path, out)
            result[arm] = out
        result["wall_seconds"] = time.time() - started
        write_json(destination, result)
        write_json(
            root / "query_progress" / f"{row['qid']}.json",
            {"qid": row["qid"], "stage": "complete", "updated_at": time.time()},
        )
        return result


def costs(root):
    stages = defaultdict(Counter)
    for path in (root / "requests").glob("*/*.json"):
        record = read(path)
        out = stages[record["stage"]]
        out["logical_requests"] += 1
        out[record["status"] + "_requests"] += 1
        for attempt in record["attempts"]:
            out["actual_attempts"] += 1
            out["seconds_including_queue"] += attempt.get("seconds", 0)
            out["unknown_interrupted_attempts"] += attempt["status"] in {"in_flight", "interrupted_unknown"}
            for key, value in attempt.get("response", {}).get("usage", {}).items():
                if key in {"prompt_tokens", "completion_tokens", "total_tokens"}:
                    out[key] += value
            for message in attempt["payload"]["messages"]:
                if isinstance(message["content"], list):
                    out["image_presentations"] += sum(x["type"] == "image_url" for x in message["content"])
    return {k: dict(v) for k, v in stages.items()}


def evaluate(root):
    c, protocol_id = verify(root, labels=True)
    truth = read_jsonl(root / "data/ground_truth/pilot.jsonl")
    records = {t["qid"]: read(root / "queries" / f"{t['qid']}.json") for t in truth}
    for q, r in records.items():
        r["D0"] = read(root / "references" / f"{q}.json")["D0"]
    sys.path.insert(0, str(root / "evaluator"))
    from standalone_eval.eval import compute_ap_from_tuple, eval_submission, mk_gt_scores

    metrics, details = {}, []
    common = [q for q, r in records.items() if all(r[a]["status"] in {"ok", "partial"} for a in GROUPS)]
    for t in truth:
        q = t["qid"]
        r = records[q]
        labels = (mk_gt_scores(t) >= 4).astype(float)
        metrics[q] = {}
        detail = {
            "qid": q,
            "vid": r["vid"],
            "query": r["query"],
            "metrics": metrics[q],
            "top_ties": {},
            "status": {},
        }
        for arm in GROUPS:
            scores = np.asarray(r[arm]["pred_saliency_scores"])
            if len(scores) != len(labels) or not np.isfinite(scores).all():
                raise ValueError("Invalid score coverage")
            metrics[q][arm] = {
                "mAP": 100
                * float(np.mean([compute_ap_from_tuple((0, i, labels[:, i], scores))[2] for i in range(3)])),
                "Hit1": float(100 * labels[int(scores.argmax())].max()),
            }
            detail["top_ties"][arm] = int((scores == scores.max()).sum())
            detail["status"][arm] = r[arm]["status"]
        details.append(detail)
    report = {
        "protocol_id": protocol_id,
        "queries": len(truth),
        "groups": {},
        "cost": costs(root),
        "common_success_qids": common,
    }
    for arm in GROUPS:
        predictions = [
            {
                "qid": t["qid"],
                "vid": records[t["qid"]]["vid"],
                "pred_saliency_scores": records[t["qid"]][arm]["pred_saliency_scores"],
            }
            for t in truth
        ]
        validate_submission(predictions, truth)
        path = root / f"{arm}_predictions.jsonl"
        if path.exists() and read_jsonl(path) != predictions:
            raise ValueError("Saved predictions changed")
        if not path.exists():
            write_jsonl(path, predictions)
        official = eval_submission(predictions, truth, verbose=False, match_number=True)
        for key, name in (("HL-mAP", "mAP"), ("HL-Hit1", "Hit1")):
            if official["HL-min-VeryGood"][key] != round(
                float(np.mean([m[arm][name] for m in metrics.values()])), 2
            ):
                raise ValueError("Per-query metrics disagree with official evaluator")
        report["groups"][arm] = {
            "metrics": official,
            "status_counts": dict(Counter(r[arm]["status"] for r in records.values())),
            "top_tied_queries": sum(d["top_ties"][arm] > 1 for d in details),
            "common_success_metrics": eval_submission(
                [p for p in predictions if p["qid"] in common],
                [t for t in truth if t["qid"] in common],
                verbose=False,
                match_number=True,
            )
            if common
            else None,
        }
    report["paired"] = {
        a + "_minus_" + b: {
            "full20": paired(metrics, records, a, b, sorted(records)),
            "common_success": paired(metrics, records, a, b, common),
        }
        for a, b in (("Memory_text", "D0"), ("Memory_cue", "D0"), ("Memory_cue", "Memory_text"))
    }
    chunks = [read(p) for p in (root / "memories").glob("*/*.json")]
    report["memory"] = {
        "windows": len(chunks),
        "status_counts": dict(Counter(x["status"] for x in chunks)),
        "units": sum(len(x["events"]) for x in chunks),
        "field_repaired_windows": sum(x.get("repair_attempted", False) for x in chunks),
        "downgraded_cues": sum(len(x.get("downgraded_cues", [])) for x in chunks),
    }
    report["limitations"] = [
        "Original repeatedly explored train20, no independent confirmation",
        "Reader ablation of shared full-cue-generated descriptions; text descriptions may already express cue semantics",
        "D0 is cached from a different serving session and uses direct visual batches, not this full-timeline text reader",
        "Memory construction still reads one sampled picture per2s; observations do not prove missed fast motion",
        "Memory construction/shared cost counted once; this set has one query per video, no measured reuse speedup",
        "Failed new conditions use predeclared flat2.0; their official AP/Hit are not forced to zero; failures separately reported",
        "Text lengths differ; bootstrap10k paired source-video clusters seed42; contrasts are exploratory",
    ]
    write_json(root / "per_query.json", details)
    write_json(root / "report.json", report)
    return report


def audit(root, output):
    if output.exists() or root in output.parents:
        raise ValueError("New audit path outside run required")
    c, protocol_id = verify(root, labels=True)
    for path, expected in c["source_artifact_sha256"].items():
        if file_sha256(path) != expected:
            raise ValueError("Previous artifact changed: " + path)
    counts = Counter()
    used = set()
    for row in read_jsonl(root / "data/manifests/pilot.jsonl"):
        data = read(root / "inputs" / f"{row['qid']}.json")
        result = read(root / "queries" / f"{row['qid']}.json")
        if set(data) != {"row", "frames", "windows"}:
            raise ValueError("Unexpected inference inputs")
        frames = load_frames(data["frames"])
        chunks = []
        for window in data["windows"]:
            chunk = read(root / "memories" / row["vid"] / f"{window['window_id']:03}.json")
            context = [e for prev in chunks[-2:] for e in prev.get("events", [])]
            if chunk["context_event_ids"] != [e["id"] for e in context]:
                raise ValueError("Context drift")
            content, contract, units = memory_request(
                [frames[i] for i in window["bin_ids"]], c["values"], row["duration"], window, context
            )
            payload = request_payload(c["vlm"], content, 8192, contract, SYSTEM)
            rid = digest({"protocol": protocol_id, "stage": "F2_full_memory", "payload": payload})
            request = read(root / "requests" / rid[:2] / f"{rid}.json")
            used.add(rid)
            if request["payload"] != redact_images(payload):
                raise ValueError("Memory payload mismatch")
            if chunk["status"] != "failed":
                replayed = contract.finalize(json.loads(request["text"]), degrade=True)
                if any(request["result"][k] != v for k, v in replayed.items()):
                    raise ValueError("Memory validation replay mismatch")
                rebuilt = namespace_result(request["result"], window, units)
                if any(chunk[k] != v for k, v in rebuilt.items()):
                    raise ValueError("Memory result mismatch")
                if chunk["links"]:
                    raise ValueError("Unexpected graph")
            counts["memory_requests_verified"] += 1
            chunks.append(chunk)
        events = [e for x in chunks for e in x["events"]]
        if digest(events) != result["memory"]["events_sha256"]:
            raise ValueError("Memory digest mismatch")
        normalized = []
        for arm in ARMS:
            out = result[arm]
            if out["status"] == "failed":
                if out["pred_saliency_scores"] != [FALLBACK_SCORE] * len(frames):
                    raise ValueError("Wrong failure policy")
                for req in out["requests"]:
                    used.add(req["request_id"])
                continue
            content, contract = score_request(row, events, arm)
            payload = request_payload(c["vlm"], content, SCORE_MAX_TOKENS, contract, SCORE_SYSTEM)
            rid = out["request_id"]
            request = read(root / "requests" / rid[:2] / f"{rid}.json")
            used.add(rid)
            if request["payload"] != payload or out["pred_saliency_scores"] != request["result"]["scores"]:
                raise ValueError("Text score request/result mismatch")
            replayed = contract.finalize(json.loads(request["text"]))
            if replayed["scores"] != out["pred_saliency_scores"]:
                raise ValueError("Score validation replay mismatch")
            if any(
                part["type"] != "text"
                for m in payload["messages"]
                if isinstance(m["content"], list)
                for part in m["content"]
            ):
                raise ValueError("Visual data supplied to scorer")
            instruction, raw = payload["messages"][1]["content"][0]["text"].split("\n", 1)
            body = json.loads(raw)
            for event in body["memory"]:
                event.pop("values", None)
            payload["messages"][1]["content"][0]["text"] = instruction + "\n" + compact(body)
            normalized.append(payload)
            counts["text_score_requests_verified"] += 1
        if len(normalized) == 2:
            if normalized[0] != normalized[1]:
                raise ValueError("Arms differ beyond cue fields")
            counts["matched_arm_queries"] += 1
        counts["queries"] += 1
        counts["initial_frame_pixels_verified"] += len(frames)
    files = list((root / "requests").glob("*/*.json"))
    if used != {p.stem for p in files}:
        raise ValueError("Missing or orphan requests")
    attempts = Counter()
    finishes = Counter()
    for p in files:
        r = read(p)
        if r["status"] not in {"ok", "failed"}:
            raise ValueError("Pending request")
        if r["stage"] not in {"F2_full_memory", *(a + "_score" for a in ARMS)}:
            raise ValueError("Unexpected phase")
        for a in r["attempts"]:
            attempts[a["status"]] += 1
            for choice in a.get("response", {}).get("choices", []):
                finishes[choice["finish_reason"]] += 1
            if any(k in a["payload"] for k in ("tools", "tool_choice")):
                raise ValueError("Unexpected tool request")
    value = {
        "status": "passed",
        "protocol_id": protocol_id,
        **counts,
        "prior_files_unchanged": len(c["source_artifact_sha256"]),
        "logical_requests": len(files),
        "attempt_status_counts": dict(attempts),
        "finish_reasons": dict(finishes),
        "no_new_visual_input_to_scorers": True,
        "no_D0_inference_dependency": True,
        "no_tool_phases": True,
    }
    write_json(output, value)
    return value


def run(root):
    from transformers import AutoTokenizer

    c, protocol_id = verify(root)
    if (root / "circuit_breaker.json").exists():
        raise RuntimeError("Preserved engineering gate failure")
    served = MemoryClient(c["endpoint"], c["vlm"], root / "requests", protocol_id).check_model()
    tokenizer = AutoTokenizer.from_pretrained(c["vlm"]["weight_path"], local_files_only=True)
    rows = read_jsonl(root / "data/manifests/pilot.jsonl")
    started = time.time()
    completed = []
    session = {"started_at": started, "model": served, "workers": c["workers"]}
    session_path = root / "sessions" / f"{time.time_ns()}.json"
    write_json(session_path, session)
    for batch in (rows[:4], rows[4:]):
        with ThreadPoolExecutor(max_workers=c["workers"]) as pool:
            futures = {pool.submit(run_query, root, c, protocol_id, row, tokenizer): row for row in batch}
            while futures:
                done, _ = wait(futures, timeout=15, return_when=FIRST_COMPLETED)
                for future in done:
                    completed.append(future.result())
                    futures.pop(future)
                progress = {
                    "status": "running",
                    "completed": len(completed),
                    "total": len(rows),
                    "active_qids": [r["qid"] for r in futures.values()],
                    "updated_at": time.time(),
                    "elapsed_seconds": time.time() - started,
                    "completed_memory_windows": len(list((root / "memories").glob("*/*.json"))),
                    "statuses": {a: dict(Counter(r[a]["status"] for r in completed)) for a in ARMS},
                }
                write_json(root / "progress.json", progress)
                if done:
                    print(json.dumps(progress), flush=True)
                if time.time() - started > c["max_wall_seconds"]:
                    raise RuntimeError("Wall budget exhausted")
        if len(completed) == 4 and not any(
            all(r[a]["status"] in {"ok", "partial"} for a in ARMS) for r in completed
        ):
            write_json(
                root / "circuit_breaker.json", {"at": time.time(), "reason": "No complete pair in first4"}
            )
            raise RuntimeError("First4 engineering gate failed")
    report = evaluate(root)
    session.update(status="complete", finished_at=time.time())
    write_json(session_path, session)
    write_json(
        root / "progress.json",
        {
            "status": "complete",
            "completed": len(rows),
            "total": len(rows),
            "elapsed_seconds": time.time() - started,
            "updated_at": time.time(),
        },
    )
    return {a: report["groups"][a]["metrics"]["HL-min-VeryGood"] for a in GROUPS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "evaluate", "audit"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == "prepare":
        result = prepare(args.source.resolve(), root, args.repo.resolve())
    elif args.action == "evaluate":
        result = evaluate(root)
    elif args.action == "audit":
        result = audit(root, args.output.resolve())
    else:
        with locked(root / "runner.lock", blocking=False):
            try:
                result = run(root)
            except Exception as error:
                p = read(root / "progress.json") if (root / "progress.json").exists() else {}
                write_json(
                    root / "progress.json",
                    {**p, "status": "stopped_with_error", "error": str(error), "updated_at": time.time()},
                )
                raise
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
