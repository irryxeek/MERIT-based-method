"""C10 memory-consumption ablation on the frozen twenty-query exploration set.

A/B/C issue new E1 requests. D_full reuses the audited C10 E1 result from the
completed repair run. No memory construction, candidate selection or new frames.
"""

import argparse
import copy
import importlib.metadata
import json
import shutil
import sys
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import numpy as np
from hd_adapter.client import Client, locked
from hd_adapter.runner import request_cost

from merit_qvh.cli import validate_submission
from merit_qvh.client import redact_images
from merit_qvh.core import digest, file_sha256, parse_json, read_jsonl, write_json, write_jsonl

from .granularity import paired, verify
from .pilot import load_frames, read
from .pilot_components import score_evidence
from .planner import merge_dense_scores

NEW_ARMS = ("A_frames", "B_description", "C_cues")
ARMS = (*NEW_ARMS, "D_full")
GROUPS = ("D0", *ARMS)
CONTRASTS = (
    ("A_frames", "D0"),
    ("B_description", "A_frames"),
    ("C_cues", "B_description"),
    ("D_full", "C_cues"),
    ("B_description", "D0"),
    ("C_cues", "D0"),
    ("D_full", "D0"),
)


def memory_view(memory, arm):
    """Hold the event/reference pool fixed; remove only declared visible fields."""
    if arm not in ARMS:
        raise ValueError("Unknown memory condition")
    view = copy.deepcopy(memory)
    if arm == "A_frames":
        view.update(selected_events=[], reference_events=[], links=[])
    elif arm == "B_description":
        for event in view["selected_events"] + view["reference_events"]:
            event.pop("values")
        view["links"] = []
    elif arm == "C_cues":
        view["links"] = []
    return view


class CaptureScorer:
    """Call the real scorer without inference to reconstruct its exact payload."""

    def __init__(self, config, scores):
        self.config, self.scores = config, scores

    def complete(self, stage, content, max_tokens, validator, *, system):
        self.payload = {
            "model": self.config["model"],
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
            "temperature": self.config["temperature"],
            "max_tokens": max_tokens,
            "seed": 42,
            "response_format": {"type": "json_object"},
        }
        return validator({"clip_scores": self.scores})


def expected_payload(c, data, memory, scores):
    probe = CaptureScorer(c["vlm"], scores)
    score_evidence(
        probe, "unused", data["row"], data["selected_bin_ids"], load_frames(data["frames"]), memory
    )
    return redact_images(probe.payload)


def prepare(source, root, repo):
    if root.exists():
        raise FileExistsError("Use a new directory; retain previous experiments")
    old, old_id = verify(source, labels=True)
    if (
        read(source / "progress.json")["status"] != "complete"
        or old.get("memory_contract") != "structured_fields_v2"
    ):
        raise ValueError("A completed structured-memory repair run is required")
    rows = read_jsonl(source / "data/manifests/pilot.jsonl")
    if len(rows) != 20 or len(old["active_qids"]) != 14:
        raise ValueError("Expected the original20 with fourteen frozen-candidate queries")
    root.mkdir(parents=True)
    for row in rows:
        qid = str(row["qid"])
        data = read(source / "inputs" / (qid + ".json"))
        original = read(source / "queries" / (qid + ".json"))
        ref = original["C10"]
        score_request = None
        if data["selected_bin_ids"]:
            if ref["status"] not in {"ok", "partial"} or not ref["memory_coverage_complete"]:
                raise ValueError("All active C10 references must have complete memory and E1")
            candidates = [e for e in ref["requests"] if e["stage"] == "C10_E1"]
            if len(candidates) != 1:
                raise ValueError("Expected one frozen C10 E1 request per active query")
            rid = candidates[0]["request_id"]
            path = source / "requests" / rid[:2] / (rid + ".json")
            score_request = read(path)
            if score_request["status"] != "ok":
                raise ValueError("Cannot reuse a failed scorer request")
            updates = parse_json(score_request["text"])["clip_scores"]
            if merge_dense_scores(data["D0"]["pred_saliency_scores"], updates) != ref["pred_saliency_scores"]:
                raise ValueError("Cached reference scores do not match their original response")
            if score_request["payload"] != expected_payload(old, data, ref["scoring_memory"], updates):
                raise ValueError("Cached D scorer payload differs from the current scorer contract")
        else:
            if ref["pred_saliency_scores"] != data["D0"]["pred_saliency_scores"]:
                raise ValueError("Inherited no-candidate query differs from D0")
        target = root / "inputs" / (qid + ".json")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / "inputs" / target.name, target)
        write_json(root / "references" / (qid + ".json"), {"C10": ref, "request": score_request})
    for name, expected in old["model_audit"]["files"].items():
        path = Path(old["vlm"]["weight_path"]) / name
        if path.stat().st_size != expected["size"] or file_sha256(path) != expected["sha256"]:
            raise ValueError("Frozen VLM weights changed: " + name)
    source_hashes = {}
    for folder in ("src", "experiments/hd_adapter", "experiments/highlight_evidence"):
        for path in sorted((repo / folder).rglob("*.py")):
            name = str(path.relative_to(repo))
            target = root / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            source_hashes[name] = file_sha256(path)
    for name in ("evaluator", "data"):
        shutil.copytree(source / name, root / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    origins = {
        str(p.relative_to(source)): file_sha256(p)
        for p in source.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.suffix not in {".lock", ".pyc"}
    }
    c = {
        "experiment": "C10_memory_consumption_ablation_v1",
        "source_run": str(source),
        "source_protocol_id": old_id,
        "source_artifact_sha256": origins,
        "vlm": old["vlm"],
        "model_audit": old["model_audit"],
        "server": old["server"],
        "endpoint": old["endpoint"],
        "workers": 4,
        "active_qids": old["active_qids"],
        "inherited_no_candidate_qids": old["inherited_no_candidate_qids"],
        "new_arms": list(NEW_ARMS),
        "cached_arm": "D_full",
        "memory_views": {
            "A_frames": "empty node/reference/edge lists; same common caution",
            "B_description": "same selected/reference nodes as D; remove values from BOTH pools and all links",
            "C_cues": "same nodes/values as D; remove links only",
            "D_full": "exact cached C10 memory, scorer payload and result from completed repair run",
        },
        "planned_new_logical_requests": 42,
        "E1_max_tokens": 1024,
        "initial_images_per_condition": 210,
        "failure": "retain complete D0 vector for a failed score request; keep all20 in evaluation",
        "scope": "memory-reader input ablation only; no construction, retrieval, new frames, E2/E3 or sample expansion",
        "max_wall_seconds": 1800,
        "contrasts": [list(x) for x in CONTRASTS],
        "source_sha256": source_hashes,
        "input_sha256": {
            str(p.relative_to(root)): file_sha256(p)
            for folder in ("inputs", "references", "data", "evaluator")
            for p in (root / folder).rglob("*")
            if p.is_file()
        },
        "packages": {n: importlib.metadata.version(n) for n in ("vllm", "torch", "transformers", "numpy")},
    }
    write_json(root / "protocol.json", {"protocol_id": digest(c), "settings": c})
    write_json(
        root / "prepared.json", {"at": time.time(), "weights_verified": True, "queries": 20, "active": 14}
    )
    return {"root": str(root), "protocol_id": digest(c), "planned_new_requests": 42}


def run_query(root, c, protocol_id, row):
    qid = str(row["qid"])
    destination = root / "queries" / (qid + ".json")
    with locked(root / "locks" / (qid + ".lock")):
        if destination.exists():
            return read(destination)
        data = read(root / "inputs" / (qid + ".json"))
        reference = read(root / "references" / (qid + ".json"))["C10"]
        started = time.time()
        result = {
            **{k: row[k] for k in ("qid", "vid", "query")},
            "protocol_id": protocol_id,
            "D0": data["D0"],
            "selected_windows": data["selected_windows"],
            "selected_bin_ids": data["selected_bin_ids"],
            "D_full": {
                "status": reference["status"],
                "pred_saliency_scores": reference["pred_saliency_scores"],
                "reused": True,
                "source_protocol_id": c["source_protocol_id"],
                "new_requests": [],
            },
        }
        frames = load_frames(data["frames"]) if data["selected_bin_ids"] else []
        order = sorted(NEW_ARMS, key=lambda a: digest([42, row["qid"], a]))
        result["arm_execution_order"] = order
        for arm in order:
            path = root / "arms" / arm / (qid + ".json")
            if path.exists():
                result[arm] = read(path)
                continue
            tick = time.time()
            out = {
                "status": "inherited_no_candidates",
                "pred_saliency_scores": data["D0"]["pred_saliency_scores"],
                "errors": [],
                "requests": [],
            }
            if frames:
                client = Client(c["endpoint"], c["vlm"], root / "requests", protocol_id)
                memory = memory_view(reference["scoring_memory"], arm)
                out["memory_sha256"] = digest(memory)
                try:
                    updates = score_evidence(
                        client, arm + "_E1", row, data["selected_bin_ids"], frames, memory
                    )
                    out.update(
                        status="ok",
                        pred_saliency_scores=merge_dense_scores(data["D0"]["pred_saliency_scores"], updates),
                    )
                except (RuntimeError, ValueError, TypeError, KeyError, OSError) as error:
                    out.update(status="fallback", errors=[{"phase": "E1", "error": str(error)}])
                out["requests"] = client.events
            out["wall_seconds"] = time.time() - tick
            write_json(path, out)
            result[arm] = out
        result["wall_seconds"] = time.time() - started
        write_json(destination, result)
        return result


def evaluate(root):
    c, protocol_id = verify(root, labels=True)
    truth = read_jsonl(root / "data/ground_truth/pilot.jsonl")
    records = {t["qid"]: read(root / "queries" / (str(t["qid"]) + ".json")) for t in truth}
    sys.path.insert(0, str(root / "evaluator"))
    from standalone_eval.eval import compute_ap_from_tuple, eval_submission, mk_gt_scores

    metrics, details = {}, []
    report = {
        "protocol_id": protocol_id,
        "queries": len(truth),
        "active_queries": len(c["active_qids"]),
        "groups": {},
        "new_request_cost": request_cost(root),
    }
    for t in truth:
        q, r = t["qid"], records[t["qid"]]
        labels = (mk_gt_scores(t) >= 4).astype(float)
        metrics[q] = {}
        detail = {
            "qid": q,
            "vid": r["vid"],
            "query": r["query"],
            "metrics": metrics[q],
            "status": {},
            "changed_bins": {},
            "top_ties": {},
        }
        for arm in GROUPS:
            scores = np.asarray(r[arm]["pred_saliency_scores"])
            if len(scores) != len(labels) or not np.isfinite(scores).all():
                raise ValueError("Invalid score coverage")
            metrics[q][arm] = {
                "mAP": float(
                    100 * np.mean([compute_ap_from_tuple((0, i, labels[:, i], scores))[2] for i in range(3)])
                ),
                "Hit1": float(100 * labels[int(scores.argmax())].max()),
            }
            changed = np.flatnonzero(scores != np.asarray(r["D0"]["pred_saliency_scores"])).tolist()
            if not set(changed) <= set(r["selected_bin_ids"]):
                raise ValueError("Scores changed outside the frozen candidate set")
            detail["status"][arm] = r[arm]["status"]
            detail["changed_bins"][arm] = changed
            detail["top_ties"][arm] = int((scores == scores.max()).sum())
        details.append(detail)
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
        path = root / (arm + "_predictions.jsonl")
        if path.exists():
            if read_jsonl(path) != predictions:
                raise ValueError("Completed predictions changed")
        else:
            write_jsonl(path, predictions)
        official = eval_submission(predictions, truth, verbose=False, match_number=True)
        for key, name in (("HL-mAP", "mAP"), ("HL-Hit1", "Hit1")):
            if official["HL-min-VeryGood"][key] != round(
                float(np.mean([m[arm][name] for m in metrics.values()])), 2
            ):
                raise ValueError("Per-query metrics differ from the official evaluator")
        active_truth = [t for t in truth if t["qid"] in c["active_qids"]]
        active_predictions = [p for p in predictions if p["qid"] in c["active_qids"]]
        report["groups"][arm] = {
            "metrics": official,
            "active14_metrics": eval_submission(
                active_predictions, active_truth, verbose=False, match_number=True
            ),
            "status_counts": dict(Counter(r[arm]["status"] for r in records.values())),
            "changed_bins_vs_D0": sum(len(d["changed_bins"][arm]) for d in details),
            "top_tied_queries": sum(d["top_ties"][arm] > 1 for d in details),
        }
    report["paired"] = {
        left + "_minus_" + right: {
            "full20": paired(metrics, records, left, right, sorted(records)),
            "active14": paired(metrics, records, left, right, c["active_qids"]),
        }
        for left, right in CONTRASTS
    }
    report["cached_reference_cost"] = read(Path(c["source_run"]) / "report.json")["cost"]["C10_E1"]
    report["limitations"] = [
        "Explored train20, six inherited no-candidate records remain D0; no new holdout or automatic expansion",
        "Reader-input ablation of previously generated memory, not independent memory-construction ablations",
        "Descriptions may already express cue semantics; removing structured values does not erase such prose",
        "Same selected/reference event pool is held fixed for B/C/D; no reference-node pruning or reretrieval",
        "D reuses a previous serving session with matching weights/settings; batch/numerical nondeterminism is not excluded",
        "Text lengths/tokens are measured, not matched; D has zero new-call cost but a recorded historical scorer cost",
        "A versus D0 changes scorer prompt/batch context as well as repeating scoring; it does not isolate repetition alone",
        "Bootstrap uses10000 paired source-video cluster resamples seed42; multiple contrasts are exploratory",
    ]
    write_json(root / "per_query.json", details)
    write_json(root / "report.json", report)
    return report


def audit(root, output):
    if output.exists() or root in output.parents:
        raise ValueError("Use a new audit file outside the frozen run")
    c, protocol_id = verify(root, labels=True)
    if read(root / "progress.json")["status"] != "complete":
        raise ValueError("Audit only completed runs")
    source = Path(c["source_run"])
    for name, expected in c["source_artifact_sha256"].items():
        if file_sha256(source / name) != expected:
            raise ValueError("An original repair-run artifact changed: " + name)
    requests = {p.stem: read(p) for p in (root / "requests").glob("*/*.json")}
    used, counts = set(), Counter()
    for path in sorted((root / "inputs").glob("*.json")):
        data = read(path)
        qid = str(data["row"]["qid"])
        r = read(root / "queries" / (qid + ".json"))
        ref = read(root / "references" / (qid + ".json"))
        if file_sha256(path) != file_sha256(source / "inputs" / path.name):
            raise ValueError("Input differs from its original twenty-query material")
        if r["D_full"]["pred_saliency_scores"] != ref["C10"]["pred_saliency_scores"]:
            raise ValueError("Reused D reference changed")
        if (r["selected_bin_ids"], r["selected_windows"]) != (
            data["selected_bin_ids"],
            data["selected_windows"],
        ):
            raise ValueError("Candidates changed")
        if not data["selected_bin_ids"]:
            if any(
                r[a]["requests"] or r[a]["pred_saliency_scores"] != data["D0"]["pred_saliency_scores"]
                for a in NEW_ARMS
            ):
                raise ValueError("Inherited no-candidate query was changed or inferred")
            counts["inherited_no_candidates"] += 1
            continue
        common = []
        for arm in ARMS:
            memory = memory_view(ref["C10"]["scoring_memory"], arm)
            if arm == "D_full":
                req = ref["request"]
                counts["cached_D_requests_verified"] += 1
            else:
                events = r[arm]["requests"]
                if len(events) != 1 or events[0]["stage"] != arm + "_E1":
                    raise ValueError("Each active arm must have exactly one logical E1 request")
                req = requests[events[0]["request_id"]]
                used.add(req["request_id"])
                if req["stage"] != arm + "_E1" or r[arm]["memory_sha256"] != digest(memory):
                    raise ValueError("Request stage or memory projection changed")
            scores = [{"clip_id": i, "score": 0} for i in data["selected_bin_ids"]]
            expected = expected_payload(c, data, memory, scores)
            if req["payload"] != expected:
                raise ValueError(
                    "Actual request differs from frozen frames/prompt/declared memory projection"
                )
            normalized = copy.deepcopy(expected)
            prompt, raw = normalized["messages"][1]["content"][0]["text"].split("\n", 1)
            body = json.loads(raw)
            body.pop("memory")
            normalized["messages"][1]["content"][0]["text"] = prompt + "\n" + json.dumps(body)
            common.append(digest(normalized))
            counts[arm + "_initial_images"] += len(data["frames"])
        if len(set(common)) != 1:
            raise ValueError("Scoring requests differ beyond their memory field")
        counts["matched_four_condition_queries"] += 1
    if used != set(requests) or len(requests) != c["planned_new_logical_requests"]:
        raise ValueError("Missing or orphan scorer requests")
    if any(r["status"] == "pending" for r in requests.values()):
        raise ValueError("Pending requests in completed experiment")
    attempts = [a for r in requests.values() for a in r["attempts"]]
    result = {
        "status": "passed",
        "protocol_id": protocol_id,
        "queries": 20,
        "source_artifacts_unchanged": len(c["source_artifact_sha256"]),
        "counters": dict(counts),
        "new_request_statuses": dict(Counter(r["status"] for r in requests.values())),
        "attempt_statuses": dict(Counter(a["status"] for a in attempts)),
        "finish_reasons": dict(
            Counter(a.get("response", {}).get("choices", [{}])[0].get("finish_reason") for a in attempts)
        ),
        "actual_attempts": len(attempts),
        "note": "All initial scorer requests match except memory. Cached D has no new inference. "
        "This verifies provenance and projection, not semantic correctness.",
    }
    write_json(output, result)
    return result


def run(root):
    c, protocol_id = verify(root)
    if (root / "circuit_breaker.json").exists():
        raise RuntimeError("Engineering gate previously failed; keep its attempts")
    model = Client(c["endpoint"], c["vlm"], root / "requests", protocol_id).check_model()
    rows = read_jsonl(root / "data/manifests/pilot.jsonl")
    started = time.time()
    session = {"started_at": started, "model": model, "workers": c["workers"]}
    session_path = root / "sessions" / (str(time.time_ns()) + ".json")
    write_json(session_path, session)
    completed = []
    for batch in (rows[:4], rows[4:]):
        with ThreadPoolExecutor(max_workers=c["workers"]) as pool:
            futures = {pool.submit(run_query, root, c, protocol_id, row): row for row in batch}
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
                    "statuses": {a: dict(Counter(r[a]["status"] for r in completed)) for a in ARMS},
                }
                write_json(root / "progress.json", progress)
                if done:
                    print(json.dumps(progress), flush=True)
                if time.time() - started > c["max_wall_seconds"]:
                    raise RuntimeError("Wall budget exhausted")
        if len(completed) == 4 and not any(
            r["selected_bin_ids"] and all(r[a]["status"] == "ok" for a in NEW_ARMS) for r in completed
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
            "completed": 20,
            "total": 20,
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
                progress = read(root / "progress.json") if (root / "progress.json").exists() else {}
                write_json(
                    root / "progress.json",
                    {
                        **progress,
                        "status": "stopped_with_error",
                        "error": str(error),
                        "updated_at": time.time(),
                    },
                )
                raise
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
