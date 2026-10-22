"""Bounded parallel queries, restartable checkpoints, and full-denominator HD evaluation."""

import argparse
import importlib.metadata
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from merit_qvh.cli import audit_models, check_revisions, put_manifest, validate_submission
from merit_qvh.core import read_jsonl, validate_manifest, write_json

from .client import Client, locked
from .pipeline import SharedEmbedder, run_query
from .prepare import freeze, prepare_data, verify


def request_cost(root):
    stages = defaultdict(lambda: defaultdict(float))
    for path in (root / "requests").glob("*/*.json"):
        record = json.loads(path.read_text())
        counts = stages[record["stage"]]
        counts["logical_requests"] += 1
        for attempt in record["attempts"]:
            counts["actual_attempts"] += 1
            counts["seconds_including_queue"] += attempt.get("seconds", 0)
            counts["unknown_interrupted_attempts"] += attempt.get("status") == "in_flight"
            usage = attempt.get("response", {}).get("usage", {})
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                counts[key] += usage.get(key, 0)
            messages = attempt.get("payload", record["payload"])["messages"]
            counts["image_presentations"] += sum(
                isinstance(p, dict) and p.get("type") == "image_url"
                for m in messages
                if isinstance(m.get("content"), list)
                for p in m["content"]
            )
    return {key: dict(value) for key, value in stages.items()}


def evaluate(config, root, split="explore"):
    check_revisions(config)
    # This is the only execution phase that reads annotation labels.
    truth = read_jsonl(root / "data/ground_truth" / (split + ".jsonl"))
    rows = [json.loads((root / "queries" / (str(r["qid"]) + ".json")).read_text()) for r in truth]
    sys.path.insert(0, config["evaluator_root"])
    from standalone_eval.eval import eval_submission

    report = {"split": split, "queries": len(rows), "groups": {}, "cost": request_cost(root)}
    sessions = [json.loads(p.read_text()) for p in (root / "sessions").glob("*.json")]
    report["execution"] = {
        "workers": config["hd"]["workers"],
        "sessions": sessions,
        "completed_session_wall_seconds": sum(
            s.get("finished_at", s["started_at"]) - s["started_at"] for s in sessions
        ),
        "query_wall_seconds_this_session_sum": sum(r["wall_seconds_this_session"] for r in rows),
        "request_cache_hit_events_in_final_query_sessions": sum(
            e["cache_hit"] for r in rows for e in r["requests_this_session"]
        ),
        "embedding_cache_hits_in_final_query_sessions": sum(
            e["cache_hit"] for r in rows for e in r.get("embedding_events_this_session", [])
        ),
        "cost_note": "Stage totals are unique actual attempts, include queue time and retries. "
        "Shared stages must not be added twice. Cold full-pipeline speedup was not benchmarked. "
        "Old checkpoint sessions may have more cache hits than final-query event lists.",
    }
    report["group_stage_accounting"] = {
        "M0": ["caption", "keys", "filter", "decide", "score"],
        "D0": ["hd_initialize"],
        "H1": ["caption", "keys", "filter", "decide", "hd_initialize", "hd_local"],
    }
    for group in ("M0", "D0", "H1"):
        results = [r[group] for r in rows]
        predictions = [{k: r[k] for k in ("qid", "vid", "pred_saliency_scores")} for r in results]
        validate_submission(predictions, truth)
        report["groups"][group] = {
            "status_counts": dict(Counter(r["status"] for r in results)),
            "failed_qids": [r["qid"] for r in results if r["status"] != "ok"],
            "metrics": eval_submission(predictions, truth, verbose=False, match_number=True),
        }
        put_manifest(root / f"{split}_{group}_predictions.jsonl", predictions)
    write_json(root / f"{split}_report.json", report)
    return report


def run(root, endpoint):
    root = Path(root)
    config, protocol_id = verify(root)
    check_revisions(config)
    if not (root / "model_audit.json").exists():
        raise RuntimeError("Run model audit before inference")
    if (root / "circuit_breaker.json").exists():
        raise RuntimeError("Circuit breaker requires inspection; no automatic repeated experiment")
    rows = validate_manifest(read_jsonl(root / "data/manifests/explore.jsonl"))
    workers = config["hd"]["workers"]
    if workers not in (1, 2, 4) or len(rows) != 100:
        raise ValueError("Only frozen 100-query exploration with 1/2/4 workers is authorized here")
    client = Client(endpoint, config["vlm"], root / "requests", protocol_id)
    served = client.check_model()
    session = str(time.time_ns())
    runtime = {
        "started_at": time.time(),
        "workers": workers,
        "model": served,
        "python": sys.version,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("vllm", "torch", "transformers", "sentence-transformers", "av", "decord")
        },
        "gpu_start": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader"], text=True
        ),
    }
    with locked(root / "run.lock", blocking=False):
        write_json(root / "sessions" / (session + ".json"), runtime)
        embedder = SharedEmbedder(config["embedding"], config["artifact_root"])
        pending_rows = []
        done = {}
        for row in rows:
            path = root / "queries" / (str(row["qid"]) + ".json")
            if path.exists():
                result = run_query(config, protocol_id, root, endpoint, embedder, row)
                done[row["qid"]] = {g: result[g]["status"] for g in ("M0", "D0", "H1")}
            else:
                pending_rows.append(row)
        failures, halted = 0, False
        queue = iter(pending_rows)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {}

            def submit_next():
                row = next(queue, None)
                if row is not None:
                    future = pool.submit(run_query, config, protocol_id, root, endpoint, embedder, row)
                    futures[future] = row

            for _ in range(workers):
                submit_next()
            while futures:
                completed, _ = wait(futures, timeout=30, return_when=FIRST_COMPLETED)
                for future in completed:
                    row = futures.pop(future)
                    result = future.result()
                    statuses = {g: result[g]["status"] for g in ("M0", "D0", "H1")}
                    done[row["qid"]] = statuses
                    failures = failures + 1 if any(s != "ok" for s in statuses.values()) else 0
                    print(
                        json.dumps(
                            {
                                "qid": row["qid"],
                                "completed": len(done),
                                "status": statuses,
                                "seconds": result["wall_seconds_this_session"],
                            }
                        ),
                        flush=True,
                    )
                    if failures >= 3:
                        halted = True
                        write_json(root / "circuit_breaker.json", {"qid": row["qid"], "at": time.time()})
                    if not halted:
                        submit_next()
                write_json(
                    root / "progress.json",
                    {
                        "at": time.time(),
                        "completed": len(done),
                        "total": len(rows),
                        "active_qids": [r["qid"] for r in futures.values()],
                        "queries": done,
                        "halted": halted,
                        "workers": workers,
                    },
                )
        runtime.update(finished_at=time.time(), completed=len(done), halted=halted)
        write_json(root / "sessions" / (session + ".json"), runtime)
        if halted:
            raise RuntimeError("Three consecutive query failures; inspect circuit_breaker.json")
        report = evaluate(config, root)
        write_json(root / "complete.json", {"at": time.time(), "queries": report["queries"]})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "freeze", "audit-models", "run", "evaluate"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8037/v1")
    args = parser.parse_args()
    if args.command in ("prepare", "freeze"):
        config = json.loads(args.config.read_text())
        result = prepare_data(config) if args.command == "prepare" else freeze(config)
        print(json.dumps(result, default=str, ensure_ascii=False))
    elif args.command == "run":
        run(args.run_root, args.endpoint)
    else:
        config, _ = verify(args.run_root)
        result = (
            audit_models(config, args.run_root)
            if args.command == "audit-models"
            else evaluate(config, args.run_root)
        )
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
