"""Small frozen train-only pilot; labels are read only by prepare/evaluate."""

import argparse
import base64
import importlib.metadata
import json
import os
import shutil
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from hd_adapter.client import Client, locked
from hd_adapter.pipeline import SharedEmbedder
from hd_adapter.runner import request_cost

from merit_qvh.cli import validate_submission
from merit_qvh.core import digest, file_sha256, read_jsonl, validate_manifest, write_json, write_jsonl

from .controller import resolve_evidence
from .inference import compare_candidates
from .pilot_components import (
    choose_windows,
    compile_policy,
    context_for,
    extract_context_memory,
    representative_frames,
    score_evidence,
    select_needs,
)
from .planner import Budget, Need, comparison_pairs, merge_dense_scores

GROUPS = ("D0", "E1", "E2", "E3")


def read(path):
    return json.loads(Path(path).read_text())


def select_rows(rows):
    return sorted(rows, key=lambda r: digest([42, "highlight_evidence_pilot_v1", r["qid"], r["vid"]]))[:20]


def load_frames(saved):
    frames = []
    for f in saved:
        encoded = base64.b64encode(Path(f["path"]).read_bytes()).decode()
        if digest(encoded) != f["jpeg_sha256"]:
            raise ValueError("Cached D0 JPEG changed")
        frames.append({**f, "url": "data:image/jpeg;base64," + encoded})
    return frames


def baseline_material(base, row):
    """Read frozen baseline artifacts without touching its locks or caches."""
    video = Path(row["video_path"])
    stat = video.stat()
    key = digest([str(video.resolve()), stat.st_size, stat.st_mtime_ns])
    info = read(base / "media_probes" / (key + ".json"))
    if file_sha256(video) != info["media_sha256"] or digest(info["pts"]) != info["timeline_sha256"]:
        raise ValueError("Video/timeline changed since D0")
    identity = digest([info["media_sha256"], info["timeline_sha256"], int(row["duration"] / 2), 672, 90])
    source_frames = read(base / "hd_frames" / identity / "frames.json")["frames"]
    frames = [
        {**f, "source_image_id": f["image_id"], "image_id": "f" + str(i)} for i, f in enumerate(source_frames)
    ]
    d0 = read(base / "initial" / (str(row["qid"]) + ".json"))
    if d0["status"] == "ok":
        for batch in d0["batches"]:
            if batch["frame_hashes"] != [frames[i]["jpeg_sha256"] for i in batch["ids"]]:
                raise ValueError("Baseline frame/prediction alignment mismatch")
    return {"D0": d0, "media": info, "frames": frames}


def prepare(config_path, base, destination, repo):
    config = read(config_path)
    base, destination, repo = Path(base), Path(destination), Path(repo)
    if destination.exists():
        raise FileExistsError("Use a new pilot directory; frozen attempts are retained")
    rows = validate_manifest(select_rows(read_jsonl(base / "data/manifests/explore.jsonl")))
    if len(rows) != 20 or len({r["vid"] for r in rows}) != 20:
        raise ValueError("Expected twenty distinct videos from the fixed exploration pool")
    destination.mkdir(parents=True)
    baseline = {str(r["qid"]): baseline_material(base, r) for r in rows}
    audit = read(base / "model_audit.json")
    for kind in ("vlm", "embedding"):
        for name, expected in audit[kind]["files"].items():
            path = Path(config[kind]["weight_path"]) / name
            if path.stat().st_size != expected["size"] or file_sha256(path) != expected["sha256"]:
                raise ValueError("Model content drift: " + kind + "/" + name)
    source_files = sorted(
        p
        for folder in ("src", "experiments/hd_adapter", "experiments/highlight_evidence")
        for p in (repo / folder).rglob("*.py")
    )
    source_hashes = {str(p.relative_to(repo)): file_sha256(p) for p in source_files}
    for p in source_files:
        target = destination / "source" / p.relative_to(repo)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)
    evaluator = Path(config["evaluator_root"]) / "standalone_eval"
    shutil.copytree(
        evaluator,
        destination / "evaluator/standalone_eval",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    evaluator_hashes = {
        str(p.relative_to(destination)): file_sha256(p) for p in (destination / "evaluator").rglob("*.py")
    }
    config["vlm"]["timeout_seconds"] = 240
    values = read(repo / "configs/highlight_value_schema.example.json")["values"]
    for r in rows:
        write_json(destination / "baseline" / (str(r["qid"]) + ".json"), baseline[str(r["qid"])])
    write_jsonl(destination / "data/manifests/pilot.jsonl", rows)
    # Labels are copied to an isolated evaluation input; inference never opens this file.
    truth = {r["qid"]: r for r in read_jsonl(base / "data/ground_truth/explore.jsonl")}
    write_jsonl(destination / "data/ground_truth/pilot.jsonl", [truth[r["qid"]] for r in rows])
    settings = {
        "vlm": config["vlm"],
        "embedding": config["embedding"],
        "values": values,
        "endpoint": "http://127.0.0.1:8038/v1",
        "workers": 4,
        "server": {**config["hd"]["server"], "port": 8038},
        "selection": "first20 sorted sha256 JSON [42,highlight_evidence_pilot_v1,qid,vid] from old train explore",
        "source_baseline": str(base),
        "source_protocol": read(base / "protocol.json"),
        "groups": {
            "D0": "cached original dense baseline",
            "E1": "four-cue memory and local rescoring",
            "E2": "E1 plus bounded missing-evidence views",
            "E3": "E2 plus adjacent pair evidence",
        },
        "memory": {
            "chunk_seconds": 10,
            "context_chunks": 2,
            "images": "D0 midpoint JPEGs",
            "query_independent": True,
            "max_events_per_chunk": 4,
            "links": "current/current or previous2chunks/current; cite both endpoints; before is time ordered",
            "invalid_optional_links": "omit with explicit rejection trace; preserve valid nodes; mark partial",
        },
        "retrieval": "3 windows, round-robin event/cue semantic cosine and D0 peak rankings",
        "budget": {"max_views": 2, "max_new_frames": 48, "frames_per_view": 24, "max_comparisons": 3},
        "comparison": "adjacent score-ranked selected windows with E2 peak >0; at most2 pairs for3 windows",
        "gap_views": "model selects up to2 enumerated <=6s start/end subwindows of selected events; fixed timestamps",
        "score": "common0-4 scale; replace only selected2s bins; cue levels are not summed",
        "failure": "keep last valid dense vector or D0; mark failed phase and all downstream fallbacks",
        "circuit": "first four queries are an engineering gate; stop if all four fail a new phase",
        "automatic_extension": False,
        "max_wall_seconds": 5400,
        "source_sha256": source_hashes,
        "evaluator_sha256": evaluator_hashes,
        "input_sha256": {
            str(p.relative_to(destination)): file_sha256(p)
            for folder in ("baseline", "data")
            for p in (destination / folder).rglob("*")
            if p.is_file()
        },
        "model_audit": audit,
        "packages": {
            n: importlib.metadata.version(n)
            for n in ("vllm", "torch", "transformers", "sentence-transformers", "av", "numpy")
        },
    }
    write_json(destination / "protocol.json", {"protocol_id": digest(settings), "settings": settings})
    write_json(
        destination / "prepared.json", {"time": time.time(), "queries": len(rows), "weights_verified": True}
    )
    return {"root": str(destination), "protocol_id": digest(settings), "qids": [r["qid"] for r in rows]}


def verify(root, *, include_labels=False):
    protocol = read(root / "protocol.json")
    c = protocol["settings"]
    if digest(c) != protocol["protocol_id"]:
        raise ValueError("Protocol changed")
    for name, expected in c["source_sha256"].items():
        if file_sha256(root / "source" / name) != expected:
            raise ValueError("Frozen source drift: " + name)
    for name, expected in {**c["input_sha256"], **c["evaluator_sha256"]}.items():
        if not include_labels and name.startswith("data/ground_truth/"):
            continue
        if file_sha256(root / name) != expected:
            raise ValueError("Frozen input/evaluator drift: " + name)
    return c, protocol["protocol_id"]


def without_urls(value):
    if isinstance(value, dict):
        return {k: without_urls(v) for k, v in value.items() if k != "url"}
    if isinstance(value, list):
        return [without_urls(v) for v in value]
    return value


def run_query(root, c, protocol_id, embedder, row):
    output = root / "queries" / (str(row["qid"]) + ".json")
    with locked(root / "locks" / (str(row["qid"]) + ".lock")):
        if output.exists():
            return read(output)
        started = time.monotonic()
        cached = read(root / "baseline" / (str(row["qid"]) + ".json"))
        scores = list(cached["D0"]["pred_saliency_scores"])
        base = {k: row[k] for k in ("qid", "vid", "query")}
        result = {**base, "protocol_id": protocol_id, "D0": cached["D0"], "errors": []}
        client = Client(c["endpoint"], c["vlm"], root / "requests", protocol_id)
        embedding_events = embedder.bind(row["qid"])
        phase = "memory"
        try:
            if cached["D0"]["status"] != "ok":
                raise ValueError("Original D0 failed; preserve that failure")
            info = cached["media"]
            if file_sha256(row["video_path"]) != info["media_sha256"]:
                raise ValueError("Source changed after freezing")
            frames = load_frames(cached["frames"])
            chunks = []
            for offset in range(0, len(scores), 5):
                index = offset // 5
                chunk_path = root / "memories" / row["vid"] / f"{index:03}.json"
                if chunk_path.exists():
                    chunk = read(chunk_path)
                else:
                    ids = list(range(offset, min(offset + 5, len(scores))))
                    start, end = 2 * ids[0], 2 * (ids[-1] + 1)
                    context = [e for prev in chunks[-2:] for e in prev["events"]]
                    memory = extract_context_memory(
                        client,
                        [frames[i] for i in ids],
                        c["values"],
                        row["duration"],
                        start,
                        end,
                        f"w{index}",
                        context,
                    )
                    chunk = {"window_id": index, "start": start, "end": end, "bin_ids": ids, **memory}
                    write_json(chunk_path, chunk)
                chunks.append(chunk)
            result["memory_chunks"] = len(chunks)
            result["rejected_memory_links"] = sum(len(chunk.get("rejected_links", [])) for chunk in chunks)
            memory_status = "partial" if result["rejected_memory_links"] else "ok"
            phase = "retrieval"
            policy = compile_policy(client, row["query"], c["values"])
            selected, ranks = choose_windows(chunks, scores, row["query"], policy, embedder)
            memory = context_for(chunks, selected)
            ids = sorted(i for chunk in chunks if chunk["window_id"] in selected for i in chunk["bin_ids"])
            images = [frames[i] for i in ids]
            result.update(policy=policy, selected_windows=selected, selected_bin_ids=ids, retrieval=ranks)
            phase = "E1"
            updates = score_evidence(client, "evidence_score_E1", row, ids, images, memory)
            scores = merge_dense_scores(scores, updates)
            result["E1"] = {**base, "status": memory_status, "pred_saliency_scores": scores}
            phase = "evidence_acquisition"
            planned = select_needs(client, row["query"], policy, chunks, selected, row["duration"])
            needs = [Need(**{k: v for k, v in need.items() if k != "reason"}) for need in planned]
            trace = resolve_evidence(
                client,
                row["query"],
                needs,
                [],
                frames,
                row["video_path"],
                info,
                row["duration"],
                budget=Budget(**c["budget"]),
            )
            new_frames = trace["frames"][len(frames) :]
            for f in new_frames:
                path = root / "evidence_frames" / str(row["qid"]) / (f["image_id"] + ".jpg")
                path.parent.mkdir(parents=True, exist_ok=True)
                raw = base64.b64decode(f["url"].split(",", 1)[1])
                if path.exists() and path.read_bytes() != raw:
                    raise ValueError("Resumed source view changed pixels")
                if not path.exists():
                    path.write_bytes(raw)
                f["path"] = str(path)
            result["acquisition"] = {"planned_needs": planned, **without_urls(trace)}
            images += new_frames
            phase = "E2"
            if new_frames:
                updates = score_evidence(
                    client,
                    "evidence_score_E2",
                    row,
                    ids,
                    images,
                    memory,
                    observations=trace["observations"],
                    gaps=trace["gaps"],
                )
                scores = merge_dense_scores(scores, updates)
            view_errors = [s for s in trace["steps"] if s.get("status") == "failed"]
            result["E2"] = {
                **base,
                "status": "partial" if view_errors else memory_status,
                "pred_saliency_scores": scores,
                "reused_E1": not bool(new_frames),
            }
            phase = "comparisons"
            windows = [
                {
                    "id": "w" + str(chunk["window_id"]),
                    "start": chunk["start"],
                    "end": chunk["end"],
                    "score": max(scores[i] for i in chunk["bin_ids"]),
                    "relevant": max(scores[i] for i in chunk["bin_ids"]) > 0,
                }
                for chunk in chunks
                if chunk["window_id"] in selected
            ]
            windows_by_id = {w["id"]: w for w in windows}
            comparisons = []
            for left, right in comparison_pairs(windows, Budget(**c["budget"])):
                evidence, candidates = [], []
                for identifier in (left, right):
                    w = windows_by_id[identifier]
                    photos = representative_frames(trace["frames"], w["start"], w["end"])
                    evidence += photos
                    candidates.append({"id": identifier, "evidence_ids": [f["image_id"] for f in photos]})
                comparison = compare_candidates(
                    client,
                    row["query"],
                    "Query-relevant highlight value: " + ", ".join(policy["facets"]),
                    *candidates,
                    evidence,
                    row["duration"],
                )
                comparisons.append(comparison)
            result["comparisons"] = comparisons
            phase = "E3"
            known = any(p["winner"] != "unknown" for p in comparisons)
            if known:
                updates = score_evidence(
                    client,
                    "evidence_score_E3",
                    row,
                    ids,
                    images,
                    memory,
                    observations=trace["observations"],
                    gaps=trace["gaps"],
                    comparisons=comparisons,
                )
                scores = merge_dense_scores(scores, updates)
            result["E3"] = {
                **base,
                "status": result["E2"]["status"],
                "pred_saliency_scores": scores,
                "reused_E2": not known,
            }
        except Exception as error:
            result["errors"].append({"phase": phase, "type": type(error).__name__, "error": str(error)})
        for group in GROUPS[1:]:
            if group not in result:
                result[group] = {
                    **base,
                    "status": "fallback",
                    "pred_saliency_scores": list(scores),
                    "reason": result["errors"][-1],
                }
        result.update(
            wall_seconds=time.monotonic() - started,
            finished_at=time.time(),
            requests=client.events,
            embedding_events=embedding_events,
        )
        write_json(output, result)
        return result


def evaluate(root):
    c, protocol_id = verify(root, include_labels=True)
    truth = read_jsonl(root / "data/ground_truth/pilot.jsonl")
    rows = [read(root / "queries" / (str(t["qid"]) + ".json")) for t in truth]
    sys.path.insert(0, str(root / "evaluator"))
    from standalone_eval.eval import eval_submission

    report = {
        "protocol_id": protocol_id,
        "queries": len(rows),
        "groups": {},
        "cost": request_cost(root),
        "limitations": [
            "20 previously explored training queries; no validation or significance claim",
            "Incremental shared-compute checkpoints, not a controlled module ablation",
            "Cached D0; pilot cost excludes historical D0/model startup cost",
            "New groups fall back to previous vector; original D0 failures remain unchanged",
            "Context is previous two chunks plus retrieved references, not whole-video visual attention",
            "Structural evidence checks do not establish semantic correctness",
        ],
    }
    for group in GROUPS:
        values = [r[group] for r in rows]
        predictions = [{k: v[k] for k in ("qid", "vid", "pred_saliency_scores")} for v in values]
        validate_submission(predictions, truth)
        path = root / (group + "_predictions.jsonl")
        if path.exists():
            if read_jsonl(path) != predictions:
                raise ValueError("Completed predictions changed")
        else:
            write_jsonl(path, predictions)
        report["groups"][group] = {
            "status_counts": dict(Counter(v["status"] for v in values)),
            "fallback_qids": [v["qid"] for v in values if v["status"] != "ok"],
            "metrics": eval_submission(predictions, truth, verbose=False, match_number=True),
            "changed_queries_vs_D0": sum(
                v["pred_saliency_scores"] != r["D0"]["pred_saliency_scores"]
                for v, r in zip(values, rows, strict=True)
            ),
        }
    report["diagnostics"] = {
        "errors": [{"qid": r["qid"], **e} for r in rows for e in r["errors"]],
        "used_views": sum(r.get("acquisition", {}).get("used_views", 0) for r in rows),
        "new_frames": sum(r.get("acquisition", {}).get("new_frames_decoded", 0) for r in rows),
        "unresolved_queries": sum(bool(r.get("acquisition", {}).get("gaps", [])) for r in rows),
        "comparison_outcomes": dict(Counter(p["winner"] for r in rows for p in r.get("comparisons", []))),
    }
    write_json(root / "report.json", report)
    return report


def run(root):
    c, protocol_id = verify(root)
    if (root / "circuit_breaker.json").exists():
        raise RuntimeError("Pilot engineering gate failed; inspect retained attempts")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"4", c["server"]["gpu_uuids"][0]}:
        raise RuntimeError("Embedding must run on the authorized physical GPU4")
    served = Client(c["endpoint"], c["vlm"], root / "requests", protocol_id).check_model()
    rows = validate_manifest(read_jsonl(root / "data/manifests/pilot.jsonl"))
    embedder = SharedEmbedder(c["embedding"], root)
    started = time.time()
    session_path = root / "sessions" / (str(time.time_ns()) + ".json")
    session = {"started_at": started, "model": served, "workers": c["workers"]}
    write_json(session_path, session)
    completed = []
    with ThreadPoolExecutor(max_workers=c["workers"]) as pool:
        for batch in (rows[:4], rows[4:]):
            futures = {pool.submit(run_query, root, c, protocol_id, embedder, row): row for row in batch}
            for future in as_completed(futures):
                result = future.result()
                completed.append(result)
                progress = {
                    "status": "running",
                    "completed": len(completed),
                    "total": len(rows),
                    "elapsed_seconds": time.time() - started,
                    "last_qid": result["qid"],
                    "E3_status": dict(Counter(r["E3"]["status"] for r in completed)),
                    "updated_at": time.time(),
                }
                write_json(root / "progress.json", progress)
                print(json.dumps(progress), flush=True)
            if len(completed) == 4 and all(r["errors"] for r in completed):
                write_json(
                    root / "circuit_breaker.json",
                    {"reason": "All first4 queries failed", "time": time.time()},
                )
                raise RuntimeError("First4 engineering gate failed")
            if time.time() - started > c["max_wall_seconds"]:
                raise RuntimeError("Pilot wall budget exhausted")
    report = evaluate(root)
    session.update(finished_at=time.time(), status="complete")
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
    return {g: report["groups"][g] for g in GROUPS}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run", "evaluate"))
    parser.add_argument("--root", required=True)
    parser.add_argument("--config")
    parser.add_argument("--baseline")
    parser.add_argument("--repo", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    if args.action == "prepare":
        result = prepare(args.config, args.baseline, root, args.repo)
    elif args.action == "run":
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
    else:
        result = evaluate(root)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
