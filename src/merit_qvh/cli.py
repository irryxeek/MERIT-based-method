"""Reproducible preparation, inference, and official QVHighlights evaluation."""

import argparse
import importlib.metadata
import json
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from .core import digest, file_sha256, read_jsonl, validate_manifest, validate_scores, write_json, write_jsonl

REPO = Path(__file__).resolve().parents[2]


def frozen_run(config):
    source = {str(path.relative_to(REPO)): file_sha256(path) for path in sorted((REPO / "src").rglob("*.py"))}
    protocol = {"config": config, "source_sha256": source}
    protocol_id = digest(protocol)
    root = Path(config["artifact_root"]) / "runs" / (config["experiment"] + "-" + protocol_id[:12])
    root.mkdir(parents=True, exist_ok=True)
    path = root / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Frozen protocol changed")
    write_json(path, protocol)
    for name, expected in source.items():
        snapshot = root / "source_snapshot" / name
        if snapshot.exists():
            if file_sha256(snapshot) != expected:
                raise ValueError("Frozen source snapshot changed")
        else:
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_bytes((REPO / name).read_bytes())
    return protocol_id, root


def put_manifest(path, rows):
    if path.exists():
        if read_jsonl(path) != rows:
            raise ValueError(f"Refusing to replace different manifest: {path}")
    else:
        write_jsonl(path, rows)


def inference_row(row, video_root):
    return {
        **{key: row[key] for key in ("qid", "vid", "query", "duration")},
        "video_path": str(Path(video_root) / (row["vid"] + ".mp4")),
    }


def prepare(config, root):
    data = Path(config["annotation_root"])
    train_path = data / "highlight_train_release.jsonl"
    val_path = data / "highlight_val_release.jsonl"
    train, val = read_jsonl(train_path), read_jsonl(val_path)
    unique_train = sorted({row["vid"] for row in train})
    selected = set(random.Random(config["seed"]).sample(unique_train, config["smoke"]["unique_videos"]))
    smoke = []
    for row in sorted(train, key=lambda row: row["qid"]):
        if row["vid"] in selected:
            smoke.append(row)
            selected.remove(row["vid"])
    if len(val) != config["full"]["expected_queries"]:
        raise ValueError("Unexpected validation query count")
    if len({row["vid"] for row in val}) != config["full"]["expected_videos"]:
        raise ValueError("Unexpected validation video count")

    def source_id(vid):
        return vid.rsplit("_", 2)[0]

    overlap = {source_id(row["vid"]) for row in train} & {source_id(row["vid"]) for row in val}
    if overlap:
        raise ValueError("Training and validation source-video overlap")
    summary = {
        "train_annotations_sha256": file_sha256(train_path),
        "val_annotations_sha256": file_sha256(val_path),
        "train_val_source_overlap": 0,
    }
    for split, truth in (("smoke", smoke), ("val", val)):
        rows = validate_manifest([inference_row(row, config["video_root"]) for row in truth])
        missing = [row["video_path"] for row in rows if not Path(row["video_path"]).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} videos: {missing[:3]}")
        # Only the separate evaluator consumes these annotation files.
        put_manifest(root / "manifests" / (split + ".jsonl"), rows)
        put_manifest(root / "ground_truth" / (split + ".jsonl"), truth)
        summary[split] = {
            "queries": len(rows),
            "videos": len({row["vid"] for row in rows}),
            "manifest_sha256": file_sha256(root / "manifests" / (split + ".jsonl")),
        }
    write_json(root / "data_audit.json", summary)
    return summary


def check_revisions(config):
    for directory, expected in (
        (REPO / "third_party/merit", config["upstream_commit"]),
        (Path(config["evaluator_root"]), config["evaluator_commit"]),
    ):
        actual = subprocess.check_output(
            ["git", "-C", str(directory), "rev-parse", "HEAD"], text=True
        ).strip()
        if actual != expected:
            raise ValueError(f"Unexpected Git revision at {directory}")
    evaluator = Path(config["evaluator_root"])
    status = subprocess.check_output(
        ["git", "-C", str(evaluator), "diff", "HEAD", "--", "standalone_eval"], text=True
    )
    if status:
        raise ValueError("Official evaluator has local changes")


def audit_models(config, root, scope="all"):
    audit_path = root / "model_audit.json"
    records = json.loads(audit_path.read_text()) if audit_path.exists() else {}
    for key in ("vlm", "embedding") if scope == "all" else (scope,):
        model = config[key]
        directory = Path(model["weight_path"])
        index = json.loads((directory / "model.safetensors.index.json").read_text())
        metadata_root = directory / ".cache/huggingface/download"
        names = sorted(
            set(index["weight_map"].values())
            | {
                str(path.relative_to(metadata_root))[: -len(".metadata")]
                for path in metadata_root.rglob("*.metadata")
            }
        )
        files = {}
        for name in names:
            path = directory / name
            metadata = directory / ".cache/huggingface/download" / (name + ".metadata")
            if not path.is_file() or not metadata.is_file():
                raise FileNotFoundError(f"Incomplete model download: {path}")
            metadata_lines = metadata.read_text().splitlines()
            if metadata_lines[0] != model["revision"]:
                raise ValueError(f"Unexpected model revision: {path}")
            files[name] = {"size": path.stat().st_size, "sha256": file_sha256(path)}
            if name.endswith(".safetensors") and files[name]["sha256"] != metadata_lines[1]:
                raise ValueError(f"Weight checksum does not match the download metadata: {path}")
        records[key] = {"model": model["model"], "revision": model["revision"], "files": files}
    write_json(audit_path, records)
    return {
        key: {"revision": record["revision"], "files": len(record["files"])}
        for key, record in records.items()
    }


def validate_submission(predictions, truth):
    qids = [row["qid"] for row in predictions]
    true_qids = [row["qid"] for row in truth]
    if len(qids) != len(set(qids)) or len(true_qids) != len(set(true_qids)) or set(qids) != set(true_qids):
        raise ValueError("Evaluation requires exactly one prediction for every ground-truth qid")
    lookup = {row["qid"]: row for row in truth}
    for row in predictions:
        target = lookup[row["qid"]]
        if row["vid"] != target["vid"]:
            raise ValueError("Prediction has the wrong video")
        validate_scores({"scores": row["pred_saliency_scores"]}, int(target["duration"] / 2))


def request_stats(results, memories, root):
    events = [event for result in results + memories for event in result.get("requests", [])]
    ids = {event["request_id"] for event in events}
    by_stage = {}
    for request_id in sorted(ids):
        path = root / "requests" / request_id[:2] / (request_id + ".json")
        if not path.is_file():
            continue
        record = json.loads(path.read_text())
        stage = by_stage.setdefault(record["stage"], Counter())
        stage["logical_requests"] += 1
        for attempt in record["attempts"]:
            stage["attempts"] += 1
            stage["seconds"] += attempt["seconds"]
            usage = attempt.get("response", {}).get("usage", {})
            for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
                stage[name] += usage.get(name, 0)
        content = record["payload"]["messages"][0]["content"]
        stage["submitted_images_including_retries"] += sum(
            part["type"] == "image_url" for part in content
        ) * len(record["attempts"])
    return by_stage


def evaluate(config, root, split):
    check_revisions(config)
    # Labels enter the process only after inference has finished.
    truth = read_jsonl(root / "ground_truth" / (split + ".jsonl"))
    results = [json.loads((root / "queries" / (str(row["qid"]) + ".json")).read_text()) for row in truth]
    predictions = [{key: row[key] for key in ("qid", "vid", "pred_saliency_scores")} for row in results]
    validate_submission(predictions, truth)
    sys.path.insert(0, config["evaluator_root"])
    from standalone_eval.eval import eval_submission

    metrics = eval_submission(predictions, truth, verbose=False, match_number=True)
    memories = [
        json.loads((root / "memories" / (vid + ".json")).read_text())
        for vid in sorted({row["vid"] for row in truth})
        if (root / "memories" / (vid + ".json")).is_file()
    ]
    constant = sum(len(set(row["pred_saliency_scores"])) == 1 for row in results)
    ties = sum(row["pred_saliency_scores"].count(max(row["pred_saliency_scores"])) > 1 for row in results)
    report = {
        "split": split,
        "scope": "MERIT-to-QVHighlights adaptation; frozen open-source models",
        "queries": len(truth),
        "status_counts": dict(Counter(row["status"] for row in results)),
        "constant_score_queries": constant,
        "tied_maximum_queries": ties,
        "mean_retrieval_rounds": sum(len(row["rounds"]) for row in results) / len(results),
        "mean_query_seconds": sum(row["seconds"] for row in results) / len(results),
        "model_requests": request_stats(results, memories, root),
        "official_metrics": metrics,
        "failed_qids": [row["qid"] for row in results if row["status"] != "ok"],
        "failure_stage_counts": dict(
            Counter(
                row.get("failure_stage", "retrieval_or_scoring") for row in results if row["status"] != "ok"
            )
        ),
        "videos_without_complete_memory": len({row["vid"] for row in truth}) - len(memories),
    }
    put_manifest(root / (split + "_predictions.jsonl"), predictions)
    write_json(root / (split + "_report.json"), report)
    if split == "smoke" and not report["failed_qids"]:
        write_json(
            root / "smoke_success.json", {"queries": len(results), "protocol": str(root / "protocol.json")}
        )
    return report


def infer_query(pipeline, row):
    """Apply the whole-query fallback to memory generation and indexing as well."""
    path = pipeline.root / "queries" / (str(row["qid"]) + ".json")
    if path.exists():
        cached = json.loads(path.read_text())
        if cached.get("failure_stage") in {"memory", "index"}:
            identity = digest(
                {
                    "protocol": pipeline.protocol_id,
                    "input": row,
                    "source_sha256": file_sha256(row["video_path"]),
                }
            )
            if cached["identity"] != identity:
                raise ValueError("Failed-query input identity changed")
            return cached
    started = time.monotonic()
    first_event = len(pipeline.client.events)
    stage = "memory"
    try:
        memory = pipeline.memory(row)
        stage = "index"
        vectors = pipeline.index(memory)
    except (RuntimeError, ValueError, OSError) as error:
        source_sha256 = file_sha256(row["video_path"])
        result = {
            "identity": digest(
                {"protocol": pipeline.protocol_id, "input": row, "source_sha256": source_sha256}
            ),
            "protocol_id": pipeline.protocol_id,
            "qid": row["qid"],
            "vid": row["vid"],
            "query": row["query"],
            "source_sha256": source_sha256,
            "status": "failed",
            "failure_stage": stage,
            "error": f"{type(error).__name__}: {error}",
            "pred_saliency_scores": [0.0] * int(row["duration"] / pipeline.config["scoring"]["clip_seconds"]),
            "rounds": [],
            "batches": [],
            "selected_clip_ids": [],
            "unique_selected_frames": 0,
            "seconds": time.monotonic() - started,
            "requests": pipeline.client.events[first_event:],
        }
        write_json(path, result)
        return result
    return pipeline.infer(row, memory, vectors)


def run(config, protocol_id, root, split, endpoint, memory_only=False):
    import fcntl

    from .client import CachedClient
    from .embedding import Embedder
    from .pipeline import Pipeline

    check_revisions(config)
    audit_path = root / "model_audit.json"
    if not audit_path.is_file():
        raise RuntimeError("Run audit-models before inference")
    audited = json.loads(audit_path.read_text())
    if not {"vlm"} <= audited.keys() or (not memory_only and "embedding" not in audited):
        raise RuntimeError("Required model weights have not passed their audit")
    if split == "val" and not memory_only and not (root / "smoke_success.json").is_file():
        raise RuntimeError("The five-video smoke run must pass before validation inference")
    rows = validate_manifest(read_jsonl(root / "manifests" / (split + ".jsonl")))
    client = CachedClient(endpoint, config["vlm"], root / "requests", protocol_id)
    served = client.check_model()
    runtime = {
        "timestamp": time.time(),
        "endpoint": endpoint,
        "served_model": served,
        "python": sys.version,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "sentence-transformers", "decord", "numpy")
        },
    }
    write_json(root / "runtime" / (str(time.time_ns()) + ".json"), runtime)
    pipeline = Pipeline(
        config, protocol_id, root, client, Embedder(config["embedding"], config["artifact_root"])
    )
    with (root / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        consecutive_failures = 0
        done = []
        for index, row in enumerate(rows):
            print(
                json.dumps(
                    {
                        "event": "start_video",
                        "index": index + 1,
                        "total": len(rows),
                        "qid": row["qid"],
                        "vid": row["vid"],
                    }
                ),
                flush=True,
            )
            if memory_only:
                pipeline.memory(row)
                continue
            result = infer_query(pipeline, row)
            consecutive_failures = consecutive_failures + 1 if result["status"] != "ok" else 0
            done.append({"qid": row["qid"], "status": result["status"]})
            write_json(
                root / (split + "_progress.json"),
                {"completed": len(done), "total": len(rows), "queries": done},
            )
            print(
                json.dumps(
                    {
                        "event": "query_done",
                        "qid": row["qid"],
                        "status": result["status"],
                        "seconds": result["seconds"],
                        "rounds": len(result["rounds"]),
                    }
                ),
                flush=True,
            )
            if consecutive_failures >= 3:
                raise RuntimeError("Three consecutive failed queries; stopped to inspect systemic errors")
    return {"memories_complete": True} if memory_only else evaluate(config, root, split)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("prepare", "audit-models", "frames", "memory", "run", "evaluate", "where")
    )
    parser.add_argument("--config", type=Path, default=REPO / "configs/main_v1.json")
    parser.add_argument("--split", choices=("smoke", "val"), default="smoke")
    parser.add_argument("--model-scope", choices=("all", "vlm", "embedding"), default="all")
    parser.add_argument("--endpoint", help="Authorized OpenAI-compatible Qwen endpoint, ending in /v1")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    if config["scoring"]["clip_seconds"] != 2:
        parser.error("The official QVHighlights scoring grid is fixed at 2 seconds")
    protocol_id, root = frozen_run(config)
    print(json.dumps({"run_root": str(root), "protocol_id": protocol_id}), flush=True)
    if args.command == "where":
        return
    if args.command == "prepare":
        output = prepare(config, root)
    elif args.command == "audit-models":
        output = audit_models(config, root, args.model_scope)
    elif args.command == "frames":
        from .media import prepare_frames

        rows = validate_manifest(read_jsonl(root / "manifests" / (args.split + ".jsonl")))
        seen = set()
        for row in rows:
            if row["vid"] not in seen:
                media = prepare_frames(row, config["memory"], config["artifact_root"])
                print(
                    json.dumps(
                        {
                            "vid": row["vid"],
                            "clips": len(media["clips"]),
                            "decoded_duration": media["decoded_duration"],
                        }
                    ),
                    flush=True,
                )
                seen.add(row["vid"])
        output = {"decoded_videos": len(seen)}
    elif args.command == "evaluate":
        output = evaluate(config, root, args.split)
    else:
        if not args.endpoint:
            parser.error("--endpoint is required for model inference")
        output = run(config, protocol_id, root, args.split, args.endpoint, args.command == "memory")
    print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
