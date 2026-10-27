"""Label-free TVSum val10 inputs and post-prediction CPU evaluation.

Only old REZE observations are reused. Selected chunks are decoded one at a
time and checked against their original PTS and RGB receipts. Nothing here
initializes a model, imports generic ``backend``/``prompts`` modules, or reads
annotation values during preparation.
"""

import ast
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
TVSUM = HERE.parent / "reze_tvsum"
EVIDENCE = HERE.parent / "reze_evidence"
QUERY = "Highlight of this video."
TRACE_KEYS = (
    "frame_indices",
    "requested_times",
    "frame_timestamps",
    "raw_decoder_frame_timestamps",
    "original_height_width",
    "resized_height_width",
    "sampled_original_rgb_sha256",
    "model_rgb_sha256",
    "model_input_frames",
)
BASELINE_SOURCES = (
    "media.py",
    "metrics.py",
    "shared.py",
    "vendor/reze_core.py",
    "vendor/umt_static.py",
    "vendor/umt_tvsum.py",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _read(path):
    return json.loads(Path(path).read_text())


def _jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _write_json(path, value):
    """Allow an identical preparation retry, never overwrite a changed artifact."""
    path = Path(path)
    data = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode() + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"Existing prepared artifact changed: {path}")
        return
    with path.open("xb") as stream:
        stream.write(data)


def _receipt(path, expected=None):
    path = Path(path).resolve()
    actual = sha256(path)
    if expected is not None and actual != expected:
        raise ValueError(f"Parent artifact changed: {path}")
    return {"path": str(path), "sha256": actual, "bytes": path.stat().st_size}


def _scoped_module(path, *, skip_from=(), injected=None, functions=None):
    """Execute pinned code without installing its generic imports in sys.modules.

    For shared.py only the named unchanged function bodies are loaded. Other
    modules keep their code apart from explicitly supplied local dependencies.
    All original source bytes are included in the parent receipts.
    """
    tree = ast.parse(Path(path).read_text())
    if functions is not None:
        tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in functions]
        if {n.name for n in tree.body} != set(functions):
            raise ValueError("Required pinned functions are absent")
    else:
        tree.body = [n for n in tree.body if not (isinstance(n, ast.ImportFrom) and n.module in skip_from)]
    namespace = {"__name__": "_revalue_tvsum_scoped", "__file__": str(path), **(injected or {})}
    exec(compile(tree, str(path), "exec"), namespace)
    return SimpleNamespace(**namespace)


def _modules():
    core = _scoped_module(TVSUM / "vendor/reze_core.py")
    shared = _scoped_module(
        TVSUM / "shared.py",
        functions=("second_scores", "validate_chunk", "load_query_chunks"),
        injected={
            "json": json,
            "math": math,
            "Path": Path,
            "extract_pyes": core.extract_pyes,
            "segments_to_curve": core.segments_to_curve,
        },
    )
    return SimpleNamespace(
        core=core,
        shared=shared,
        media=_scoped_module(TVSUM / "media.py", skip_from=("shared",), injected={"segments_for": core.segments_for}),
        metrics=_scoped_module(TVSUM / "metrics.py", skip_from=("shared",), injected={"HERE": TVSUM}),
        frames=_scoped_module(EVIDENCE / "candidate_frames.py", skip_from=("reze_media",)),
        ranking=_scoped_module(EVIDENCE / "ranking.py"),
    )


def val_ids():
    """Read only the pinned author split, never annotation values or titles."""
    tree = ast.parse((TVSUM / "vendor/umt_static.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "TVSUM_SPLITS" for t in node.targets):
            groups = ast.literal_eval(node.value)
            selected = sorted(vid for group in groups.values() for vid in group["val"])
            if len(selected) != 10 or len(set(selected)) != 10:
                raise ValueError("Expected exactly ten unique author validation IDs")
            return selected
    raise ValueError("Missing pinned author split")


def _baseline_index(baseline_root):
    """Read only completion, predictions and label-free manifests."""
    baseline_root = Path(baseline_root).resolve()
    plan = _read(baseline_root / "plan.json")
    complete = _read(baseline_root / "complete.json")
    if complete["status"] != "complete" or sorted(plan["umt_val_ids"]) != val_ids():
        raise ValueError("Baseline is incomplete or has a different val10 split")
    receipts = [
        _receipt(baseline_root / "complete.json"),
        _receipt(baseline_root / "plan.json", complete["plan_sha256"]),
        _receipt(baseline_root / "evaluation.json", complete["evaluation_sha256"]),
    ]
    evaluation = _read(baseline_root / "evaluation.json")
    receipts.append(_receipt(baseline_root / "hd_predictions.jsonl", evaluation["hd_predictions_sha256"]))
    predictions = {r["vid"]: r["scores"] for r in _jsonl(baseline_root / "hd_predictions.jsonl")}
    index = {}
    for part in plan["shards"]:
        shard = baseline_root / part["directory"]
        binding = _read(shard / "binding.json")
        done = _read(shard / "complete.json")
        if done["status"] != "complete":
            raise ValueError("Incomplete baseline shard")
        receipts.extend(
            [
                _receipt(shard / "complete.json"),
                _receipt(shard / "binding.json", done["binding_sha256"]),
                _receipt(shard / "evaluation.json", done["evaluation_sha256"]),
                _receipt(shard / "inference.jsonl", binding["inference_sha256"]),
                _receipt(shard / "media_verification.json", binding["media_verification_sha256"]),
            ]
        )
        for name in BASELINE_SOURCES:
            receipts.append(_receipt(TVSUM / name, binding["source_sha256"][name]))
        rows = _jsonl(shard / "inference.jsonl")
        if [r["qid"] for r in rows] != binding["selected_qids"]:
            raise ValueError("Changed baseline query manifest")
        for row in rows:
            if row["vid"] in index:
                raise ValueError("Duplicate baseline video ID")
            if row["qid"] != row["vid"] or row["query"] != QUERY:
                raise ValueError("Unexpected baseline query")
            index[row["vid"]] = (shard, binding, row)
    if set(index) != set(plan["selected_ids"]) or not set(val_ids()) <= set(predictions):
        raise ValueError("Baseline video coverage changed")
    return index, predictions, receipts


def _prepare_video(shard, binding, source, baseline_scores, output_dir, modules):
    vid = source["vid"]
    binding_hash = sha256(shard / "binding.json")
    receipts = [_receipt(source["video_path"], source["media_sha256"])]
    media_path = shard / "media" / f"{vid}.json"
    receipts.append(_receipt(media_path))
    description = _read(media_path)
    frozen_media = _read(shard / "media_verification.json")[vid]["decoded_timeline"]
    if description != frozen_media:
        raise ValueError("Cached timeline differs from verified parent timeline")
    segments = description["segments"]
    records, chunks = modules.shared.load_query_chunks(shard, source, binding_hash, segments, binding["batch_size"])
    for chunk in chunks:
        receipts.append(_receipt(shard / "chunks" / vid / f"{chunk['start']:06d}.json"))
    curve = modules.shared.second_scores([r["pyes"] for r in records], segments, source["duration"], source["n_seconds"])
    prediction_path = shard / "predictions" / f"{vid}.json"
    receipts.append(_receipt(prediction_path))
    old = _read(prediction_path)
    if old["binding_sha256"] != binding_hash or old["second_scores"] != curve:
        raise ValueError("Saved B0 curve differs from reconstructed raw probabilities")
    scores = modules.metrics.two_second_scores(curve, int(source["duration"] / 2))
    if scores != baseline_scores:
        raise ValueError("B0 two-second readout differs from published baseline")
    selected = modules.ranking.candidates(scores)
    media = {key: [] for key in ("requested_times", "frame_indices", "frame_timestamps")}
    for chunk in chunks:
        for key in media:
            media[key].extend(chunk["trace"][key])
    needed_windows = {
        window
        for candidate in selected
        for _, window, _ in modules.frames.eligible_observations(media, candidate)
    }
    if not needed_windows:
        raise ValueError("Candidates have no old observations")
    reader = modules.media.StreamVideo(source)
    if _canonical(reader.describe()) != _canonical(description):
        raise ValueError("Media timeline changed during candidate preparation")
    clips, verified_chunks, decoded_frames = {}, [], 0
    for chunk in chunks:
        start = chunk["start"]
        windows = set(range(start, start + len(chunk["records"])))
        if not windows & needed_windows:
            continue
        arrays, actual = reader.read(start, len(chunk["records"]))
        for key in TRACE_KEYS:
            if _canonical(actual[key]) != _canonical(chunk["trace"][key]):
                raise ValueError(f"Old frame evidence changed: {vid}/{start}/{key}")
        for window in windows & needed_windows:
            clips[window] = arrays[window - start]
        verified_chunks.append(start)
        decoded_frames += actual["model_input_frames"]
        del arrays
    targets, traces = modules.frames.select_frames(clips, media, selected)
    payload = output_dir / "candidate_payloads" / f"{vid}.npz"
    payload.parent.mkdir(parents=True, exist_ok=True)
    arrays = {str(key): item["video"] for key, item in targets.items()}
    if payload.exists():
        with np.load(payload, allow_pickle=False) as saved:
            if set(saved.files) != set(arrays) or any(not np.array_equal(saved[key], arrays[key]) for key in arrays):
                raise ValueError("Existing candidate payload changed")
    else:
        with payload.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
    trace_path = output_dir / "candidate_traces" / f"{vid}.json"
    _write_json(trace_path, traces)
    summary_path = shard / "query_summaries" / f"{vid}.json"
    receipts.append(_receipt(summary_path))
    summary = _read(summary_path)
    expected_cost = {
        "model_requests": len(records),
        "output_tokens": len(records),
        "input_tokens": sum(r["prompt_tokens"] for r in records),
        "frames": sum(c["trace"]["model_input_frames"] for c in chunks),
    }
    if summary["binding_sha256"] != binding_hash or any(summary[k] != v for k, v in expected_cost.items()):
        raise ValueError("Baseline query cost changed")
    return {
        "qid": vid,
        "vid": vid,
        "query": QUERY,
        "duration": source["duration"],
        "video_path": source["video_path"],
        "media_sha256": source["media_sha256"],
        "scores": scores,
        "baseline_scores": scores,
        "baseline_cost": {
            **expected_cost,
            "inference_seconds": summary["inference_seconds"],
            "decode_preprocess_seconds": summary["decode_preprocess_seconds"],
            "shared_baseline_smoke_and_failures_included": False,
        },
        "candidate_ids": selected,
        "candidate_frames": traces,
        "candidate_payload": str(payload.relative_to(output_dir)),
        "candidate_payload_sha256": sha256(payload),
        "candidate_trace": str(trace_path.relative_to(output_dir)),
        "candidate_trace_sha256": sha256(trace_path),
        "preparation": {
            "verified_baseline_chunk_starts": verified_chunks,
            "decoded_old_frames_for_verification": decoded_frames,
            "selected_old_frames": sum(t["frames_observed"] for t in traces.values()),
            "new_model_observations": 0,
        },
        "parent_receipts": receipts,
    }


def prepare_inputs(baseline_root: Path, output_dir: Path):
    """Persist ten label-free rows and verified old-frame NPZ/JSON payloads.

    ``output_dir`` may be a fresh run subdirectory. An existing complete adapter
    manifest can be resumed only while all saved bytes and parent receipts match.
    The rows deliberately contain no titles, category cues, labels or metrics.
    """
    baseline_root, output_dir = Path(baseline_root).resolve(), Path(output_dir).resolve()
    manifest_path = output_dir / "adapter_manifest.json"
    if manifest_path.exists():
        manifest = _read(manifest_path)
        if manifest["baseline_root"] != str(baseline_root) or manifest["selected_ids"] != val_ids():
            raise ValueError("Adapter manifest belongs to another source or split")
        _receipt(output_dir / "rows.json", manifest["rows_sha256"])
        for receipt in manifest["parent_receipts"]:
            _receipt(receipt["path"], receipt["sha256"])
        rows = _read(output_dir / "rows.json")
        for row in rows:
            load_candidates(output_dir, row)
        return rows
    index, baseline, receipts = _baseline_index(baseline_root)
    modules = _modules()
    rows = []
    for vid in val_ids():
        shard, binding, source = index[vid]
        row = _prepare_video(shard, binding, source, baseline[vid], output_dir, modules)
        rows.append(row)
        receipts.extend(row["parent_receipts"])
    receipts.extend(_receipt(p) for p in (Path(__file__), EVIDENCE / "candidate_frames.py", EVIDENCE / "ranking.py"))
    unique = {r["path"]: r for r in receipts}
    _write_json(output_dir / "rows.json", rows)
    _write_json(
        manifest_path,
        {
            "schema": "revalue_tvsum_val10_adapter_v1",
            "baseline_root": str(baseline_root),
            "selected_ids": val_ids(),
            "query": QUERY,
            "selection_uses_labels": False,
            "annotation_values_loaded": False,
            "candidate_policy": "Original REZE evidence stable Top5; <=4 previously observed frames per candidate",
            "rows_sha256": sha256(output_dir / "rows.json"),
            "parent_receipts": [unique[key] for key in sorted(unique)],
        },
    )
    return rows


def load_candidates(output_dir: Path, row):
    """Load only persisted old frames; validate hashes, shape, time and provenance."""
    output_dir = Path(output_dir).resolve()
    paths = [(output_dir / row[name]).resolve() for name in ("candidate_payload", "candidate_trace")]
    if any(not path.is_relative_to(output_dir) for path in paths):
        raise ValueError("Candidate artifact escapes preparation directory")
    payload, trace_path = paths
    _receipt(payload, row["candidate_payload_sha256"])
    _receipt(trace_path, row["candidate_trace_sha256"])
    traces = _read(trace_path)
    if traces != row["candidate_frames"] or set(traces) != {str(c) for c in row["candidate_ids"]}:
        raise ValueError("Candidate trace IDs changed")
    targets = {}
    with np.load(payload, allow_pickle=False) as arrays:
        if set(arrays.files) != set(traces):
            raise ValueError("Candidate payload IDs changed")
        for candidate in row["candidate_ids"]:
            array, trace = arrays[str(candidate)], traces[str(candidate)]
            if array.dtype != np.uint8 or array.ndim != 4 or array.shape[-1] != 3 or not 1 <= len(array) <= 4:
                raise ValueError("Invalid candidate video tensor")
            if len(array) != trace["frames_observed"] or len(trace["frames"]) != len(array):
                raise ValueError("Candidate frame count changed")
            if hashlib.sha256(array.tobytes()).hexdigest() != trace["model_rgb_sha256"]:
                raise ValueError("Candidate RGB hash changed")
            for frame, observation in zip(array, trace["frames"], strict=True):
                if hashlib.sha256(frame.tobytes()).hexdigest() != observation["resized_rgb_sha256"]:
                    raise ValueError("Candidate per-frame RGB hash changed")
                if not 2 * candidate <= observation["requested_time"] < 2 * candidate + 2:
                    raise ValueError("Candidate frame outside its original interval")
            targets[candidate] = {"video": array, "metadata": trace["metadata"]}
    return targets, traces


def evaluate_predictions(predictions: dict, baseline_root: Path):
    """Evaluate an already frozen complete prediction set, then load CPU-only GT.

    The caller must save final predictions before calling this function. The
    returned digest binds the evaluated values independently of a file format.
    Labels remain in the baseline's evaluation_inputs directory and never enter
    the prepared GPU rows.
    """
    if set(predictions) != set(val_ids()):
        raise ValueError("Evaluation requires exactly the frozen val10 predictions")
    # Copy before opening label bytes, so later caller mutation cannot affect GT use.
    snapshot = json.loads(_canonical(predictions))
    index, baseline, receipts = _baseline_index(baseline_root)
    for vid in val_ids():
        scores = np.asarray(snapshot[vid], dtype=float)
        if scores.shape != (len(baseline[vid]),) or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
            raise ValueError("Prediction length or score range differs from frozen B0")
    label_maps, annotations = {}, {}
    for vid in val_ids():
        shard, binding, _ = index[vid]
        path = shard / "evaluation_inputs/tvsum_anno.json"
        if path not in label_maps:
            receipts.append(_receipt(path, binding["labels_sha256"]["tvsum_anno.json"]))
            label_maps[path] = _read(path)
        annotations[vid] = label_maps[path][vid]
    metrics = _modules().metrics
    details = [{"vid": vid, **metrics.video_metrics(snapshot[vid], annotations[vid]["anno"])} for vid in val_ids()]
    summary = metrics.summarize(details)
    author = metrics.author_top5_map(snapshot, annotations)
    if abs(author - summary["Top5_AP"]["mean"]) > 0.000501:
        raise ValueError("Native UMT evaluator and independent metric disagree")
    return {
        "status": "complete",
        "dataset": "TVSum",
        "scope": "umt_val10_development_transfer",
        "evaluated_videos": len(snapshot),
        "prediction_values_sha256": hashlib.sha256(_canonical(snapshot)).hexdigest(),
        "metrics_percent": summary,
        "author_evaluator_rounded_percent": author,
        "author_metric_parity_passed": True,
        "per_video": details,
        "parent_receipts": receipts,
        "summary_F1_computed": False,
    }
