"""Bind a new prompt run to completed B0 evidence and verify every reused pixel trace."""

import json
from pathlib import Path

from vendor.common import read_jsonl, sha256


def freeze_baseline(parent, rows, audit, label_hashes):
    parent = Path(parent).resolve()
    binding = json.loads((parent / "binding.json").read_text())
    binding_hash = sha256(parent / "binding.json")
    complete = json.loads((parent / "complete.json").read_text())
    if complete["status"] != "complete" or complete["binding_sha256"] != binding_hash:
        raise ValueError("Paired B0 must be complete")
    if sha256(parent / "evaluation.json") != complete["evaluation_sha256"]:
        raise ValueError("B0 evaluation receipt changed")
    if binding.get("prompt_mode", "reze_action_v1") != "reze_action_v1":
        raise ValueError("Paired reference must use the original action prompt")
    if binding["scope"] != "pilot4" or binding["n_queries"] != 4 or binding["summary_fraction"] != 0.02:
        raise ValueError("This adaptation is restricted to the original four-query diagnostic subset")
    if binding["batch_size"] != 16 or binding["window_seconds"] != 3 or binding["sampling_fps"] != 2:
        raise ValueError("Paired B0 sampling protocol differs")
    for name, digest in binding["source_sha256"].items():
        if sha256(parent / "source_snapshot" / name) != digest:
            raise ValueError(f"Frozen B0 source changed: {name}")
    for file, key in (
        ("inference.jsonl", "inference_sha256"),
        ("model_verification.json", "model_verification_sha256"),
        ("media_verification.json", "media_verification_sha256"),
    ):
        if sha256(parent / file) != binding[key]:
            raise ValueError(f"Frozen B0 input changed: {file}")
    old_audit = json.loads((parent / "model_verification.json").read_text())["audit"]
    if audit["files"] != old_audit["files"] or audit["revision"] != old_audit["revision"]:
        raise ValueError("Model weights differ from B0")
    old_rows = read_jsonl(parent / "inference.jsonl")
    if len(rows) != len(old_rows):
        raise ValueError("Paired query count changed")
    for old, new in zip(old_rows, rows, strict=True):
        if {k: v for k, v in old.items() if k != "query"} != {k: v for k, v in new.items() if k != "query"}:
            raise ValueError("Paired query identities, video metadata or order changed")
    if label_hashes != binding["labels_sha256"]:
        raise ValueError("Paired reference files changed")
    files = {
        name: sha256(parent / name) for name in ("binding.json", "complete.json", "evaluation.json", "inference.jsonl")
    }
    for folder in ("chunks", "predictions", "media", "query_summaries", "evaluation_inputs"):
        for path in sorted((parent / folder).rglob("*")):
            if path.is_file():
                files[str(path.relative_to(parent))] = sha256(path)
    return {"schema": "qfvs_paired_baseline_v1", "run": str(parent), "binding_sha256": binding_hash, "files": files}


def paired_receipt(run, binding):
    key, name = "paired_baseline_sha256", "paired_baseline.json"
    if binding.get("pixel_reference_sha256"):
        if binding.get(key):
            raise ValueError("Use one pixel reference, not two")
        key, name = "pixel_reference_sha256", "pixel_reference.json"
    if not binding.get(key):
        return None
    path = Path(run) / name
    if sha256(path) != binding[key]:
        raise ValueError("Paired baseline receipt changed")
    receipt = json.loads(path.read_text())
    if sha256(Path(receipt["run"]) / "binding.json") != receipt["binding_sha256"]:
        raise ValueError("Paired baseline binding changed")
    return receipt


def freeze_pixel_reference(parent, rows, audit, label_hashes):
    """Reuse query-independent video pixels, never another query's scores."""
    parent = Path(parent).resolve()
    old_binding = json.loads((parent / "binding.json").read_text())
    old_rows = read_jsonl(parent / "inference.jsonl")
    receipt = freeze_baseline(parent, old_rows, audit, old_binding["labels_sha256"])
    by_video = {row["vid"]: row for row in old_rows}
    if len(by_video) != 4 or {r["vid"] for r in rows} != set(by_video):
        raise ValueError("Pixel reference must cover exactly the same four videos")
    video_keys = {"vid", "duration", "video_path", "media_sha256", "media_size", "media_mtime_ns", "n_shots"}
    for row in rows:
        if any(row[key] != by_video[row["vid"]][key] for key in video_keys):
            raise ValueError("Pixel reference media identity or native grid differs")
    for name, digest in old_binding["labels_sha256"].items():
        if label_hashes.get(name) != digest:
            raise ValueError("Full validation changed previously frozen evaluation inputs")
    return {
        **receipt,
        "schema": "qfvs_video_pixel_reference_v1",
        "trace_qid_by_video": {vid: row["qid"] for vid, row in by_video.items()},
        "reuses_scores": False,
    }


def check_paired_trace(receipt, row, start, segments, trace, batch_size=16):
    if receipt is None:
        return
    parent = Path(receipt["run"])
    begin = start // batch_size * batch_size
    trace_qid = receipt["trace_qid_by_video"][row["vid"]] if "trace_qid_by_video" in receipt else row["qid"]
    name = f"chunks/{trace_qid}/{begin:06d}.json"
    if sha256(parent / name) != receipt["files"][name]:
        raise ValueError("Paired baseline chunk changed")
    chunk = json.loads((parent / name).read_text())
    offset, count = start - begin, len(segments)
    if chunk["segments"][offset : offset + count] != [list(segment) for segment in segments]:
        raise ValueError("Paired temporal windows differ")
    for key in ("frame_indices", "requested_times", "frame_timestamps", "model_rgb_sha256"):
        if trace[key] != chunk["trace"][key][offset : offset + count]:
            raise ValueError(f"Paired input differs: {key}")
    for key in ("original_height_width", "resized_height_width"):
        if trace[key] != chunk["trace"][key]:
            raise ValueError(f"Paired geometry differs: {key}")
    if offset == 0 and count == len(chunk["segments"]):
        for key in ("sampled_original_rgb_sha256", "model_input_frames"):
            if trace[key] != chunk["trace"][key]:
                raise ValueError(f"Paired input differs: {key}")
