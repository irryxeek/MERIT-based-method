"""Frozen OpenAI CLIP center-frame baseline with bounded parallel decoding."""

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from clip_media import normalized_features, select_centers
from common import (
    ROOT,
    STORE,
    atomic_json,
    read_jsonl,
    sha256,
    utc_now,
    validate_manifest,
    validate_scores,
    write_jsonl,
)


def save_array(path, array):
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config_path = ROOT / "configs/clip_protocol_v1.json"
    config = json.loads(config_path.read_text())
    source = ROOT / "third_party/CLIP"
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    assert commit == config["upstream_commit"]
    assert sha256(config["weights"]) == config["weights_sha256"]
    data_audit = json.loads((ROOT / "manifests/data_audit.json").read_text())
    manifest = ROOT / "manifests" / ("smoke_inference.jsonl" if args.smoke else "val_inference.jsonl")
    rows = read_jsonl(manifest)
    validate_manifest(rows)
    if not args.smoke:
        assert sha256(manifest) == data_audit["manifest_sha256"]
        gate = json.loads((STORE / "runs/clip_smoke/completion.json").read_text())
        assert gate["status"] == "complete" and gate["single_frame_check_passed"]
    run = STORE / "runs" / ("clip_smoke" if args.smoke else "clip_val")
    run.mkdir(parents=True, exist_ok=True)
    lock = (run / "run.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    binding = {
        "config_sha256": sha256(config_path),
        "manifest_sha256": sha256(manifest),
        "runner_sha256": sha256(__file__),
        "media_sha256": sha256(ROOT / "scripts/clip_media.py"),
        "common_sha256": sha256(ROOT / "scripts/common.py"),
    }
    if (run / "binding.json").exists():
        assert json.loads((run / "binding.json").read_text()) == binding, "Frozen run changed"
    else:
        atomic_json(run / "binding.json", binding)
        snapshot = run / "source_snapshot"
        snapshot.mkdir()
        for name in ["run_clip.py", "clip_media.py", "common.py", "evaluate.py"]:
            shutil.copy2(ROOT / "scripts" / name, snapshot / name)
        shutil.copy2(config_path, snapshot / config_path.name)
    started = time.time()
    by_video = defaultdict(list)
    for index, row in enumerate(rows):
        by_video[row["vid"]].append((index, row))
    status = {
        "status": "loading_model",
        "invocation_started": utc_now(),
        "planned_queries": len(rows),
        "planned_videos": len(by_video),
        "completed_queries": 0,
        "completed_videos": 0,
    }
    atomic_json(run / "progress.json", status)
    import decord
    import torch
    from PIL import Image

    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    sys.path.insert(0, str(source))
    import clip

    model, preprocess = clip.load(config["weights"], device="cuda:0", jit=False)
    model.eval()
    packages = ["torch", "torchvision", "numpy", "decord", "Pillow", "ftfy", "regex"]
    atomic_json(
        run / "runtime.json",
        {
            "python": sys.version,
            "packages": {name: importlib.metadata.version(name) for name in packages},
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu": torch.cuda.get_device_name(0),
            "model_dtype": str(model.dtype),
            "model_load_seconds": time.time() - started,
        },
    )
    text_start = time.time()
    text_features = []
    with torch.inference_mode():
        for begin in range(0, len(rows), config["text_batch_size"]):
            tokens = clip.tokenize(
                [r["query"] for r in rows[begin : begin + config["text_batch_size"]]], context_length=77, truncate=False
            ).to("cuda:0")
            text_features.append(model.encode_text(tokens).float().cpu().numpy())
    text_features = normalized_features(np.concatenate(text_features))
    save_array(run / "query_features.npy", text_features)
    text_seconds = time.time() - text_start
    cache_root = STORE / "cache/clip_center_v1"
    cache_root.mkdir(parents=True, exist_ok=True)

    def prepare_video(vid):
        row = by_video[vid][0][1]
        before = time.time()
        video_sha = sha256(row["video_path"])
        directory = cache_root / vid
        trace_path = directory / "trace.json"
        feature_path = directory / "features.npy"
        # Smoke retries repeat the single-image equivalence check on fresh pixels.
        if not args.smoke and trace_path.exists() and feature_path.exists():
            trace = json.loads(trace_path.read_text())
            assert trace["source_sha256"] == video_sha
            assert trace["config_sha256"] == binding["config_sha256"]
            assert trace["media_sha256"] == binding["media_sha256"]
            assert trace["features_sha256"] == sha256(feature_path)
            return vid, None, trace, np.load(feature_path, allow_pickle=False)
        reader = decord.VideoReader(row["video_path"], ctx=decord.cpu(0), num_threads=1)
        times = np.asarray(reader.get_frame_timestamp(np.arange(len(reader))))
        indices, requested, observed = select_centers(times, row["duration"])
        arrays = reader.get_batch(indices.tolist()).asnumpy()
        pixels = torch.stack([preprocess(Image.fromarray(frame)) for frame in arrays])
        trace = {
            "vid": vid,
            "source_path": row["video_path"],
            "source_sha256": video_sha,
            "config_sha256": binding["config_sha256"],
            "media_sha256": binding["media_sha256"],
            "annotation_duration": row["duration"],
            "decoded_duration": observed,
            "requested_times": requested.tolist(),
            "frame_indices": indices.tolist(),
            "frame_timestamps": times[indices].tolist(),
            "terminal_centers_clamped": int(np.sum(requested >= observed)),
            "sampled_rgb_sha256": hashlib.sha256(arrays.tobytes()).hexdigest(),
            "frame_count": len(indices),
            "decode_preprocess_seconds": time.time() - before,
        }
        return vid, pixels, trace, None

    videos = iter(by_video)
    futures = deque()
    feature_seconds, reused_videos = 0.0, 0
    single_frame_check = False
    with ThreadPoolExecutor(max_workers=config["decoder_workers"]) as pool:
        for _ in range(config["decoder_workers"] * 2):
            vid = next(videos, None)
            if vid is not None:
                futures.append((vid, pool.submit(prepare_video, vid)))
        while futures:
            vid, future = futures.popleft()
            before = time.time()
            try:
                _, pixels, trace, features = future.result()
                if features is None:
                    chunks = []
                    infer_start = time.time()
                    with torch.inference_mode():
                        for begin in range(0, len(pixels), config["image_batch_size"]):
                            chunks.append(
                                model.encode_image(pixels[begin : begin + config["image_batch_size"]].to("cuda:0"))
                                .float()
                                .cpu()
                                .numpy()
                            )
                    features = normalized_features(np.concatenate(chunks))
                    if args.smoke:
                        with torch.inference_mode():
                            single = model.encode_image(pixels[:1].to("cuda:0")).float().cpu().numpy()
                        np.testing.assert_allclose(features[:1], normalized_features(single), atol=2e-3, rtol=2e-3)
                        single_frame_check = True
                    trace["encode_seconds"] = time.time() - infer_start
                    feature_seconds += trace["encode_seconds"]
                    directory = cache_root / vid
                    directory.mkdir(parents=True, exist_ok=True)
                    save_array(directory / "features.npy", features)
                    trace["features_sha256"] = sha256(directory / "features.npy")
                    atomic_json(directory / "trace.json", trace)
                else:
                    reused_videos += 1
                assert features.shape == (int(by_video[vid][0][1]["duration"] / 2), 512)
                assert np.isfinite(features).all()
                for index, row in by_video[vid]:
                    path = run / "queries" / f"{row['qid']}.json"
                    if not path.exists():
                        scores = (features @ text_features[index]).tolist()
                        validate_scores(scores, int(row["duration"] / 2))
                        prediction = {k: row[k] for k in ["qid", "vid", "query"]}
                        prediction["pred_saliency_scores"] = scores
                        atomic_json(
                            path,
                            {
                                "status": "ok",
                                "qid": row["qid"],
                                "vid": vid,
                                "prediction": prediction,
                                "finished_at": utc_now(),
                                "video_trace": str(cache_root / vid / "trace.json"),
                            },
                        )
                status.update(
                    status="running",
                    completed_queries=status["completed_queries"] + len(by_video[vid]),
                    completed_videos=status["completed_videos"] + 1,
                    updated_at=utc_now(),
                    invocation_elapsed_seconds=time.time() - started,
                    reused_video_features=reused_videos,
                    last_video_elapsed_seconds=time.time() - before,
                )
                atomic_json(run / "progress.json", status)
                if args.smoke or status["completed_videos"] % 50 == 0:
                    print(json.dumps(status), flush=True)
                del pixels, features
            except Exception as exc:
                atomic_json(
                    run / "failures" / f"{vid}_{int(time.time())}.json",
                    {"vid": vid, "error_type": type(exc).__name__, "error": str(exc), "time": utc_now()},
                )
                atomic_json(run / "progress.json", {**status, "status": "failed", "failed_vid": vid})
                raise
            next_vid = next(videos, None)
            if next_vid is not None:
                futures.append((next_vid, pool.submit(prepare_video, next_vid)))
    predictions = [json.loads((run / "queries" / f"{r['qid']}.json").read_text())["prediction"] for r in rows]
    write_jsonl(run / "predictions.jsonl", predictions)
    complete = {
        **status,
        "status": "complete",
        "finished_at": utc_now(),
        "single_frame_check_passed": single_frame_check if args.smoke else gate["single_frame_check_passed"],
        "invocation_elapsed_seconds": time.time() - started,
        "query_encoding_seconds": text_seconds,
        "new_visual_encoding_seconds": feature_seconds,
        "predictions_sha256": sha256(run / "predictions.jsonl"),
        "failures": 0,
    }
    atomic_json(run / "completion.json", complete)
    atomic_json(run / "progress.json", complete)
    print(json.dumps(complete), flush=True)


if __name__ == "__main__":
    main()
