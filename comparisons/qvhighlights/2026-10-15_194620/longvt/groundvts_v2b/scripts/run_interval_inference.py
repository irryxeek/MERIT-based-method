#!/usr/bin/env python3
"""Resumable LongVT-RFT QVHighlights interval-format inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

from openai import DefaultHttpxClient, OpenAI

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
ROOT = EXPERIMENT_DIR.parents[2]
sys.path.insert(0, str(ROOT / "reproduction/qvhighlights/scripts"))
sys.path.insert(0, str(SCRIPT_DIR))

from inference_common import (  # noqa: E402
    CROP_VIDEO_TOOL,
    GpuMemoryMonitor,
    encode_crop,
    encode_global_video,
    format_timestamps,
    image_content_sha256,
    response_record,
)
from interval_protocol_v2 import (  # noqa: E402
    append_tool_observation,
    clamp_crop_to_official_duration,
    completed_qids,
    eligible_frame_indices,
    interval_response_format,
    intervals_to_scores,
    parse_intervals,
)


EVIDENCE_PROMPT = (EXPERIMENT_DIR / "prompts/evidence.txt").read_text().strip()
FINAL_PROMPT = (EXPERIMENT_DIR / "prompts/finalization.txt").read_text().strip()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classify_error(exc: Exception) -> str:
    text = f"{type(exc).__name__}: {exc}".lower()
    if "out of memory" in text or "cuda" in text and "memory" in text:
        return "oom"
    if "timeout" in text or "timed out" in text:
        return "timeout"
    if "maximum model length" in text or "context length" in text:
        return "context_overflow"
    if any(word in text for word in ("video", "decord", "ffmpeg", "no such file")):
        return "video_read_failure"
    return "request_failure"


def chat(client, model, messages, config, *, tools=None, response_format=None, max_tokens):
    started = time.time()
    kwargs = dict(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=config["inference"]["temperature"],
        top_p=config["inference"]["top_p"],
        seed=config["inference"]["seed"],
        timeout=config["inference"]["request_timeout_seconds"],
    )
    if tools:
        kwargs.update({"tools": tools, "tool_choice": "auto"})
    if response_format is not None:
        kwargs["response_format"] = response_format
    response = client.chat.completions.create(**kwargs)
    record = response_record(response)
    record["latency_seconds"] = time.time() - started
    record["constrained_decoding"] = response_format is not None
    return response, record


def build_evidence_prompt(sample: dict, mode: str, timestamps: list[float], max_tool_calls: int) -> str:
    tool_instruction = (
        f"You may call crop_video at most {max_tool_calls} times to inspect useful local intervals. "
        "Continue reasoning after each crop and keep the complete timeline in view. "
        if mode == "tool" else "No tools are available; use the uniform global observations. "
    )
    return (
        f"Query: {sample['query']}\n"
        f"Video local duration: {sample['duration']:.3f} seconds. Video path: {sample['video_path']}\n"
        "The supplied global frames are chronological at local timestamps (seconds): "
        f"[{format_timestamps(timestamps)}].\n\n{tool_instruction}{EVIDENCE_PROMPT}"
    )


def build_final_prompt(sample: dict) -> str:
    return (
        f"Video local duration: {sample['duration']:.3f} seconds; official clip count: {sample['num_clips']}.\n"
        + FINAL_PROMPT
    )


def run_sample(client, model: str, sample: dict, mode: str, config: dict, artifact_dir: Path) -> dict:
    started = time.time()
    result: dict = {
        "qid": sample["qid"], "vid": sample["vid"], "query": sample["query"],
        "duration": sample["duration"], "num_clips": sample["num_clips"], "mode": mode,
        "status": None, "rounds": [], "tool_calls": [], "parse_attempts": [],
        "prediction_valid": False, "fallback_applied": False,
    }

    with GpuMemoryMonitor() as gpu_monitor:
        try:
            images, timestamps = encode_global_video(
                sample["video_path"], config["data"]["global_max_frames"], config["data"]["max_pixels"]
            )
            result.update({
                "global_frame_count": len(images),
                "global_frame_timestamps": timestamps,
                "global_frame_sha256": image_content_sha256(images),
            })
            messages = [
                {"role": "system", "content": EVIDENCE_PROMPT},
                {"role": "user", "content": images + [{
                    "type": "text", "text": build_evidence_prompt(
                        sample, mode, timestamps, config["inference"]["max_tool_calls"]
                    ),
                }]},
            ]
            calls_used = 0
            analysis_text = ""
            while True:
                available_tools = [CROP_VIDEO_TOOL] if mode == "tool" and calls_used < config["inference"]["max_tool_calls"] else None
                response, record = chat(
                    client, model, messages, config, tools=available_tools,
                    max_tokens=config["output_protocol"]["evidence_max_tokens"],
                )
                message = response.choices[0].message
                record["stage"] = "evidence"
                result["rounds"].append(record)
                if record["finish_reason"] != "tool_calls" or not message.tool_calls:
                    analysis_text = message.content or ""
                    messages.append({"role": "assistant", "content": analysis_text})
                    break

                messages.append({
                    "role": "assistant", "content": message.content or "",
                    "tool_calls": [{
                        "id": call.id, "type": "function",
                        "function": {"name": call.function.name, "arguments": call.function.arguments},
                    } for call in message.tool_calls],
                })
                for call in message.tool_calls:
                    tool_record = {
                        "id": call.id, "name": call.function.name,
                        "arguments_raw": call.function.arguments,
                        "reasoning": message.content or "", "executed": False, "success": False,
                    }
                    result["tool_calls"].append(tool_record)
                    try:
                        if calls_used >= config["inference"]["max_tool_calls"]:
                            tool_record["budget_rejected"] = True
                            raise RuntimeError("tool_budget_exhausted")
                        calls_used += 1
                        arguments = json.loads(call.function.arguments)
                        tool_record["arguments"] = arguments
                        if call.function.name != "crop_video":
                            raise ValueError(f"unknown_tool:{call.function.name}")
                        if Path(arguments["video_path"]).resolve() != Path(sample["video_path"]).resolve():
                            raise ValueError("tool_video_path_mismatch")
                        requested_start = float(arguments["start_time"])
                        requested_end = float(arguments["end_time"])
                        official_duration = float(sample["duration"])
                        requested_start, executed_end, end_clamped = clamp_crop_to_official_duration(
                            requested_start, requested_end, official_duration
                        )
                        crop_images, crop_timestamps, execution = encode_crop(
                            sample["video_path"], requested_start, executed_end,
                            config["data"]["local_max_frames"], config["data"]["max_pixels"],
                        )
                        eligible_frames = eligible_frame_indices(crop_timestamps, official_duration)
                        crop_images = [crop_images[i] for i in eligible_frames]
                        crop_timestamps = [crop_timestamps[i] for i in eligible_frames]
                        if not crop_images:
                            raise ValueError("crop_has_no_frames_inside_official_local_duration")
                        tool_record.update({
                            "executed": True, "success": True, "execution": execution,
                            "requested_end_exceeded_official_duration": end_clamped,
                            "official_duration_clip_seconds": official_duration,
                            "returned_frame_count": len(crop_images),
                            "frame_timestamps": crop_timestamps,
                            "returned_frame_sha256": image_content_sha256(crop_images),
                            "returned_frames_added_to_next_request": True,
                        })
                        tool_content = crop_images + [{"type": "text", "text": (
                            f"Observed local interval {execution['executed_start_time']:.3f}-"
                            f"{execution['executed_end_time']:.3f} seconds with {len(crop_images)} chronological frames. "
                            f"Their local timestamps are [{format_timestamps(crop_timestamps)}]. "
                            "Continue assessing the complete timeline."
                        )}]
                    except Exception as exc:
                        tool_record.update({"executed": True, "error": repr(exc), "traceback": traceback.format_exc()})
                        tool_content = [{"type": "text", "text": f"Tool error: {exc}"}]
                    append_tool_observation(messages, call.id, tool_content)

            result["analysis_response"] = analysis_text
            result["analysis_prompt_sha256"] = hashlib.sha256(build_evidence_prompt(
                sample, mode, timestamps, config["inference"]["max_tool_calls"]
            ).encode()).hexdigest()
            messages.append({"role": "user", "content": build_final_prompt(sample)})
            response_format = interval_response_format(sample["duration"])
            response, record = chat(
                client, model, messages, config, tools=None, response_format=response_format,
                max_tokens=config["output_protocol"]["final_max_tokens"],
            )
            record["stage"] = "final_interval_json"
            result["rounds"].append(record)
            final_text = response.choices[0].message.content or ""
            messages.append({"role": "assistant", "content": final_text})

            segments = None
            for attempt in range(config["output_protocol"]["format_retries"] + 1):
                try:
                    segments, order_canonicalized = parse_intervals(final_text, sample["duration"], sample["num_clips"])
                    result["parse_attempts"].append({
                        "attempt": attempt, "success": True,
                        "order_canonicalized": order_canonicalized,
                    })
                    break
                except Exception as exc:
                    result["parse_attempts"].append({
                        "attempt": attempt, "success": False, "error": str(exc), "raw_response": final_text,
                    })
                    if attempt >= config["output_protocol"]["format_retries"]:
                        break
                    messages.append({"role": "user", "content": (
                        "Format correction only. Return one JSON object with exactly the key `segments`; each item "
                        "must have exactly string `clip_span` and one allowed string `level`. `clip_span` must be "
                        "a schema-permitted start:end range of official two-second clip indices, end exclusive. "
                        "Use levels very_important/important/less_important. Do not change the evidence judgment, "
                        "call tools, add explanation, or use markdown."
                    )})
                    response, retry_record = chat(
                        client, model, messages, config, tools=None, response_format=response_format,
                        max_tokens=config["output_protocol"]["final_max_tokens"],
                    )
                    retry_record.update({"stage": "format_correction", "format_retry": True})
                    result["rounds"].append(retry_record)
                    final_text = response.choices[0].message.content or ""
                    messages.append({"role": "assistant", "content": final_text})

            result["raw_final_response"] = final_text
            if segments is None:
                result.update({
                    "status": "format_parse_failure", "fallback_applied": True,
                    "pred_saliency_scores": [0.0] * sample["num_clips"], "predicted_segments": [],
                })
            else:
                scores = intervals_to_scores(
                    segments, sample["duration"], sample["num_clips"], config["data"]["clip_seconds"]
                )
                failed_tools = [call for call in result["tool_calls"] if not call.get("success")]
                result.update({
                    "status": "success_with_tool_failure" if failed_tools else "success",
                    "prediction_valid": True, "predicted_segments": segments,
                    "pred_saliency_scores": scores,
                    "fallback_applied": False,
                })
        except Exception as exc:
            result.update({
                "status": classify_error(exc), "error": repr(exc), "traceback": traceback.format_exc(),
                "fallback_applied": True, "pred_saliency_scores": [0.0] * sample["num_clips"],
                "predicted_segments": [],
            })

    result["peak_gpu_memory_mib"] = gpu_monitor.maximum
    result["local_frame_count"] = sum(call.get("returned_frame_count", 0) for call in result["tool_calls"])
    result["total_read_frame_count"] = result.get("global_frame_count", 0) + result["local_frame_count"]
    result["elapsed_seconds"] = time.time() - started
    return result


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", default=str(EXPERIMENT_DIR / "configs/protocol.json"))
    parser.add_argument("--mode", choices=["global", "tool"], required=True)
    parser.add_argument("--api-base", default="http://127.0.0.1:18007/v1")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reset-infra-retry-budget", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    manifest_path = Path(args.manifest).resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text())
    samples = json.loads(manifest_path.read_text())
    if not samples or len({sample["qid"] for sample in samples}) != len(samples):
        raise ValueError("Manifest must contain unique qids")
    if args.mode == "global" and config["inference"]["max_tool_calls"] < 0:
        raise ValueError("Invalid tool budget")

    client = OpenAI(
        api_key="EMPTY", base_url=args.api_base,
        timeout=config["inference"]["request_timeout_seconds"],
        http_client=DefaultHttpxClient(trust_env=False),
    )
    model = client.models.list().data[0].id
    prompt_hashes = {
        "evidence": hashlib.sha256(EVIDENCE_PROMPT.encode()).hexdigest(),
        "finalization": hashlib.sha256(FINAL_PROMPT.encode()).hexdigest(),
    }
    code_paths = [Path(__file__), SCRIPT_DIR / "interval_protocol_v2.py", ROOT / "reproduction/qvhighlights/scripts/inference_common.py"]
    snapshot = {
        "experiment_id": config["experiment_id"], "mode": args.mode, "model": model,
        "model_revision": config["model"]["revision"], "api_base": args.api_base,
        "manifest_path": str(manifest_path), "manifest_sha256": sha256(manifest_path),
        "config_path": str(config_path), "config_sha256": sha256(config_path),
        "prompt_sha256": prompt_hashes,
        "source_sha256": {str(path): sha256(path) for path in code_paths},
    }
    snapshot_path = output_dir / "config_snapshot.json"
    if snapshot_path.exists():
        previous = json.loads(snapshot_path.read_text())
        keys = ("experiment_id", "mode", "model", "model_revision", "manifest_sha256", "config_sha256", "prompt_sha256", "source_sha256")
        changed = [key for key in keys if previous.get(key) != snapshot.get(key)]
        if changed:
            raise RuntimeError(f"Refusing incompatible resume; changed {changed}")
    else:
        snapshot_path.write_text(json.dumps(snapshot, indent=2) + "\n")
        (output_dir / "manifest_snapshot.json").write_bytes(manifest_path.read_bytes())

    client_meta = {
        "model": model,
        "models_endpoint": [row.model_dump() for row in client.models.list().data],
    }
    (output_dir / "service_identity.json").write_text(json.dumps(client_meta, indent=2) + "\n")

    attempts_path = output_dir / "attempts.jsonl"
    traces_path = output_dir / "traces.jsonl"
    attempts = read_jsonl(attempts_path)
    trace_rows = read_jsonl(traces_path)
    latest = {row["qid"]: row for row in trace_rows}
    done_qids = completed_qids(trace_rows)
    infra_count: dict = {}
    for row in attempts:
        if row.get("status") in {"timeout", "request_failure", "oom", "context_overflow"}:
            infra_count[row["qid"]] = infra_count.get(row["qid"], 0) + 1
    if args.reset_infra_retry_budget:
        infra_count = {}

    transient = {"timeout", "request_failure", "oom", "context_overflow"}
    max_total_attempts = 1 + config["inference"]["infra_retries"]
    pending = []
    for sample in samples:
        qid = sample["qid"]
        if qid in done_qids:
            continue
        if infra_count.get(qid, 0) >= max_total_attempts:
            pending.append(qid)
            continue
        result = None
        while infra_count.get(qid, 0) < max_total_attempts:
            attempt_number = infra_count.get(qid, 0) + 1
            result = run_sample(client, model, sample, args.mode, config, output_dir / "artifacts")
            result["attempt_number"] = attempt_number
            result["protocol_sha256"] = snapshot["config_sha256"]
            append_jsonl(attempts_path, result)
            if result["status"] not in transient:
                break
            infra_count[qid] = attempt_number
            if infra_count[qid] < max_total_attempts:
                time.sleep(min(2 ** (attempt_number - 1), 8))
        if result is None or result["status"] in transient:
            pending.append(qid)
            continue
        append_jsonl(traces_path, result)
        latest[qid] = result
        done_qids.add(qid)
        print(json.dumps({
            "qid": qid, "mode": args.mode, "status": result["status"],
            "tool_calls": len(result["tool_calls"]), "elapsed_seconds": round(result["elapsed_seconds"], 2),
        }), flush=True)

    if pending:
        (output_dir / "pending_qids.json").write_text(json.dumps(pending, indent=2) + "\n")
        print(json.dumps({"pending": len(pending), "first_qids": pending[:10]}), flush=True)
        return

    if len(latest) != len(samples):
        raise RuntimeError(f"Trace coverage incomplete: {len(latest)}/{len(samples)}")
    predictions = output_dir / "predictions.jsonl"
    with predictions.open("w") as handle:
        for sample in samples:
            trace = latest[sample["qid"]]
            handle.write(json.dumps({
                "qid": sample["qid"], "vid": sample["vid"], "query": sample["query"],
                "pred_saliency_scores": trace["pred_saliency_scores"],
            }) + "\n")
    (output_dir / "failures.json").write_text(json.dumps([
        row for row in latest.values() if row["status"] not in {"success", "success_with_tool_failure"}
    ], indent=2) + "\n")
    print(json.dumps({"completed": len(samples), "predictions": str(predictions)}), flush=True)


if __name__ == "__main__":
    main()
