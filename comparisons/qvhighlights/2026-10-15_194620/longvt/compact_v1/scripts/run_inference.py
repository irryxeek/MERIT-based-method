#!/usr/bin/env python3
"""Run label-free LongVT global or tool QVHighlights inference."""

import argparse
import hashlib
import json
import shutil
import time
import traceback
from pathlib import Path

from inference_common import (
    CROP_VIDEO_TOOL,
    SYSTEM_PROMPT,
    GpuMemoryMonitor,
    build_finalization_prompt,
    build_prompt,
    encode_crop,
    encode_global_video,
    format_timestamps,
    image_content_sha256,
    parse_scores,
    response_record,
    save_image_contents,
)
from openai import DefaultHttpxClient, OpenAI


def classify_error(exc: Exception) -> str:
    text = str(exc).lower()
    if "out of memory" in text:
        return "oom"
    if "timeout" in text or "timed out" in text:
        return "timeout"
    if "maximum model length" in text:
        return "context_overflow"
    if any(word in text for word in ("video", "decord", "ffmpeg")):
        return "video_read_failure"
    return "request_failure"


def score_response_format(num_clips: int, duration: float, max_override_segments: int = 5) -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "qvhighlights_scores",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "background_score": {"type": "integer", "minimum": 0, "maximum": 4},
                    "segments": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "start_time": {"type": "number", "minimum": 0, "maximum": duration},
                                "end_time": {"type": "number", "minimum": 0, "maximum": duration},
                                "score": {"type": "integer", "minimum": 0, "maximum": 4},
                            },
                            "required": ["start_time", "end_time", "score"],
                            "additionalProperties": False,
                        },
                        "maxItems": max_override_segments,
                    },
                },
                "required": ["background_score", "segments"],
                "additionalProperties": False,
            },
        },
    }


def complete(client, model, messages, config, tools=None, response_format=None, max_tokens=None):
    kwargs = {
        "model": model, "messages": messages, "max_tokens": max_tokens or config["max_output_tokens"],
        "temperature": config["temperature"], "top_p": config["top_p"], "seed": config["seed"],
        "timeout": config["request_timeout_seconds"],
    }
    if tools:
        kwargs.update({"tools": tools, "tool_choice": "auto"})
    if response_format:
        kwargs["response_format"] = response_format
    started = time.time()
    response = client.chat.completions.create(**kwargs)
    record = response_record(response)
    record["constrained_decoding"] = response_format is not None
    record["latency_seconds"] = time.time() - started
    return response, record


def run_sample(client, model, sample, mode, config, artifacts_dir: Path):
    started = time.time()
    result = {
        "qid": sample["qid"], "vid": sample["vid"], "query": sample["query"],
        "duration": sample["duration"], "num_clips": sample["num_clips"], "mode": mode,
        "status": None, "rounds": [], "tool_calls": [], "parse_attempts": [],
        "prediction_valid": False, "fallback_applied": False,
    }
    with GpuMemoryMonitor() as monitor:
        try:
            encode_start = time.time()
            images, timestamps = encode_global_video(
                sample["video_path"], config["global_max_frames"], config["max_pixels"]
            )
            result.update({
                "global_frame_count": len(images), "global_frame_timestamps": timestamps,
                "global_frame_sha256": image_content_sha256(images),
                "global_encode_seconds": time.time() - encode_start,
            })
            prompt_text = build_prompt(sample, mode == "tool", timestamps, config["max_tool_calls"])
            result["prompt_text"] = prompt_text
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": images + [{
                    "type": "text", "text": prompt_text,
                }]},
            ]
            analysis_text = ""
            calls_used = 0
            while True:
                tools = [CROP_VIDEO_TOOL] if mode == "tool" and calls_used < config["max_tool_calls"] else None
                response, record = complete(
                    client, model, messages, config, tools, response_format=None,
                    max_tokens=config["analysis_max_output_tokens"],
                )
                message = response.choices[0].message
                record["stage"] = "evidence_analysis"
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
                        "arguments_raw": call.function.arguments, "reasoning": message.content or "",
                        "executed": False, "success": False,
                    }
                    result["tool_calls"].append(tool_record)
                    try:
                        if calls_used >= config["max_tool_calls"]:
                            tool_record["budget_rejected"] = True
                            raise RuntimeError("tool_budget_exhausted")
                        calls_used += 1
                        arguments = json.loads(call.function.arguments)
                        tool_record["arguments"] = arguments
                        if call.function.name != "crop_video":
                            raise ValueError(f"unknown_tool_{call.function.name}")
                        if Path(arguments["video_path"]).resolve() != Path(sample["video_path"]).resolve():
                            raise ValueError("tool_video_path_mismatch")
                        crop_started = time.time()
                        crop_images, crop_timestamps, execution = encode_crop(
                            sample["video_path"], float(arguments["start_time"]), float(arguments["end_time"]),
                            config["local_max_frames"], config["max_pixels"],
                        )
                        tool_record.update({
                            "executed": True, "success": True, "execution": execution,
                            "returned_frame_count": len(crop_images), "frame_timestamps": crop_timestamps,
                            "returned_frame_sha256": image_content_sha256(crop_images),
                            "latency_seconds": time.time() - crop_started,
                        })
                        if config["save_local_frames"]:
                            call_dir = artifacts_dir / str(sample["qid"]) / f"tool_{len(result['tool_calls']):02d}"
                            tool_record["returned_frame_paths"] = save_image_contents(crop_images, call_dir)
                        content = crop_images + [{"type": "text", "text": (
                            f"Observed local interval {execution['executed_start_time']:.3f}-"
                            f"{execution['executed_end_time']:.3f} seconds with {len(crop_images)} frames. "
                            f"The frames are chronological at local timestamps (seconds): "
                            f"[{format_timestamps(crop_timestamps)}]. "
                            "Continue assessing the complete timeline."
                        )}]
                        tool_record["returned_frames_added_to_next_request"] = True
                    except Exception as exc:
                        tool_record.update({"executed": True, "error": repr(exc), "traceback": traceback.format_exc()})
                        content = [{"type": "text", "text": f"Tool error: {exc}"}]
                    messages.append({"role": "tool", "tool_call_id": call.id, "content": content})

            result["analysis_response"] = analysis_text
            messages.append({
                "role": "user",
                "content": build_finalization_prompt(sample, config["max_override_segments"]),
            })
            response, final_record = complete(
                client, model, messages, config, tools=None,
                response_format=None, max_tokens=config["score_max_output_tokens"],
            )
            final_record["stage"] = "score_finalization"
            result["rounds"].append(final_record)
            final_text = response.choices[0].message.content or ""
            messages.append({"role": "assistant", "content": final_text})

            scores = None
            for attempt in range(config["format_retries"] + 1):
                cleanup_actions = []
                try:
                    scores = parse_scores(
                        final_text, sample["num_clips"], config["score_min"], config["score_max"],
                        config["max_override_segments"], sample["duration"], config["clip_seconds"],
                        cleanup_actions,
                    )
                    result["parse_attempts"].append({
                        "attempt": attempt, "success": True, "cleanup_actions": cleanup_actions,
                    })
                    break
                except Exception as exc:
                    result["parse_attempts"].append({
                        "attempt": attempt, "success": False, "error": str(exc), "raw_response": final_text,
                        "cleanup_actions": cleanup_actions,
                    })
                    if attempt >= config["format_retries"]:
                        break
                    messages.append({"role": "user", "content": (
                        f"Format correction only for {sample['num_clips']} clips: return exactly "
                        '"background_score" and "segments". Each segment must contain exactly numeric '
                        '"start_time", "end_time", and integer "score" fields, using valid local seconds within '
                        f"[0,{sample['duration']:.3f}]. Use no more than {config['max_override_segments']} "
                        "segments. Do not add prose, call tools, or change the content judgment."
                    )})
                    response, retry_record = complete(
                        client, model, messages, config, tools=None,
                        response_format=score_response_format(
                            sample["num_clips"], sample["duration"], config["max_override_segments"]
                        ),
                        max_tokens=config["score_max_output_tokens"],
                    )
                    retry_record["is_format_retry"] = True
                    retry_record["stage"] = "format_retry"
                    result["rounds"].append(retry_record)
                    final_text = response.choices[0].message.content or ""
                    messages.append({"role": "assistant", "content": final_text})
            result["raw_final_response"] = final_text
            if scores is None:
                result.update({
                    "status": "format_parse_failure", "fallback_applied": True,
                    "pred_saliency_scores": [0.0] * sample["num_clips"],
                })
            elif any(not call.get("success") for call in result["tool_calls"]):
                result.update({
                    "status": "success_with_tool_failure", "prediction_valid": True,
                    "pred_saliency_scores": scores,
                })
            else:
                result.update({"status": "success", "prediction_valid": True, "pred_saliency_scores": scores})
        except Exception as exc:
            result.update({
                "status": classify_error(exc), "error": repr(exc), "traceback": traceback.format_exc(),
                "fallback_applied": True, "pred_saliency_scores": [0.0] * sample["num_clips"],
            })
    result["peak_gpu_memory_mib"] = monitor.maximum
    result["local_frame_count"] = sum(call.get("returned_frame_count", 0) for call in result["tool_calls"])
    result["total_read_frame_count"] = result.get("global_frame_count", 0) + result["local_frame_count"]
    result["repeated_local_frames_counted"] = True
    result["elapsed_seconds"] = time.time() - started
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=["global", "tool"], required=True)
    parser.add_argument("--api-base", default="http://127.0.0.1:18006/v1")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if config.get("failure_fallback") != "all_zero":
        raise ValueError("Only the predeclared all_zero failure fallback is supported")
    samples = json.loads(Path(args.manifest).read_text())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    traces_path = output_dir / "traces.jsonl"
    completed = set()
    if traces_path.exists():
        completed = {json.loads(line)["qid"] for line in traces_path.read_text().splitlines() if line.strip()}
    client = OpenAI(
        api_key="EMPTY",
        base_url=args.api_base,
        timeout=config["request_timeout_seconds"],
        http_client=DefaultHttpxClient(trust_env=False),
    )
    model = client.models.list().data[0].id
    manifest_path = Path(args.manifest).resolve()
    config_path = Path(args.config).resolve()
    source_paths = [Path(__file__).resolve(), Path(__file__).with_name("inference_common.py").resolve()]
    snapshot = {
        **config, "mode": args.mode, "model": model, "api_base": args.api_base,
        "manifest_path": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
    }
    snapshot_path = output_dir / "config_snapshot.json"
    if snapshot_path.exists():
        previous = json.loads(snapshot_path.read_text())
        comparison_keys = (
            "mode", "model", "manifest_sha256", "config_sha256", "source_sha256",
        )
        changed = [key for key in comparison_keys if previous.get(key) != snapshot.get(key)]
        if changed:
            raise RuntimeError(
                f"Refusing to mix incompatible resume state in {output_dir}; changed: {', '.join(changed)}"
            )
    else:
        snapshot_path.write_text(json.dumps(snapshot, indent=2) + "\n")
        shutil.copyfile(manifest_path, output_dir / "manifest_snapshot.json")
    for sample in samples:
        if sample["qid"] in completed:
            continue
        result = run_sample(client, model, sample, args.mode, config, output_dir / "local_frames")
        with traces_path.open("a") as handle:
            handle.write(json.dumps(result) + "\n")
        print(json.dumps({
            "qid": result["qid"], "status": result["status"],
            "tools": len(result["tool_calls"]), "seconds": round(result["elapsed_seconds"], 2),
        }), flush=True)
    traces = [json.loads(line) for line in traces_path.read_text().splitlines() if line.strip()]
    trace_qids = [row["qid"] for row in traces]
    manifest_qids = [row["qid"] for row in samples]
    if len(trace_qids) != len(set(trace_qids)) or set(trace_qids) != set(manifest_qids):
        raise RuntimeError("Trace qids are duplicated or differ from the frozen manifest")
    by_qid = {x["qid"]: x for x in traces}
    with (output_dir / "predictions.jsonl").open("w") as handle:
        for sample in samples:
            trace = by_qid[sample["qid"]]
            handle.write(json.dumps({
                "qid": sample["qid"], "vid": sample["vid"], "query": sample["query"],
                "pred_saliency_scores": trace["pred_saliency_scores"],
            }) + "\n")
    failures = [x for x in traces if x["status"] != "success"]
    (output_dir / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")


if __name__ == "__main__":
    main()
