#!/usr/bin/env python3
"""Probe whether a running server enforces the v2b 75-clip regex schema."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from openai import DefaultHttpxClient, OpenAI

from interval_protocol_v2 import interval_response_format, parse_intervals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", default="http://127.0.0.1:18008/v1")
    parser.add_argument("--model", default="longvideotool/LongVT-RFT")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    response_format = interval_response_format(150.0)
    schema_json = json.dumps(response_format, sort_keys=True, separators=(",", ":"))
    messages = [
        {"role": "system", "content": "Return the requested JSON exactly and follow the supplied schema."},
        {"role": "user", "content": (
            'Return exactly one segment with level "important" and clip_span "0:150". '
            'Do not change the requested span.'
        )},
    ]
    record = {
        "api_base": args.api_base,
        "model": args.model,
        "duration_seconds": 150.0,
        "num_clips": 75,
        "requested_invalid_span": "0:150",
        "response_format_sha256": hashlib.sha256(schema_json.encode()).hexdigest(),
        "schema_pattern_chars": len(
            response_format["json_schema"]["schema"]["properties"]["segments"]["items"]["properties"]["clip_span"]["pattern"]
        ),
        "started_unix_time": time.time(),
    }
    started = time.monotonic()
    try:
        client = OpenAI(
            base_url=args.api_base,
            api_key="EMPTY",
            timeout=120,
            http_client=DefaultHttpxClient(trust_env=False),
        )
        result = client.chat.completions.create(
            model=args.model,
            messages=messages,
            response_format=response_format,
            temperature=0,
            top_p=1,
            seed=42,
            max_tokens=128,
        )
        text = result.choices[0].message.content or ""
        segments, canonicalized = parse_intervals(text, 150.0, 75)
        record.update({
            "success": True,
            "raw_content": text,
            "parsed_segments": segments,
            "order_canonicalized": canonicalized,
            "invalid_span_rejected": len(segments) == 1 and all(item["clip_span"] != "0:150" for item in segments),
            "finish_reason": result.choices[0].finish_reason,
            "usage": result.usage.model_dump() if result.usage else None,
            "response_id": result.id,
        })
    except Exception as exc:  # preserve service/schema errors as probe evidence
        record.update({"success": False, "error_type": type(exc).__name__, "error": str(exc)})
    record["elapsed_seconds"] = time.monotonic() - started
    record["completed_unix_time"] = time.time()
    record["passed"] = bool(record.get("success") and record.get("invalid_span_rejected"))
    output_path = Path(args.output)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite probe record: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    if not record["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
