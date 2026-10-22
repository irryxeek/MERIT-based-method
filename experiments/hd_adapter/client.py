"""Query-local ledgers, cross-worker request locks, and bounded crash recovery."""

import copy
import fcntl
import json
import time
import urllib.error
from contextlib import contextmanager
from pathlib import Path

from merit_qvh.client import CachedClient, redact_images
from merit_qvh.core import digest, parse_json, write_json


@contextmanager
def locked(path, *, blocking=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        yield


class Client(CachedClient):
    """Legacy payloads are unchanged; HD requests add system and JSON mode.

    A terminal failed response stays failed. Only a process-interrupted request
    can be recovered, at most twice, with all old attempts retained.
    """

    def complete(self, stage, content, max_tokens, validator=None, *, system=None):
        messages = [{"role": "user", "content": content}]
        if system is not None:
            messages.insert(0, {"role": "system", "content": system})
        payload = {
            "model": self.config["model"],
            "messages": messages,
            "temperature": self.config["temperature"],
            "max_tokens": max_tokens,
            "seed": 42,
        }
        if system is not None:
            payload["response_format"] = {"type": "json_object"}
        request_id = digest({"protocol": self.protocol_id, "stage": stage, "payload": payload})
        path = self.cache_root / request_id[:2] / (request_id + ".json")
        with locked(path.with_suffix(".lock")):
            return self._complete_locked(path, request_id, stage, payload, validator)

    def _complete_locked(self, path, request_id, stage, payload, validator):
        def parse(response):
            choice = response["choices"][0]
            if choice["finish_reason"] != "stop":
                raise ValueError(f"Incomplete generation: {choice['finish_reason']}")
            output = choice["message"]["content"]
            if not isinstance(output, str) or not output.strip():
                raise ValueError("Empty model response")
            return output, validator(parse_json(output)) if validator else output.strip()

        existed = path.exists()
        event = {"request_id": request_id, "cache_hit": existed, "stage": stage}
        self.events.append(event)
        if existed:
            record = json.loads(path.read_text())
            if record["status"] == "ok":
                return validator(parse_json(record["text"])) if validator else record["text"].strip()
            if record["status"] == "failed":
                raise RuntimeError(f"Previously failed {stage} request {request_id}: {record.get('error')}")
            # Recover a response persisted just before interruption without generating again.
            if record["attempts"] and "response" in record["attempts"][-1]:
                try:
                    output, result = parse(record["attempts"][-1]["response"])
                except (ValueError, TypeError, KeyError, IndexError):
                    pass
                else:
                    record.update(status="ok", text=output, recovered_response=True)
                    write_json(path, record)
                    return result
            record["interruptions"] = record.get("interruptions", 0) + 1
            if record["interruptions"] > 2:
                record.update(status="failed", error="Interrupted recovery budget exhausted")
                write_json(path, record)
                raise RuntimeError(record["error"])
            event.update(cache_hit=False, recovered_pending=True)
        else:
            record = {
                "request_id": request_id,
                "stage": stage,
                "status": "pending",
                "endpoint": self.endpoint,
                "payload": redact_images(payload),
                "attempts": [],
                "created_at": time.time(),
            }
        write_json(path, record)
        payload = copy.deepcopy(payload)
        last_error = None
        for format_attempt in range(self.config["format_retries"] + 1):
            response = None
            for transport_attempt in range(self.config["transport_retries"] + 1):
                started = time.monotonic()
                attempt = {
                    "format_attempt": format_attempt,
                    "transport_attempt": transport_attempt,
                    "started_at": time.time(),
                    "status": "in_flight",
                    "payload": redact_images(payload),
                }
                record["attempts"].append(attempt)
                write_json(path, record)
                try:
                    response = self._post(payload)
                    attempt.update(response=response, status="responded")
                except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as error:
                    last_error = error
                    attempt.update(error=str(error), status="transport_error")
                    if isinstance(error, urllib.error.HTTPError):
                        attempt["http_error_body"] = error.read(4000).decode(errors="replace")
                attempt["seconds"] = time.monotonic() - started
                write_json(path, record)
                if response is not None:
                    break
            if response is None:
                break
            try:
                output, result = parse(response)
                record.update(status="ok", text=output)
                write_json(path, record)
                return result
            except (ValueError, TypeError, KeyError, IndexError) as error:
                last_error = error
                attempt["validation_error"] = str(error)
                write_json(path, record)
                if format_attempt < self.config["format_retries"]:
                    payload["messages"].extend(
                        [
                            {
                                "role": "assistant",
                                "content": response.get("choices", [{}])[0]
                                .get("message", {})
                                .get("content", "")
                                or "",
                            },
                            {
                                "role": "user",
                                "content": "Regenerate the complete answer in the requested format. "
                                "Validation error: " + str(error),
                            },
                        ]
                    )
        record.update(status="failed", error=str(last_error))
        write_json(path, record)
        raise RuntimeError(f"{stage} failed: {last_error} (request {request_id})")
