"""Bounded, cached requests to a user-provided local Qwen endpoint."""

import base64
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from .core import digest, parse_json, write_json


def image_part(path):
    encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + encoded}}


def text_part(text):
    return {"type": "text", "text": text}


def redact_images(value):
    if isinstance(value, dict):
        return {k: redact_images(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_images(v) for v in value]
    if isinstance(value, str) and value.startswith("data:image/"):
        return {"image_data_digest": digest(value), "encoded_length": len(value)}
    return value


class CachedClient:
    def __init__(self, endpoint, config, cache_root, protocol_id):
        self.endpoint = endpoint.rstrip("/")
        self.config = config
        self.cache_root = Path(cache_root)
        self.protocol_id = protocol_id
        self.events = []

    def check_model(self):
        with urllib.request.urlopen(self.endpoint + "/models", timeout=15) as response:
            models = json.load(response)["data"]
        matches = [model for model in models if model["id"] == self.config["model"]]
        if len(matches) != 1:
            raise RuntimeError("Endpoint does not serve the configured model")
        root = matches[0].get("root")
        if root and root != self.config["model"] and root != self.config["weight_path"]:
            raise RuntimeError("Endpoint weight path differs from the frozen configuration")
        return matches[0]

    def _post(self, payload):
        request = urllib.request.Request(
            self.endpoint + "/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.config["timeout_seconds"]) as response:
            return json.load(response)

    def complete(self, stage, content, max_tokens, validator=None):
        payload = {
            "model": self.config["model"],
            "messages": [{"role": "user", "content": content}],
            "temperature": self.config["temperature"],
            "max_tokens": max_tokens,
            "seed": 42,
        }
        request_id = digest({"protocol": self.protocol_id, "stage": stage, "payload": payload})
        path = self.cache_root / request_id[:2] / (request_id + ".json")
        if path.exists():
            record = json.loads(path.read_text())
            self.events.append({"request_id": request_id, "cache_hit": True, "stage": stage})
            if record["status"] != "ok":
                raise RuntimeError(f"Previously failed/incomplete request {request_id}; inspect {path}")
            return validator(parse_json(record["text"])) if validator else record["text"]

        record = {
            "request_id": request_id,
            "stage": stage,
            "status": "pending",
            "endpoint": self.endpoint,
            "payload": redact_images(payload),
            "attempts": [],
        }
        write_json(path, record)
        last_error = None
        for format_attempt in range(self.config["format_retries"] + 1):
            response = None
            for transport_attempt in range(self.config["transport_retries"] + 1):
                started = time.monotonic()
                attempt = {"format_attempt": format_attempt, "transport_attempt": transport_attempt}
                try:
                    response = self._post(payload)
                    attempt["response"] = response
                except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as error:
                    last_error = error
                    attempt["error"] = str(error)
                    if isinstance(error, urllib.error.HTTPError):
                        attempt["http_error_body"] = error.read(4000).decode(errors="replace")
                attempt["seconds"] = time.monotonic() - started
                record["attempts"].append(attempt)
                write_json(path, record)
                if response is not None:
                    break
            if response is None:
                break
            try:
                choice = response["choices"][0]
                if choice["finish_reason"] != "stop":
                    raise ValueError(f"Incomplete generation: {choice['finish_reason']}")
                output = choice["message"]["content"]
                if not isinstance(output, str) or not output.strip():
                    raise ValueError("Empty model response")
                result = validator(parse_json(output)) if validator else output.strip()
                record.update(status="ok", text=output)
                write_json(path, record)
                self.events.append({"request_id": request_id, "cache_hit": False, "stage": stage})
                return result
            except (ValueError, TypeError, KeyError, IndexError) as error:
                last_error = error
                record["attempts"][-1]["validation_error"] = str(error)
                # A fresh generation, bounded to one format repair; no score coercion.
                if format_attempt < self.config["format_retries"]:
                    payload["messages"].append(
                        {
                            "role": "assistant",
                            "content": response.get("choices", [{}])[0].get("message", {}).get("content", "")
                            or "",
                        }
                    )
                    payload["messages"].append(
                        {
                            "role": "user",
                            "content": "Regenerate the complete answer in the requested format. "
                            "Validation error: " + str(error),
                        }
                    )
        record.update(status="failed", error=str(last_error))
        write_json(path, record)
        self.events.append({"request_id": request_id, "cache_hit": False, "stage": stage})
        raise RuntimeError(f"{stage} failed: {last_error} (request {request_id})")
