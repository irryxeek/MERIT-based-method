"""Bounded structured memory requests with replayable field-repair ledgers."""

import copy
import json
import time
import urllib.error

from hd_adapter.client import Client, locked

from merit_qvh.client import redact_images
from merit_qvh.core import digest, write_json


class MemoryClient(Client):
    """E1 uses the unmodified legacy client; only memory uses this contract."""

    def complete_memory(self, stage, content, max_tokens, contract, *, system):
        initial = {
            "model": self.config["model"],
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
            "temperature": self.config["temperature"],
            "max_tokens": max_tokens,
            "seed": 42,
            "response_format": self.response_format(contract.schema),
        }
        rid = digest({"protocol": self.protocol_id, "stage": stage, "payload": initial})
        path = self.cache_root / rid[:2] / (rid + ".json")
        with locked(path.with_suffix(".lock")):
            self.events.append({"request_id": rid, "cache_hit": path.exists(), "stage": stage})
            if path.exists():
                record = json.loads(path.read_text())
                if record["status"] == "ok":
                    return copy.deepcopy(record["result"])
                if record["status"] == "failed":
                    raise RuntimeError(f"Previously failed {stage} request {rid}: {record['error']}")
            else:
                record = {
                    "request_id": rid,
                    "stage": stage,
                    "status": "pending",
                    "endpoint": self.endpoint,
                    "payload": redact_images(initial),
                    "attempts": [],
                    "created_at": time.time(),
                }
            write_json(path, record)
            candidate, issues, last_error = None, [], "No valid response"
            # Replay persisted responses deterministically. An unknown in-flight
            # attempt consumes its transport slot; a restart never resets budgets.
            for fmt in range(self.config["format_retries"] + 1):
                payload = copy.deepcopy(initial)
                patching = candidate is not None
                if patching:
                    schema, prompt = contract.repair_spec(candidate, issues)
                    payload["response_format"] = self.response_format(schema)
                    payload["messages"].append({"role": "user", "content": prompt})
                elif fmt:
                    payload["messages"].append(
                        {
                            "role": "user",
                            "content": "Previous generation was incomplete. Return compact JSON.",
                        }
                    )
                response = None
                for transport in range(self.config["transport_retries"] + 1):
                    prior = [
                        a
                        for a in record["attempts"]
                        if a["format_attempt"] == fmt and a["transport_attempt"] == transport
                    ]
                    if prior:
                        attempt = prior[0]
                        if attempt["payload"] != redact_images(payload):
                            raise RuntimeError("Recovery payload differs from persisted attempt")
                        if "response" in attempt:
                            response = attempt["response"]
                            break
                        if attempt["status"] == "in_flight":
                            attempt.update(
                                status="interrupted_unknown", error="Outcome unknown after interruption"
                            )
                            write_json(path, record)
                        last_error = attempt.get("error", "Interrupted request")
                        continue
                    attempt = {
                        "format_attempt": fmt,
                        "transport_attempt": transport,
                        "mode": "field_patch" if patching else "initial",
                        "started_at": time.time(),
                        "status": "in_flight",
                        "payload": redact_images(payload),
                    }
                    record["attempts"].append(attempt)
                    write_json(path, record)
                    tick = time.monotonic()
                    try:
                        response = self._post(payload)
                        attempt.update(response=response, status="responded")
                    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as error:
                        last_error = str(error)
                        attempt.update(error=last_error, status="transport_error")
                        if isinstance(error, urllib.error.HTTPError):
                            attempt["http_error_body"] = error.read(4000).decode(errors="replace")
                    attempt["seconds"] = time.monotonic() - tick
                    write_json(path, record)
                    if response is not None:
                        break
                if response is None:
                    break
                try:
                    choice = response["choices"][0]
                    if choice["finish_reason"] != "stop":
                        raise ValueError(f"Incomplete generation: {choice['finish_reason']}")
                    decoded = json.loads(choice["message"]["content"])
                    candidate = contract.apply_patch(candidate, decoded, issues) if patching else decoded
                    issues = contract.problems(candidate)
                    if issues:
                        raise ValueError(json.dumps(issues))
                    result = contract.finalize(candidate)
                except (ValueError, TypeError, KeyError, IndexError) as error:
                    last_error = str(error)
                    attempt["validation_error"] = last_error
                    write_json(path, record)
                else:
                    return self.save_result(path, record, result, candidate)
            if candidate is not None:
                try:
                    result = contract.finalize(candidate, degrade=True)
                except (ValueError, TypeError, KeyError) as error:
                    last_error = str(error)
                else:
                    return self.save_result(path, record, result, candidate)
            record.update(status="failed", error=last_error)
            write_json(path, record)
            raise RuntimeError(f"{stage} failed: {last_error} (request {rid})")

    @staticmethod
    def response_format(schema):
        return {"type": "json_schema", "json_schema": {"name": "memory", "strict": True, "schema": schema}}

    @staticmethod
    def save_result(path, record, result, candidate):
        result["request_id"] = record["request_id"]
        result["repair_attempted"] = any(a["format_attempt"] > 0 for a in record["attempts"])
        result["quality"] = (
            "degraded" if result["downgraded_cues"] or result["rejected_links"] else "validated"
        )
        record.update(status="ok", result=result, text=json.dumps(candidate), finished_at=time.time())
        write_json(path, record)
        return copy.deepcopy(result)
