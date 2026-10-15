from __future__ import annotations

import ipaddress
from urllib.parse import urlsplit

import requests


class LocalBackend:
    def __init__(self, config):
        self.config = config
        endpoint = urlsplit(config["base_url"])
        if endpoint.scheme != "http" or endpoint.hostname not in {"127.0.0.1", "::1"}:
            raise ValueError("only a literal loopback HTTP backend is allowed")
        if (
            endpoint.query
            or endpoint.fragment
            or endpoint.username
            or endpoint.password
        ):
            raise ValueError("invalid local URL")
        if not ipaddress.ip_address(endpoint.hostname).is_loopback:
            raise ValueError("remote backend forbidden")
        self.base = config["base_url"].rstrip("/")
        self.session = requests.Session()
        self.session.trust_env = (
            False  # A proxy must never redirect local video requests.
        )

    def verify(self):
        r = self.session.get(self.base + "/models", timeout=10, allow_redirects=False)
        r.raise_for_status()
        models = r.json()["data"]
        if self.config["model"] not in {m["id"] for m in models}:
            raise ValueError("served model identity differs")
        return models

    def complete(self, messages, output_tokens, timeout):
        payload = {
            "model": self.config["model"],
            "messages": messages,
            "temperature": 0,
            "seed": self.config["seed"],
            "max_tokens": output_tokens,
            "response_format": {"type": "json_object"},
        }
        response = self.session.post(
            self.base + "/chat/completions",
            json=payload,
            timeout=min(900, timeout),
            allow_redirects=False,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"local backend HTTP {response.status_code}: {response.text[:1000]}"
            )
        body = response.json()
        if body.get("model") != self.config["model"]:
            raise ValueError("response model identity mismatch")
        choice = body["choices"][0]
        usage = body.get("usage", {})
        if not all(
            isinstance(usage.get(k), int) and usage[k] >= 0
            for k in ("prompt_tokens", "completion_tokens")
        ):
            raise ValueError("missing authoritative backend token usage")
        return {
            "raw": choice["message"]["content"],
            "usage": usage,
            "finish_reason": choice["finish_reason"],
            "id": body.get("id"),
            "model": body["model"],
        }
