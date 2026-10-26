"""Reserve before sending, preserve failed attempts and charge shared dependencies."""

from pathlib import Path

from io_utils import digest, read_json, utc_now, write_json


class BudgetStop(RuntimeError):
    pass


class Ledger:
    def __init__(self, path, limits):
        self.path = Path(path)
        self.limits = limits
        if self.path.exists():
            self.state = read_json(self.path)
            if self.state["limits"] != limits:
                raise ValueError("Budget cannot change on resume")
        else:
            self.state = {"limits": limits, "attempts": []}
            write_json(self.path, self.state)

    def attempts_for(self, key):
        return [a for a in self.state["attempts"] if a["key"] == key]

    def record_reuse(self, key, receipt):
        reused = self.state.setdefault("reused", {})
        if key in reused and reused[key] != receipt:
            raise ValueError("Reused dependency changed on resume")
        if key not in reused:
            reused[key] = receipt
            write_json(self.path, self.state)

    def cost(self, arm=None):
        attempts = [a for a in self.state["attempts"] if arm is None or arm in a["arms"]]
        return sum(a.get("charged_tokens", a["reserved_tokens"]) for a in attempts)

    def reserve(self, key, prompt_tokens, output_cap, arms, request_hash, frames):
        attempts = self.state["attempts"]
        previous = self.attempts_for(key)
        if previous and previous[-1]["status"] in {"complete", "invalid"}:
            raise ValueError("Finished semantic requests cannot be regenerated")
        if previous and previous[-1]["request_sha256"] != request_hash:
            raise ValueError("Retry changed prompt, schema or evidence")
        if len(previous) >= 2:
            raise BudgetStop("Per-request transient retry limit")
        distinct = len({a["key"] for a in attempts})
        retries = len(attempts) - distinct
        if (
            (not previous and distinct >= self.limits["requests"])
            or (previous and retries >= self.limits["extra_attempts"])
            or len(attempts) >= self.limits["requests"] + self.limits["extra_attempts"]
        ):
            raise BudgetStop("Stage request budget")
        reservation = prompt_tokens + output_cap
        for arm in arms:
            if self.cost(arm) + reservation > self.limits["arm_incremental_tokens"].get(arm, float("inf")):
                raise BudgetStop(f"End-to-end 6x B0 token budget: {arm}")
        entry = {
            "key": key,
            "number": len(attempts),
            "arms": arms,
            "status": "reserved",
            "created_at": utc_now(),
            "prompt_tokens": prompt_tokens,
            "output_cap": output_cap,
            "reserved_tokens": reservation,
            "request_sha256": request_hash,
            "model_frames": frames,
        }
        attempts.append(entry)
        write_json(self.path, self.state)
        return entry["number"]

    def settle(self, number, record):
        item = self.state["attempts"][number]
        if item["status"] != "reserved":
            raise ValueError("Attempt already settled")
        item.update(status=record["status"], record_sha256=digest(record), finished_at=utc_now())
        if "prompt_tokens" in record:
            if record["prompt_tokens"] != item["prompt_tokens"] or record["output_tokens"] > item["output_cap"]:
                raise ValueError("Actual tokens disagree with reservation")
            item["charged_tokens"] = record["prompt_tokens"] + record["output_tokens"]
        # Lost/runtime attempts retain the full reservation, never a zero charge.
        write_json(self.path, self.state)
