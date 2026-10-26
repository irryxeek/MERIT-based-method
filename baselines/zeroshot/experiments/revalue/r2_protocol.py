"""Frozen R2 inputs, exclusive query ownership and atomic global budgets."""

import fcntl
import hashlib
from contextlib import contextmanager
from pathlib import Path

from budget import BudgetStop, Ledger
from io_utils import ROOT, legacy, read_json, sha256, write_json

ARMS = ["C_direct4", "E_memory", "E_uniform", "E_targeted"]
OBSERVERS = ["E_uniform", "E_targeted"]
R1_ARMS = ["C_direct4", "M_plain", "M_decision"]


def mapped_arms(arms):
    mapped = set()
    for arm in arms:
        if arm == "M_decision":
            mapped.update(["E_memory", *OBSERVERS])
        elif arm != "M_plain":
            mapped.add(arm)
    if not mapped <= set(ARMS):
        raise ValueError("Unknown R2 cost owner")
    return sorted(mapped)


def ordered_rows(rows):
    if len(rows) != 100 or len({r["qid"] for r in rows}) != 100:
        raise ValueError("R2 requires the original complete explore100")
    return sorted(rows, key=lambda r: hashlib.sha256(("revalue-v3:" + str(r["qid"])).encode()).hexdigest())


def load_run(run):
    run = Path(run)
    binding = read_json(run / "binding.json")
    if binding["protocol"] != "revalue-v3-r2-v3":
        raise ValueError("Unknown R2 protocol")
    for name, expected in binding["source_sha256"].items():
        if sha256(ROOT / name) != expected or sha256(run / "source_snapshot" / name) != expected:
            raise ValueError(f"Frozen R2 source drift: {name}")
    for name, expected in binding["input_sha256"].items():
        if sha256(run / name) != expected:
            raise ValueError(f"Frozen R2 input drift: {name}")
    for name, expected in binding["parent_receipts"].items():
        if sha256(name) != expected:
            raise ValueError(f"R2 parent receipt drift: {name}")
    rows = legacy.read_jsonl(run / "inference.jsonl")
    legacy.validate_rows(rows)
    expected = ordered_rows(legacy.read_jsonl(Path(binding["pair_run"]) / "inference.jsonl"))
    if rows != expected:
        raise ValueError("Frozen R2 query selection changed")
    return binding, rows, sha256(run / "binding.json")


class CostLedger(Ledger):
    def cost(self, arm=None):
        total = super().cost(arm)
        if arm is not None:
            total += sum(
                r["charged_tokens"]
                for r in self.state.get("reused", {}).values()
                if arm in mapped_arms(r["original_arms"])
            )
            total += sum(v.get(arm, 0) for v in self.state.get("fallback_costs", {}).values())
        return total


class SharedLedger:
    """One atomic request ledger shared by independent GPU workers.

    The original request specification retains R1 arm names for exact cache
    verification; billing expands M_decision into its three R2 descendants.
    """

    def __init__(self, directory, limits):
        self.directory, self.limits = Path(directory), limits
        self.path = self.directory / "ledger.json"
        self.directory.mkdir(parents=True, exist_ok=True)
        with self.transaction() as ledger:
            if not self.path.exists():
                write_json(self.path, ledger.state)

    @contextmanager
    def transaction(self):
        with (self.directory / "ledger.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield CostLedger(self.path, self.limits)

    @property
    def state(self):
        return read_json(self.path)

    def attempts_for(self, key):
        return [a for a in self.state["attempts"] if a["key"] == key]

    def reserve(self, key, prompt_tokens, output_cap, arms, request_hash, frames):
        with self.transaction() as ledger:
            if (self.directory / "stop_request.json").exists():
                raise BudgetStop("R2 global stop requested")
            for arm in mapped_arms(arms):
                fallback_paid = sum(v.get(arm, 0) for v in ledger.state.get("fallback_costs", {}).values())
                contingency = max(0, self.limits.get("fallback_reserve_tokens", 0) - fallback_paid)
                if (
                    ledger.cost(arm) + prompt_tokens + output_cap + contingency
                    > self.limits["arm_incremental_tokens"][arm]
                ):
                    raise BudgetStop("R2 token cap including fallback contingency")
            return ledger.reserve(key, prompt_tokens, output_cap, mapped_arms(arms), request_hash, frames)

    def settle(self, number, record):
        with self.transaction() as ledger:
            ledger.settle(number, record)

    def record_reuse(self, key, receipt):
        with self.transaction() as ledger:
            if key not in ledger.state.get("reused", {}):
                for arm in mapped_arms(receipt["original_arms"]):
                    fallback_paid = sum(v.get(arm, 0) for v in ledger.state.get("fallback_costs", {}).values())
                    contingency = max(0, self.limits.get("fallback_reserve_tokens", 0) - fallback_paid)
                    if (
                        ledger.cost(arm) + receipt["charged_tokens"] + contingency
                        > self.limits["arm_incremental_tokens"][arm]
                    ):
                        raise BudgetStop("R2 cold-start token cap, including reused requests")
            ledger.record_reuse(key, receipt)

    def fallback_cost(self, qid, arm, tokens):
        with self.transaction() as ledger:
            entry = ledger.state.setdefault("fallback_costs", {}).setdefault(str(qid), {})
            if arm in entry and entry[arm] != tokens:
                raise ValueError("Fallback cost changed on resume")
            entry[arm] = tokens
            write_json(self.path, ledger.state)

    def cost(self, arm=None):
        with self.transaction() as ledger:
            return ledger.cost(arm)


@contextmanager
def claim_query(directory, qid):
    path = Path(directory) / "query_locks" / f"{qid}.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield not (Path(directory) / "records" / f"{qid}.json").exists()
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)
