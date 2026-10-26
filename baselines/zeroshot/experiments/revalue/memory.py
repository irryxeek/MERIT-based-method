"""Immutable query-independent facts, bounded views and dependent judgments."""

from copy import deepcopy

from evidence_grounding import memory_source, validate_source
from io_utils import compact, digest
from schema import Facts, MemoryDraft


class FactStore:
    def __init__(self):
        self._facts = {}

    def extract(self, candidate, response, frame_ids):
        result = Facts.model_validate(response).model_dump()
        for index, fact in enumerate(result["facts"]):
            if not set(fact["frames"]) <= set(frame_ids):
                raise ValueError("Fact cites an unobserved or foreign frame")
            identity = f"c{candidate}_f{index}"
            value = {"id": identity, "candidate": candidate, **fact}
            if identity in self._facts and self._facts[identity] != value:
                raise ValueError("An immutable fact cannot be overwritten")
            self._facts[identity] = value
        # Unknown questions are facts about missing evidence, not invented events.
        for index, text in enumerate(result["unknown"]):
            identity = f"c{candidate}_u{index}"
            value = {"id": identity, "candidate": candidate, "kind": "unknown", "text": text, "frames": []}
            if identity in self._facts and self._facts[identity] != value:
                raise ValueError("An unknown entry cannot be overwritten")
            self._facts[identity] = value

    def dump(self):
        return deepcopy(self._facts)

    @property
    def version(self):
        return digest(self._facts)


def build_memory(draft, facts, tokenizer, capacity=1024):
    """Whole-entry trimming; only this serialized view may enter comparisons."""
    items = MemoryDraft.model_validate(draft).model_dump()["items"]
    retained = []
    evicted = []
    for item in items:
        ids = item["fact_ids"]
        if len(set(ids)) != len(ids) or not set(ids) <= set(facts):
            raise ValueError("Memory refers to missing or duplicate facts")
        # Side-specific source identities stay available for citation validation.
        sources = {i: memory_source(facts[i]) for i in ids}
        entry = {**item, "sources": sources}
        if len(tokenizer.encode(compact({"items": retained + [entry]}), add_special_tokens=False)) <= capacity:
            retained.append(entry)
        else:
            evicted.extend(ids)
    view = {"items": retained}
    kept = sorted({i for x in retained for i in x["fact_ids"]})
    return {
        "view": view,
        "view_sha256": digest(view),
        "tokens": len(tokenizer.encode(compact(view), add_special_tokens=False)),
        "capacity": capacity,
        "retained_fact_ids": kept,
        "deleted_fact_ids": sorted(set(facts) - set(kept)),
        "capacity_evictions": sorted(set(evicted) - set(kept)),
        "fact_store_version": digest(facts),
    }


def retrieve(memory, tokenizer):
    view = deepcopy(memory["view"])
    if digest(view) != memory["view_sha256"]:
        raise ValueError("Active memory was mutated")
    count = len(tokenizer.encode(compact(view), add_special_tokens=False))
    if count != memory["tokens"] or count > memory["capacity"] or memory["capacity"] > 1024:
        raise ValueError("Memory capacity exceeded, including on cache hits")
    for item in view["items"]:
        for source in item["sources"].values():
            validate_source(source)
    return view


def judgment(task, facts, result):
    return {"task_version": digest(task), "facts_version": digest(facts), "result": deepcopy(result)}


def valid_judgment(record, task, facts):
    return record["task_version"] == digest(task) and record["facts_version"] == digest(facts)
