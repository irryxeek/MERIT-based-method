"""Small strict contracts; schema validity does not imply factual correctness."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

VALUE = Literal["outcome", "information", "emotion", "action", "sensory"]
ELIGIBILITY = Literal["satisfied", "violated", "unknown"]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Constraint(Contract):
    kind: Literal["object", "event", "count", "duration_seconds"]
    quote: str
    number: float | None


class TaskSpec(Contract):
    request_type: Literal["generic", "broad", "specific"]
    purpose: str
    constraints: list[Constraint] = Field(max_length=6)


class FactItem(Contract):
    kind: Literal["event", "phase", "outcome", "reaction", "identity", "context", "unknown"]
    text: str
    frames: list[str] = Field(min_length=1, max_length=4)


class Facts(Contract):
    facts: list[FactItem] = Field(max_length=6)
    unknown: list[str] = Field(max_length=3)


class ObservationFact(Contract):
    kind: Literal["event", "phase", "outcome", "reaction", "identity", "context", "unknown"]
    text: str = Field(min_length=8, max_length=96)
    frames: list[str] = Field(min_length=1, max_length=4)


class ObservationFacts(Contract):
    facts: list[ObservationFact] = Field(min_length=1, max_length=6)
    unknown: list[Annotated[str, Field(min_length=1, max_length=64)]] = Field(max_length=3)


class MemoryItem(Contract):
    text: str
    fact_ids: list[str] = Field(min_length=1, max_length=6)


class MemoryDraft(Contract):
    items: list[MemoryItem] = Field(max_length=12)


class Comparison(Contract):
    a_eligibility: ELIGIBILITY
    b_eligibility: ELIGIBILITY
    preference: Literal["a_better", "b_better", "tie", "insufficient"]
    values: list[VALUE] = Field(max_length=5)
    reason: str
    a_evidence: list[str] = Field(max_length=4)
    b_evidence: list[str] = Field(max_length=4)
    missing_question: str | None


class Action(Contract):
    gap: Literal["coverage", "attribution", "value", "boundary"]
    tool: Literal["Scan", "Segment", "Stitch"]
    question: str
    intervals: list[list[float]] = Field(min_length=1, max_length=4)
    frame_limit: int = Field(ge=2, le=24)


def validate_task(value, query):
    task = TaskSpec.model_validate(value).model_dump()
    for item in task["constraints"]:
        if not item["quote"] or item["quote"] not in query:
            raise ValueError("Constraint lacks verbatim query provenance")
        if item["kind"] in {"count", "duration_seconds"}:
            if item["number"] is None or item["number"] <= 0:
                raise ValueError("Invalid explicit numerical constraint")
            if item["kind"] == "count" and not item["number"].is_integer():
                raise ValueError("Count must be integral")
        elif item["number"] is not None:
            raise ValueError("Only count and duration may have a number")
    if len({c["kind"] for c in task["constraints"] if c["number"] is not None}) != sum(
        c["number"] is not None for c in task["constraints"]
    ):
        raise ValueError("Conflicting numeric constraints")
    return {"query": query, **task}


def numeric_limit(task, kind):
    return next((c["number"] for c in task["constraints"] if c["kind"] == kind), None)
