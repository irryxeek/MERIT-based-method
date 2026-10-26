"""Constrain extraction to exact source spans and explicit output limits, before decoding."""

import re

from prompts import TASK
from schema import TaskSpec, validate_task

TASK_GROUNDED = TASK + (
    " Preserve original spelling, capitalization and repeated spaces in every quote; do not correct typos. "
    "count means the requested number of OUTPUT highlights, not the number of people or objects in a scene. "
    "duration_seconds requires an explicitly numerical time budget; seasons are not duration budgets. "
    "If no numerical output limit is stated, extract only object/event constraints."
)
WORDS = dict(zip("one two three four five six seven eight nine ten eleven twelve".split(), range(1, 13)))
NUMBER = r"(?:\d+(?:\.\d+)?|" + "|".join(WORDS) + r")"


def quote_spans(query):
    """All contiguous whitespace-token spans; exact original bytes, including internal whitespace."""
    words = list(re.finditer(r"\S+", query))
    return sorted({query[a.start() : b.end()] for i, a in enumerate(words) for b in words[i:]})


def numeric_options(query):
    patterns = [
        ("count", rf"\btop[- ]?(?P<n>{NUMBER})\b", 1),
        ("count", rf"\b(?:show|select|pick|return|give|find|choose)\s+(?:me\s+)?(?P<n>{NUMBER})\b", 1),
        ("duration_seconds", rf"\b(?P<n>{NUMBER})\s*(?:seconds?|secs?|s)\b", 1),
        ("duration_seconds", rf"\b(?P<n>{NUMBER})\s*(?:minutes?|mins?)\b", 60),
    ]
    options = []
    for kind, pattern, scale in patterns:
        for match in re.finditer(pattern, query, re.IGNORECASE):
            word = match.group("n").lower()
            number = float(WORDS[word]) if word in WORDS else float(word)
            if number <= 0 or (kind == "count" and not number.is_integer()):
                continue
            option = {"kind": kind, "quote": match.group(0), "number": number * scale}
            if option not in options:
                options.append(option)
    return options


def task_schema(query):
    schema = TaskSpec.model_json_schema()
    spans = quote_spans(query)
    if not spans:
        raise ValueError("Empty query cannot be grounded")
    ordinary = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "kind": {"type": "string", "enum": ["object", "event"]},
            "quote": {"type": "string", "enum": spans},
            "number": {"type": "null"},
        },
        "required": ["kind", "quote", "number"],
    }
    branches = [ordinary]
    for item in numeric_options(query):
        branches.append(
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {k: {"const": v} for k, v in item.items()},
                "required": list(item),
            }
        )
    schema["properties"]["constraints"]["items"] = {"anyOf": branches}
    schema.pop("$defs", None)
    return schema


def validate_grounded_task(value, query):
    task = validate_task(value, query)
    allowed_numbers = numeric_options(query)
    for item in task["constraints"]:
        if item["kind"] in {"count", "duration_seconds"} and item not in allowed_numbers:
            raise ValueError("Numerical output limit lacks an explicit grounded request span")
    return task


def extract_task(row, backend, arms, cap=256):
    return backend.call(
        str(row["qid"]) + "/task",
        TASK_GROUNDED.format(query=row["query"]),
        [],
        [],
        TaskSpec,
        cap,
        arms,
        lambda x: validate_grounded_task(x, row["query"]),
        output_schema=task_schema(row["query"]),
    )
