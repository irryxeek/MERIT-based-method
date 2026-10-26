"""A bounded visual verifier contract; structural validity is not factual truth."""

import re
from typing import Literal

from schema import Contract

ARMS = ("direct", "evidence_first")
OUTPUT_CAP = 160
COMMON_PROMPT = (
    "The input contains two identical copies of ONE still frame for encoder compatibility. "
    "Check only the object-presence claim in the specified image region. "
    "supported means the object is clearly identifiable there. contradicted means the region "
    "is clearly visible and shows absence of that object or an incompatible object. "
    "cannot_determine means blur, occlusion, framing or ambiguous visual detail prevents a decision. "
    "Report a short concrete visible observation in evidence, the status, and any relevant "
    "uncertainty (empty if none). Do not infer motion, identity or unseen events. "
)
EVIDENCE_INSTRUCTION = (
    "Before deciding, inspect the specified region for recognizable object features. "
    "Distinguish an object's actual outline and parts from a hand, shadow, splash or background. "
    "A suggestive pose or scene is not proof of the object. If identifying features are hidden "
    "or too blurred, use cannot_determine instead of guessing either presence or absence. "
    "Keep evidence limited to what is visible; do not presuppose the queried object in uncertainty. "
)


class ObjectPresence(Contract):
    evidence: str
    status: Literal["supported", "contradicted", "cannot_determine"]
    uncertainty: str


def validate_presence(value):
    result = ObjectPresence.model_validate(value).model_dump()
    if sum(c.isalpha() for c in result["evidence"]) < 3:
        raise ValueError("A decision needs an observable evidence statement")
    if result["status"] == "cannot_determine" and sum(c.isalpha() for c in result["uncertainty"]) < 3:
        raise ValueError("Cannot determine must explain the missing visual evidence")
    return result


def prompt(case, arm):
    if arm not in ARMS:
        raise ValueError("Unknown verifier arm")
    # Allowlist: labels, old captions, saliency GT and selection strata cannot enter.
    return (
        COMMON_PROMPT + (EVIDENCE_INSTRUCTION if arm == "evidence_first" else "")
        + f"\nObject: {case['object']}\nRegion: {case['region']}\nClaim: {case['claim']}"
    )


def case_jobs(cases):
    """Paired arms alternate order; no outcome-dependent selection or retries."""
    ids = [c["case_id"] for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate case ID")
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in ids):
        raise ValueError("Case ID must be safe for local filenames")
    jobs = []
    for index, case in enumerate(cases):
        for arm in (ARMS if index % 2 == 0 else ARMS[::-1]):
            public = {key: case[key] for key in ("case_id", "image_key", "qid", "object", "region", "claim")}
            jobs.append({**public, "arm": arm, "key": f"{case['case_id']}/{arm}", "text": prompt(public, arm)})
    return jobs


def validate_case_sources(cases, sources):
    """Check every claim, including the second claim sharing the same image."""
    for case in cases:
        if case["qid"] != sources[case["image_key"]]["qid"]:
            raise ValueError("Every claim must retain its source query identity")
