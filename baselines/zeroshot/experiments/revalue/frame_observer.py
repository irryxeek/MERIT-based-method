"""A still-frame observer: backend assigns the sole source, never generated IDs."""

import hashlib
from copy import deepcopy

import numpy as np
from evidence_grounding import ground_observation
from pydantic import Field
from schema import Contract

FRAME_PROMPT = (
    "The input contains two identical copies of ONE still frame for video-encoder compatibility. "
    "Describe only the clearly visible people, objects, clothing, pose, and setting in this image. "
    "Use one short, concrete sentence in visible. Do not infer motion, intent, identity, unseen outcomes, "
    "earlier/later events, or text that is not legible. Do not guess from a possible video topic. "
    "Use uncertainty for important details you cannot discern, or an empty string if none. "
    "If nothing can be discerned, leave visible empty and explain in uncertainty."
)
FRAME_OUTPUT_CAP = 160


class FrameDescription(Contract):
    visible: str = Field(max_length=160)
    uncertainty: str = Field(max_length=80)


def validate_description(value):
    value = FrameDescription.model_validate(value).model_dump()
    if not value["visible"].strip() and not value["uncertainty"].strip():
        raise ValueError("A blank observation must explain uncertainty")
    return value


def has_visible_claim(text):
    """Exclude empty/punctuation placeholders, without rewriting generated claims."""
    return sum(character.isalpha() for character in text) >= 3


def still_payload(video, trace, position):
    frame = deepcopy(trace["frames"][position])
    pixels = video["video"][position]
    if hashlib.sha256(pixels.tobytes()).hexdigest() != frame["rgb_sha256"]:
        raise ValueError("Still pixels differ from frozen parent pixels")
    repeated = np.stack([pixels, pixels])
    metadata = deepcopy(video["metadata"])
    metadata["frames_indices"] = [metadata["frames_indices"][position]] * 2
    proof = {
        "frames": [frame], "model_frame_ids": [frame["id"], frame["id"]],
        "displayed_frames": 2, "unique_source_frames": 1, "repeated_source_frames": 1,
        "additional_source_content_vs_parent": 0,
        "representation": "same_source_frame_duplicated_for_temporal_patch_size",
        "rgb_sha256": hashlib.sha256(repeated.tobytes()).hexdigest(),
    }
    return {"video": repeated, "metadata": metadata}, proof


def grounded_description(value, proof, candidate, target_interval, namespace):
    value = validate_description(value)
    refs = [f["id"] for f in proof["frames"]]
    if len(refs) != 1:
        raise ValueError("Still observation requires exactly one source frame")
    result = {
        "facts": [{"kind": "context", "text": value["visible"], "frames": refs}] if has_visible_claim(value["visible"]) else [],
        "unknown": [value["uncertainty"]] if value["uncertainty"].strip() else [],
    }
    return ground_observation(result, proof, candidate, target_interval, namespace)
