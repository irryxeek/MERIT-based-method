"""One-factor character-limit probe; no observation is admitted to memory here."""

from copy import deepcopy
from pathlib import Path

from frame_observer import FrameDescription
from io_utils import read_json, sha256
from schema import Contract

FINDING_GROUPS = ("corrected_known_scene_misattributions", "remaining_examples")
ARMS = {"E_uniform", "E_targeted"}


def unbounded_schema(**kwargs):
    """Remove exactly two character caps from the original generation schema."""
    schema = deepcopy(FrameDescription.model_json_schema(**kwargs))
    for field in ("visible", "uncertainty"):
        del schema["properties"][field]["maxLength"]
    return schema


class RelaxedFrameDescription(Contract):
    visible: str
    uncertainty: str

    @classmethod
    def model_json_schema(cls, **kwargs):
        # Preserve even the original title: the two caps are the only changes
        # in the schema supplied to the generation backend.
        return unbounded_schema(**kwargs)


def validate_unbounded(value):
    """Apply the old nonblank rule without truncation or semantic filtering."""
    value = RelaxedFrameDescription.model_validate(value).model_dump()
    if not value["visible"].strip() and not value["uncertainty"].strip():
        raise ValueError("A blank observation must explain uncertainty")
    return value


def _integer(value, label, *, maximum=None):
    if type(value) is not int or value < 0 or (maximum is not None and value > maximum):
        raise ValueError(f"Invalid {label}")


def _identity(record):
    qid, arm, number, position = (record[k] for k in ("qid", "arm", "number", "position"))
    _integer(qid, "qid")
    _integer(number, "observation number", maximum=1)
    _integer(position, "frame position", maximum=23)
    if not isinstance(arm, str) or arm not in ARMS:
        raise ValueError("Invalid observation arm")
    if record["key"] != f"{qid}/{arm}/frame_observe/{number}/{position}":
        raise ValueError("Frame record key disagrees with its identity")
    frames = record["proof"]["frames"]
    if not isinstance(frames, list) or len(frames) != 1:
        raise ValueError("A frame record must identify exactly one source")
    frame = frames[0]
    _integer(frame["frame_index"], "source frame index")
    if frame["id"] != f"s{frame['frame_index']}":
        raise ValueError("Frame ID/index mismatch")
    return qid, arm, number, position


def _findings(path):
    findings = read_json(path)
    selected = {}
    for group in FINDING_GROUPS:
        entries = findings[group]
        if not isinstance(entries, list):
            raise ValueError("Historical finding group must be a list")
        for entry in entries:
            qid, frame = entry["qid"], entry["frame"]
            _integer(qid, "finding qid")
            if not isinstance(frame, str) or not frame.startswith("s") or not frame[1:].isdigit():
                raise ValueError("Invalid finding frame ID")
            reasons = selected.setdefault((qid, frame), [])
            if group not in reasons:
                reasons.append(group)
    if len(selected) != 10:
        raise ValueError("Expected the ten frozen historical qid/frame pairs")
    return selected


def select_jobs(grounding_dir):
    """Select the frozen 28 middle positions plus ten historical regression pairs.

    Selection uses only occurrence identity, position and the two findings'
    (qid, frame) pairs. Old response text/status/length and GT are never used to
    include or exclude a record. Full old records and file receipts are returned
    for audit; the caller must keep those audit fields out of the model prompt.
    """
    root = Path(grounding_dir).resolve()
    known = _findings(root / "review/assistant_findings.json")
    indexed = {}
    for path in sorted((root / "frames").glob("*.json")):
        record = read_json(path)
        identity = _identity(record)
        key = record["key"]
        if key in indexed:
            if indexed[key][2] != record:
                raise ValueError("Conflicting records for the same frame key")
            continue
        indexed[key] = (identity, path, record)
    if len(indexed) != 672:
        raise ValueError("Expected all 672 unique parent frame occurrences")
    groups = {}
    for identity, _, _ in indexed.values():
        groups.setdefault(identity[:3], set()).add(identity[3])
    if len(groups) != 28 or any(positions != set(range(24)) for positions in groups.values()):
        raise ValueError("Expected 28 complete groups of 24 parent positions")

    matched, jobs = set(), []
    for identity, path, record in sorted(indexed.values(), key=lambda item: item[0]):
        qid, arm, number, position = identity
        frame = record["proof"]["frames"][0]
        pair = qid, frame["id"]
        reasons = ["systematic_position_12"] if position == 12 else []
        if pair in known:
            reasons.extend(known[pair])
            matched.add(pair)
        if not reasons:
            continue
        response_path = Path(record["response_path"])
        response_path = (response_path if response_path.is_absolute() else root / response_path).resolve()
        if not response_path.is_relative_to(root):
            raise ValueError("Parent response path escapes the declared grounding directory")
        response = read_json(response_path)
        if response["key"] != record["key"]:
            raise ValueError("Response and frame record keys disagree")
        jobs.append(
            {
                "qid": qid,
                "arm": arm,
                "number": number,
                "position": position,
                "key": record["key"],
                "frame": deepcopy(frame),
                "record": deepcopy(record),
                "record_path": str(path),
                "record_sha256": sha256(path),
                "response_path": str(response_path),
                "response_sha256": sha256(response_path),
                "request_sha256": response["request_sha256"],
                "selection_reasons": reasons,
            }
        )
    if matched != set(known):
        raise ValueError("A frozen historical regression frame is absent")
    if len(jobs) != 37 or sum(job["position"] == 12 for job in jobs) != 28:
        raise ValueError("Frozen selection must contain 37 occurrences including 28 middle positions")
    return jobs
