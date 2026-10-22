"""Fixed metadata, constrained output, and explicit field-level memory repair.

Validation establishes shape/provenance only, not semantic truth. Unsupported
claims may be removed or marked unknown, never repaired by inventing citations.
"""

import copy
import json

from .inference import visual_content
from .memory import SYSTEM, check_values, frame_index, references, text, validate_memory
from .pilot_components import brief, validate_context_link


def object_schema(properties):
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def ids_schema(allowed, minimum=0):
    return {
        "type": "array",
        "items": {"type": "string", **({"enum": list(allowed)} if allowed else {})},
        "minItems": minimum,
        "maxItems": len(allowed),
    }


def pointer(path):
    return "/" + "/".join(path)


def get_field(data, path):
    for key in path:
        data = data[key]
    return data


class MemoryContract:
    def __init__(self, units, frames, values, duration, context):
        self.units = {u["id"]: u for u in units}
        self.frames, self.values, self.duration = frames, values, duration
        self.cues = check_values(values)
        self.previous = {e["id"]: e for e in context}
        indexed = frame_index(frames, duration)
        for u in units:
            references(u["evidence_ids"], indexed)
            if not 0 <= u["start"] < u["end"] <= duration or any(
                not u["start"] <= indexed[i]["pts"] < u["end"] for i in u["evidence_ids"]
            ):
                raise ValueError("Observation unit does not contain its supplied pixels")
        observations = {}
        for eid, unit in self.units.items():
            cue_schema = object_schema(
                {
                    "status": {"enum": ["observed", "unknown", "not_applicable"]},
                    "level": {"enum": [None, 0, 1, 2, 3]},
                    "cue": {"type": "string", "minLength": 1, "maxLength": 160},
                    "evidence_ids": ids_schema(unit["evidence_ids"]),
                    "reference_event_ids": ids_schema(self.allowed_refs(eid)),
                }
            )
            observations[eid] = object_schema(
                {
                    "description": {"type": "string", "minLength": 1, "maxLength": 240},
                    "evidence_ids": ids_schema(unit["evidence_ids"], 1),
                    "values": object_schema({c: copy.deepcopy(cue_schema) for c in self.cues}),
                }
            )
        all_ids = list(self.previous) + list(self.units)
        all_frames = sorted(set(indexed) | {i for e in context for i in e["evidence_ids"]})
        self.schema = object_schema(
            {
                "observations": object_schema(observations),
                "links": {
                    "type": "array",
                    "maxItems": 8,
                    "items": object_schema(
                        {
                            "source": {"enum": all_ids},
                            "target": {"enum": all_ids},
                            "relation": {"enum": ["before", "possible_link"]},
                            "evidence_ids": ids_schema(all_frames, 1),
                        }
                    ),
                },
            }
        )

    def allowed_refs(self, eid):
        return sorted((set(self.previous) | set(self.units)) - {eid})

    def field_schema(self, path):
        schema = self.schema
        for key in path:
            schema = schema["properties"][key]
        return schema

    def problems(self, payload):
        """Return disjoint repair paths; valid fields are held fixed on retry."""
        issues = []

        def issue(path, reason):
            issues.append({"path": path, "reason": reason})

        def keys(value, expected, path):
            if not isinstance(value, dict) or set(value) != set(expected):
                issue(path, f"Required keys exactly {list(expected)}")
                return False
            return True

        if not keys(payload, ("observations", "links"), []):
            return issues
        if not keys(payload["observations"], self.units, ["observations"]):
            return issues
        for eid, unit in self.units.items():
            obs = payload["observations"][eid]
            base = ["observations", eid]
            if not keys(obs, ("description", "evidence_ids", "values"), base):
                continue
            if not text(obs["description"], 240):
                issue(base + ["description"], "description must contain 1-240 nonblank characters")
            try:
                references(obs["evidence_ids"], unit["evidence_ids"])
            except ValueError as error:
                issue(base + ["evidence_ids"], f"{error}; allowed={unit['evidence_ids']}")
            if not keys(obs["values"], self.cues, base + ["values"]):
                continue
            for dimension, cue in obs["values"].items():
                path = base + ["values", dimension]
                if not keys(cue, self.field_schema(path)["properties"], path):
                    continue
                try:
                    if cue["status"] not in {"observed", "unknown", "not_applicable"}:
                        raise ValueError("status must be observed, unknown or not_applicable")
                    if not text(cue["cue"], 160):
                        length = len(cue["cue"]) if isinstance(cue["cue"], str) else None
                        raise ValueError(f"cue must contain 1-160 nonblank characters; received={length}")
                    observed = cue["status"] == "observed"
                    references(cue["evidence_ids"], unit["evidence_ids"], required=observed)
                    references(cue["reference_event_ids"], self.allowed_refs(eid), required=False)
                    if observed and (type(cue["level"]) is not int or not 0 <= cue["level"] <= 3):
                        raise ValueError("observed level must be an integer 0-3")
                    if not observed and cue["level"] is not None:
                        raise ValueError("unknown/not_applicable requires level=null")
                    if (
                        observed
                        and dimension in {"novelty_cue", "importance_cue"}
                        and not cue["reference_event_ids"]
                    ):
                        raise ValueError("observed novelty/importance requires another supplied observation")
                except (ValueError, TypeError) as error:
                    issue(
                        path,
                        f"{error}; allowed frames={unit['evidence_ids']}; "
                        f"allowed references={self.allowed_refs(eid)}. "
                        "Use unknown, null, and empty citations when evidence is insufficient.",
                    )
        return issues

    def repair_spec(self, payload, issues):
        schema = object_schema({pointer(i["path"]): self.field_schema(i["path"]) for i in issues})
        instruction = (
            "Repair ONLY the listed invalid fields. Return an object keyed by the exact supplied paths, "
            "each containing its replacement value. All other fields remain frozen in the controller. "
            "Do not invent support or replace citations just to pass validation. If a cue lacks valid "
            "evidence, return status=unknown, level=null, evidence_ids=[], reference_event_ids=[].\n"
            + json.dumps({"validation_errors": issues, "previous_candidate": payload})
        )
        return schema, instruction

    def apply_patch(self, original, patch, issues):
        if not isinstance(patch, dict) or set(patch) != {pointer(i["path"]) for i in issues}:
            raise ValueError("Repair may replace only the listed invalid fields")
        result = copy.deepcopy(original)
        for issue in issues:
            path = issue["path"]
            if not path:
                result = copy.deepcopy(patch["/"])
            else:
                get_field(result, path[:-1])[path[-1]] = copy.deepcopy(patch[pointer(path)])
        return result

    def finalize(self, payload, *, degrade=False):
        payload = copy.deepcopy(payload)
        issues = self.problems(payload)
        if issues and not degrade:
            raise ValueError(json.dumps(issues))
        downgraded = []
        for issue in issues:
            path = issue["path"]
            if len(path) != 4 or path[2] != "values":
                raise ValueError("Unrecoverable observation field: " + json.dumps(issue))
            prior = copy.deepcopy(get_field(payload, path))
            get_field(payload, path[:-1])[path[-1]] = {
                "status": "unknown",
                "level": None,
                "cue": "insufficient valid evidence after validation",
                "evidence_ids": [],
                "reference_event_ids": [],
            }
            downgraded.append({**issue, "original": prior, "action": "cue_to_unknown"})
        events = []
        for eid, unit in self.units.items():
            obs = payload["observations"][eid]
            # Only explicit model citations enter the node's provenance union.
            cited = set(obs["evidence_ids"]) | {i for v in obs["values"].values() for i in v["evidence_ids"]}
            event = {
                "id": eid,
                "start": unit["start"],
                "end": unit["end"],
                "description": obs["description"],
                "evidence_ids": [i for i in unit["evidence_ids"] if i in cited],
                "values": [{"dimension": d, **obs["values"][d]} for d in self.cues],
            }
            stripped = copy.deepcopy(event)
            for v in stripped["values"]:
                v.pop("reference_event_ids")
            validate_memory({"events": [stripped], "links": []}, self.frames, self.values, self.duration)
            events.append(event)
        current = {e["id"]: e for e in events}
        available = {**self.previous, **current}
        accepted, rejected, seen = [], [], set()
        links = payload["links"]
        if not isinstance(links, list) or len(links) > 8:
            rejected.append({"edge": links, "reason": "links must be an array of at most eight edges"})
            links = []
        for edge in links:
            try:
                key = validate_context_link(edge, available, current, seen)
            except (ValueError, TypeError, KeyError) as error:
                rejected.append({"edge": edge, "reason": str(error)})
            else:
                accepted.append(edge)
                seen.add(key)
        return {
            "events": events,
            "links": accepted,
            "rejected_links": rejected,
            "downgraded_cues": downgraded,
        }


def extract(client, frames, values, duration, chunk, arm, context, units):
    contract = MemoryContract(units, frames, values, duration, context)
    prompt = (
        "Build query-independent visual evidence memory for the CURRENT input window. "
        "Return one observation for EACH observation_unit, keyed by its supplied ID. "
        "The controller supplies fixed IDs, start/end times and dimension names; do not generate them. "
        "A unit is observation support, not a true action boundary. Describe only facts visible in its "
        "own supplied pictures. A still frame cannot prove motion between samples. "
        "description <=240 characters; each cue <=160 characters. Use short sentences. "
        "Each observation's evidence_ids cite its description; each cue cites its own evidence. "
        "All citations must belong to that unit. Every observation needs all four cue dimensions. "
        "Status is observed/unknown/not_applicable; observed levels 0 absent,1 mild,2 moderate,3 strong; "
        "otherwise level=null. Observed cues require frame citations. "
        "Prior observations are imperfect frozen model outputs, not truth. "
        "Observed novelty/importance MUST cite another supplied current/prior event AND describe a "
        "concrete contrast or event-goal/outcome; otherwise unknown with null level and empty citations. "
        "Do not copy a reference from a prior record if the target is not supplied. "
        "Return optional top-level links only, at most eight. Each cites BOTH endpoints and has at "
        "least one current endpoint. before means earlier nonoverlapping source -> later target; "
        "possible_link is an unverified association, never causation. Empty links allowed. "
        "Return a compact JSON object matching the supplied schema, without commentary.\n"
        + json.dumps(
            {
                "video_duration": duration,
                "current_interval": [chunk["start"], chunk["end"]],
                "value_definitions": values,
                "prior_observations": [brief(e) for e in context],
                "allowed_prior_reference_event_ids": sorted(contract.previous),
                "observation_units": units,
            }
        )
    )
    return client.complete_memory(
        arm + "_memory", visual_content(prompt, frames), 8192, contract, system=SYSTEM
    )
