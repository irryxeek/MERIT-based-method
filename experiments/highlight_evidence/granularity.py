"""Matched E1 memory-granularity experiment with frozen candidates and pixels.

Both arms rebuild only the already selected windows, using the same frozen
previous-two-window observations. C10 emits one observation per input window;
F2 emits one per official 2-second bin. These are observation supports, not
claims that an action starts or ends at an artificial grid boundary.
"""

import argparse
import copy
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import numpy as np
from hd_adapter.client import Client, locked
from hd_adapter.runner import request_cost

from merit_qvh.cli import validate_submission
from merit_qvh.core import digest, file_sha256, read_jsonl, validate_manifest, write_json, write_jsonl

from .inference import visual_content
from .memory import SYSTEM, check_values, references, validate_memory
from .pilot import load_frames, read
from .pilot import verify as verify_pilot
from .pilot_components import brief, score_evidence, validate_context_link
from .planner import merge_dense_scores

ARMS = ("C10", "F2")
GROUPS = ("D0", "E1_original", *ARMS)
GPU_UUIDS = [uuid.strip() for uuid in os.environ.get("REPRO_GPU_UUIDS", "").split(",") if uuid.strip()]


def units_for(chunk, arm):
    if arm not in ARMS:
        raise ValueError("Unknown memory arm")
    ids = chunk["bin_ids"]
    groups = [ids] if arm == "C10" else [[i] for i in ids]
    return [
        {
            "id": f"e{j}",
            "start": 2 * group[0],
            "end": 2 * (group[-1] + 1),
            "evidence_ids": ["f" + str(i) for i in group],
        }
        for j, group in enumerate(groups)
    ]


def extract_units(client, frames, values, duration, chunk, arm, context, *, robust=False):
    units = units_for(chunk, arm)
    if robust:
        from .robust_memory import extract

        result = extract(client, frames, values, duration, chunk, arm, context, units)
        return namespace_result(result, chunk, units)
    expected = {u["id"]: u for u in units}
    previous = {e["id"]: e for e in context}
    cue_ids = check_values(values)

    def validate(payload):
        if not isinstance(payload, dict) or set(payload) != {"events", "links"}:
            raise ValueError("Return events and links only")
        events = payload["events"]
        if (
            not isinstance(events, list)
            or any(not isinstance(e, dict) for e in events)
            or [e.get("id") for e in events] != list(expected)
        ):
            raise ValueError("Return exactly one observation per unit, in order: " + str(list(expected)))
        current = {e["id"]: e for e in events}
        available = {**previous, **current}
        for event in events:
            unit = expected[event["id"]]
            if (event["start"], event["end"]) != (unit["start"], unit["end"]):
                raise ValueError("Observation support must equal its supplied unit bounds: " + str(unit))
            references(event["evidence_ids"], dict.fromkeys(unit["evidence_ids"]))
            stripped = copy.deepcopy(event)
            for v in stripped["values"]:
                refs = v.pop("reference_event_ids")
                references(refs, available, required=False)
                if event["id"] in refs:
                    raise ValueError("No self-reference in contextual evidence")
                if (
                    v["status"] == "observed"
                    and v["dimension"] in {"novelty_cue", "importance_cue"}
                    and not refs
                ):
                    raise ValueError("Observed novelty/importance requires another supplied event reference")
            # Validate nodes separately: the shared legacy validator caps a call
            # at four nodes, whereas five F2 units are required for a full window.
            validate_memory({"events": [stripped], "links": []}, frames, values, duration)
        if not isinstance(payload["links"], list) or len(payload["links"]) > 8:
            raise ValueError("At most eight optional links")
        accepted, rejected, seen = [], [], set()
        for edge in payload["links"]:
            try:
                key = validate_context_link(edge, available, current, seen)
            except (ValueError, TypeError, KeyError) as error:
                rejected.append({"edge": edge, "reason": str(error)})
            else:
                seen.add(key)
                accepted.append(edge)
        return {"events": events, "links": accepted, "rejected_links": rejected}

    example = {
        "events": [
            {
                **units[0],
                "description": "brief observation supported by supplied pictures",
                "values": [
                    {
                        "dimension": c,
                        "status": "unknown",
                        "level": None,
                        "cue": "insufficient visible evidence",
                        "evidence_ids": [],
                        "reference_event_ids": [],
                    }
                    for c in cue_ids
                ],
            }
        ],
        "links": [],
    }
    prompt = (
        "Build query-independent visual evidence memory for the CURRENT input window. "
        "Return exactly ONE observation per supplied observation_unit, in the supplied order. "
        "Use each unit's exact id,start,end; cite only its supplied evidence_ids. "
        "A unit is an observation support, NOT a claim about the true action onset or offset. "
        "Record a neutral observation or unknown cues when little happens; do not invent a distinct action. "
        "You may use all current pictures and the supplied prior observations as context, but describe "
        "only facts supported at the current unit's cited pictures. A still frame does not prove motion "
        "between samples. Each observation has all four cue dimensions exactly once. "
        "Status is observed/unknown/not_applicable; observed ordinal levels are 0 absent,1 mild,2 moderate,3 strong; "
        "otherwise level=null. Observed cues cite frames belonging to their own observation. "
        "Description <=240 characters; cue <=160 characters. "
        "Prior observations are imperfect frozen model outputs, not truth. "
        "Observed novelty/importance MUST cite another supplied current/prior event in reference_event_ids "
        "and a concrete contrast or event-goal/outcome; otherwise use unknown with null level. "
        "Do not copy a reference from a prior record if its target is not supplied here. "
        "Return up to eight optional links {source,target,relation:before|possible_link,evidence_ids}. "
        "Cite a frame from BOTH endpoints; at least one endpoint is current. "
        "before is earlier nonoverlapping source -> later target. No causal edges. "
        "Do not repeat all observations just to create links. Empty links allowed.\n"
        + json.dumps(
            {
                "video_duration": duration,
                "current_interval": [chunk["start"], chunk["end"]],
                "value_definitions": values,
                "prior_observations": [brief(e) for e in context],
                "allowed_prior_reference_event_ids": sorted(previous),
                "observation_units": units,
                "schema_example_first_unit_only": example,
            }
        )
    )
    result = copy.deepcopy(
        client.complete(arm + "_memory", visual_content(prompt, frames), 8192, validate, system=SYSTEM)
    )
    return namespace_result(result, chunk, units)


def namespace_result(result, chunk, units):
    result = copy.deepcopy(result)
    namespace = "g" + str(chunk["window_id"]) + ":"
    local = {u["id"] for u in units}
    for event in result["events"]:
        event["id"] = namespace + event["id"]
        for value in event["values"]:
            value["reference_event_ids"] = [
                namespace + r if r in local else r for r in value["reference_event_ids"]
            ]
    for edge in result["links"]:
        for key in ("source", "target"):
            if edge[key] in local:
                edge[key] = namespace + edge[key]
    return {**{k: chunk[k] for k in ("window_id", "start", "end", "bin_ids")}, **result}


def scoring_memory(chunks, originals):
    selected = [e for c in chunks for e in c["events"]]
    current = {e["id"] for e in selected}
    refs = {i for e in selected for v in e["values"] for i in v["reference_event_ids"]}
    links = [edge for c in chunks for edge in c["links"]]
    refs.update(i for edge in links for i in (edge["source"], edge["target"]))
    available = {e["id"]: e for e in originals}
    if not refs - current <= set(available):
        raise ValueError("Memory references an unavailable frozen context observation")
    return {
        "selected_events": selected,
        "reference_events": [available[i] for i in sorted(refs - current)],
        "links": links,
        "caution": "Model observations, not truth. Unknown/inapplicable cues are not zero highlight value.",
    }


def prepare(source, root, repo, *, robust=False):
    if not GPU_UUIDS:
        raise ValueError("Set REPRO_GPU_UUIDS before preparing a GPU experiment")
    if root.exists():
        raise FileExistsError("Use a new directory; previous runs are retained")
    old, old_id = verify_pilot(source, include_labels=True)
    rows = validate_manifest(read_jsonl(source / "data/manifests/pilot.jsonl"))
    if len(rows) != 20:
        raise ValueError("Expected the original twenty queries")
    root.mkdir(parents=True)
    origins = {"protocol.json": file_sha256(source / "protocol.json")}
    active = []
    for row in rows:
        qid = str(row["qid"])
        previous = read(source / "queries" / (qid + ".json"))
        baseline = read(source / "baseline" / (qid + ".json"))
        for name in ("queries/" + qid + ".json", "baseline/" + qid + ".json"):
            origins[name] = file_sha256(source / name)
        selected = previous.get("selected_windows", [])
        bins = previous.get("selected_bin_ids", [])
        originals = []
        chunks = []
        if selected:
            active.append(row["qid"])
            for path in sorted((source / "memories" / row["vid"]).glob("*.json")):
                c = read(path)
                chunks.append(c)
                originals.extend(c["events"])
                origins[str(path.relative_to(source))] = file_sha256(path)
            by_window = {c["window_id"]: c for c in chunks}
            if bins != sorted(i for w in selected for i in by_window[w]["bin_ids"]):
                raise ValueError("Frozen candidate/bin alignment changed")
            load_frames([baseline["frames"][i] for i in bins])
        windows = []
        for c in chunks:
            if c["window_id"] in selected:
                context = [
                    e
                    for prev in chunks
                    if c["window_id"] - 2 <= prev["window_id"] < c["window_id"]
                    for e in prev["events"]
                ]
                windows.append(
                    {**{k: c[k] for k in ("window_id", "start", "end", "bin_ids")}, "context": context}
                )
        value = {
            "row": row,
            "D0": {k: baseline["D0"][k] for k in ("status", "pred_saliency_scores")},
            "E1_original": previous["E1"],
            "selected_windows": selected,
            "selected_bin_ids": bins,
            "frames": [baseline["frames"][i] for i in bins],
            "windows": windows,
            "original_observations": originals,
            "inherited_errors": previous["errors"],
        }
        write_json(root / "inputs" / (qid + ".json"), value)
    # Full local weight hashing; no new model, data, or dependency download.
    audit = old["model_audit"]["vlm"]
    for name, expected in audit["files"].items():
        path = Path(old["vlm"]["weight_path"]) / name
        if path.stat().st_size != expected["size"] or file_sha256(path) != expected["sha256"]:
            raise ValueError("Frozen VLM weight drift: " + name)
    source_hashes = {}
    for folder in ("src", "experiments/hd_adapter", "experiments/highlight_evidence"):
        for path in sorted((repo / folder).rglob("*.py")):
            name = str(path.relative_to(repo))
            target = root / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            source_hashes[name] = file_sha256(path)
    shutil.copytree(source / "evaluator", root / "evaluator", ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("data/manifests/pilot.jsonl", "data/ground_truth/pilot.jsonl"):
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)
        origins[name] = file_sha256(source / name)
    settings = {
        "source_run": str(source),
        "source_protocol_id": old_id,
        "source_artifact_sha256": origins,
        "vlm": old["vlm"],
        "model_audit": audit,
        "values": old["values"],
        "endpoint": "http://127.0.0.1:8039/v1",
        "workers": 4,
        "server": {**old["server"], "port": 8039, "gpu_uuids": GPU_UUIDS},
        "arms": {"C10": "one observation per frozen 10s input window", "F2": "one observation per 2s bin"},
        "control": "same frames/order, prior2 frozen coarse observations, schema, score prompt, budgets and model",
        "active_qids": active,
        "inherited_no_candidate_qids": [r["qid"] for r in rows if r["qid"] not in active],
        "memory_calls_per_active_query_per_arm": 3,
        "memory_max_tokens_both_arms": 8192,
        "failure": "retain D0 for a failed arm; no candidate changes or selective reruns",
        "historical_reference": "E1_original is descriptive; C10 vs F2 is the matched granularity comparison",
        "scope": "selected-window E1 only; no retrieval, E2/E3, source views, or automatic expansion",
        "max_wall_seconds": 5400,
        "source_sha256": source_hashes,
        "input_sha256": {
            str(p.relative_to(root)): file_sha256(p)
            for folder in ("inputs", "data", "evaluator")
            for p in (root / folder).rglob("*")
            if p.is_file()
        },
        "packages": {n: importlib.metadata.version(n) for n in ("vllm", "torch", "transformers", "numpy")},
    }
    if robust:
        settings.update(
            memory_contract="structured_fields_v2",
            failure="invalid cue after one field repair becomes unknown with trace; failed window retains D0; "
            "E1 receives all fixed candidate frames/bins, updates only successful windows; E1 failure retains D0",
            control="same pixels/order, prior2 frozen coarse observations, structured contract, score prompt, "
            "budgets, failure policy and model in both arms",
        )
        settings["server"]["structured_outputs_config"] = {
            "backend": "xgrammar",
            "disable_any_whitespace": True,
        }
    write_json(root / "protocol.json", {"protocol_id": digest(settings), "settings": settings})
    write_json(
        root / "prepared.json",
        {"at": time.time(), "queries": len(rows), "active": len(active), "weights_verified": True},
    )
    return {"root": str(root), "protocol_id": digest(settings), "active": len(active)}


def verify(root, labels=False):
    value = read(root / "protocol.json")
    c = value["settings"]
    if digest(c) != value["protocol_id"]:
        raise ValueError("Protocol changed")
    for name, expected in c["source_sha256"].items():
        if file_sha256(root / "source" / name) != expected:
            raise ValueError("Frozen source changed: " + name)
    for name, expected in c["input_sha256"].items():
        if not labels and name.startswith("data/ground_truth/"):
            continue
        if file_sha256(root / name) != expected:
            raise ValueError("Frozen input changed: " + name)
    return c, value["protocol_id"]


def run_query(root, c, protocol_id, row):
    qid = str(row["qid"])
    path = root / "queries" / (qid + ".json")
    with locked(root / "locks" / (qid + ".lock")):
        if path.exists():
            return read(path)
        data = read(root / "inputs" / (qid + ".json"))
        started = time.time()
        result = {
            **{k: row[k] for k in ("qid", "vid", "query")},
            "protocol_id": protocol_id,
            "D0": data["D0"],
            "E1_original": data["E1_original"],
            "selected_windows": data["selected_windows"],
            "selected_bin_ids": data["selected_bin_ids"],
        }
        frames = load_frames(data["frames"]) if data["selected_windows"] else []
        by_id = {f["clip_id"]: f for f in frames}
        order = ARMS if int(digest([42, row["qid"]]), 16) % 2 == 0 else tuple(reversed(ARMS))
        result["arm_execution_order"] = list(order)
        for arm in order:
            arm_path = root / "arms" / arm / (qid + ".json")
            if arm_path.exists():
                result[arm] = read(arm_path)
                continue
            out = {
                "status": "inherited_no_candidates",
                "pred_saliency_scores": data["D0"]["pred_saliency_scores"],
                "errors": [],
                "requests": [],
            }
            tick = time.time()
            if data["selected_windows"]:
                robust = c.get("memory_contract") == "structured_fields_v2"
                if robust:
                    from .robust_client import MemoryClient

                    client_type = MemoryClient
                else:
                    client_type = Client
                client = client_type(c["endpoint"], c["vlm"], root / "requests", protocol_id)
                phase = "memory"
                try:
                    chunks = []
                    failed_windows = []
                    for window in data["windows"]:
                        memory_path = root / "memories" / arm / row["vid"] / f"{window['window_id']:03}.json"
                        try:
                            if memory_path.exists():
                                chunk = read(memory_path)
                                if robust:
                                    client.events.append(
                                        {
                                            "request_id": chunk["request_id"],
                                            "cache_hit": True,
                                            "stage": arm + "_memory",
                                        }
                                    )
                            else:
                                chunk = extract_units(
                                    client,
                                    [by_id[i] for i in window["bin_ids"]],
                                    c["values"],
                                    row["duration"],
                                    window,
                                    arm,
                                    window["context"],
                                    robust=robust,
                                )
                                write_json(memory_path, chunk)
                        except (RuntimeError, ValueError, TypeError, KeyError, OSError) as error:
                            if not robust:
                                raise
                            failed = {
                                "phase": "memory",
                                "window_id": window["window_id"],
                                "bin_ids": window["bin_ids"],
                                "error": str(error),
                            }
                            failed_windows.append(failed)
                            write_json(
                                root
                                / "memory_failures"
                                / arm
                                / row["vid"]
                                / f"{window['window_id']:03}.json",
                                failed,
                            )
                            continue
                        chunks.append(chunk)
                    valid_bins = {i for chunk in chunks for i in chunk["bin_ids"]}
                    if robust:
                        out.update(
                            failed_windows=failed_windows,
                            memory_coverage_complete=not failed_windows,
                            successful_window_ids=[x["window_id"] for x in chunks],
                            valid_memory_bin_ids=sorted(valid_bins),
                            fallback_bin_ids=sorted(set(data["selected_bin_ids"]) - valid_bins),
                            downgraded_cues=sum(len(x.get("downgraded_cues", [])) for x in chunks),
                            errors=failed_windows.copy(),
                        )
                    if not chunks:
                        raise RuntimeError("No successful memory windows; retain D0")
                    memory = scoring_memory(chunks, data["original_observations"])
                    out["scoring_memory"] = memory
                    phase = "E1"
                    updates = score_evidence(
                        client, arm + "_E1", row, data["selected_bin_ids"], frames, memory
                    )
                    if robust:
                        updates = [item for item in updates if item["clip_id"] in valid_bins]
                    out.update(
                        status="partial"
                        if failed_windows
                        or any(x["rejected_links"] or x.get("downgraded_cues") for x in chunks)
                        else "ok",
                        pred_saliency_scores=merge_dense_scores(data["D0"]["pred_saliency_scores"], updates),
                    )
                except (RuntimeError, ValueError, TypeError, KeyError, OSError) as error:
                    out.update(
                        status="fallback", errors=out["errors"] + [{"phase": phase, "error": str(error)}]
                    )
                out["requests"] = client.events
            out["wall_seconds"] = time.time() - tick
            write_json(arm_path, out)
            result[arm] = out
        result["wall_seconds"] = time.time() - started
        write_json(path, result)
        return result


def paired(metrics, records, left, right, qids):
    if not qids:
        return None
    sources = sorted({records[q]["vid"].rsplit("_", 2)[0] for q in qids})
    delta = np.array([[metrics[q][left][k] - metrics[q][right][k] for k in ("mAP", "Hit1")] for q in qids])
    sums, counts = np.zeros((len(sources), 2)), np.zeros(len(sources))
    for q, d in zip(qids, delta, strict=True):
        index = sources.index(records[q]["vid"].rsplit("_", 2)[0])
        sums[index] += d
        counts[index] += 1
    samples = np.random.default_rng(42).integers(0, len(sources), (10000, len(sources)))
    ci = np.percentile(sums[samples].sum(1) / counts[samples].sum(1)[:, None], [2.5, 97.5], axis=0)
    return {
        k: {
            "delta_pp": float(delta[:, i].mean()),
            "ci95_pp": ci[:, i].tolist(),
            "wins": int((delta[:, i] > 1e-9).sum()),
            "ties": int((abs(delta[:, i]) <= 1e-9).sum()),
            "losses": int((delta[:, i] < -1e-9).sum()),
        }
        for i, k in enumerate(("mAP", "Hit1"))
    }


def evaluate(root):
    c, protocol_id = verify(root, labels=True)
    truth = read_jsonl(root / "data/ground_truth/pilot.jsonl")
    records = {t["qid"]: read(root / "queries" / (str(t["qid"]) + ".json")) for t in truth}
    sys.path.insert(0, str(root / "evaluator"))
    from standalone_eval.eval import compute_ap_from_tuple, eval_submission, mk_gt_scores

    report = {
        "protocol_id": protocol_id,
        "queries": len(truth),
        "active_queries": len(c["active_qids"]),
        "groups": {},
        "cost": request_cost(root),
    }
    metrics, details = {}, []
    for t in truth:
        q = t["qid"]
        r = records[q]
        binary = (mk_gt_scores(t) >= 4).astype(float)
        metrics[q] = {}
        detail = {"qid": q, "vid": r["vid"], "metrics": metrics[q], "top_ties": {}, "changed_bins": {}}
        for g in GROUPS:
            scores = np.asarray(r[g]["pred_saliency_scores"])
            if len(scores) != len(binary) or not np.isfinite(scores).all():
                raise ValueError("Invalid score coverage")
            metrics[q][g] = {
                "mAP": float(
                    100 * np.mean([compute_ap_from_tuple((0, i, binary[:, i], scores))[2] for i in range(3)])
                ),
                "Hit1": float(100 * binary[int(scores.argmax())].max()),
            }
            changed = np.flatnonzero(scores != np.asarray(r["D0"]["pred_saliency_scores"])).tolist()
            if not set(changed) <= set(r["selected_bin_ids"]):
                raise ValueError("Scores changed outside frozen candidates")
            detail["changed_bins"][g] = changed
            detail["top_ties"][g] = int((scores == scores.max()).sum())
        detail["status"] = {g: r[g]["status"] for g in GROUPS}
        detail["errors"] = {g: r[g]["errors"] for g in ARMS}
        details.append(detail)
    for g in GROUPS:
        predictions = [
            {
                "qid": t["qid"],
                "vid": records[t["qid"]]["vid"],
                "pred_saliency_scores": records[t["qid"]][g]["pred_saliency_scores"],
            }
            for t in truth
        ]
        validate_submission(predictions, truth)
        path = root / (g + "_predictions.jsonl")
        if path.exists():
            if read_jsonl(path) != predictions:
                raise ValueError("Completed predictions changed")
        else:
            write_jsonl(path, predictions)
        official = eval_submission(predictions, truth, verbose=False, match_number=True)
        for name, key in (("mAP", "HL-mAP"), ("Hit1", "HL-Hit1")):
            if (
                round(float(np.mean([m[g][name] for m in metrics.values()])), 2)
                != official["HL-min-VeryGood"][key]
            ):
                raise ValueError("Per-query metric differs from official evaluator")
        report["groups"][g] = {
            "metrics": official,
            "status_counts": dict(Counter(r[g]["status"] for r in records.values())),
            "changed_bins_vs_D0": sum(len(d["changed_bins"][g]) for d in details),
            "top_tied_queries": sum(d["top_ties"][g] > 1 for d in details),
        }
    all_ids = sorted(records)
    common = [q for q in c["active_qids"] if all(records[q][g]["status"] in {"ok", "partial"} for g in ARMS)]
    report["paired"] = {
        "F2_minus_C10_full20": paired(metrics, records, "F2", "C10", all_ids),
        "F2_minus_C10_active": paired(metrics, records, "F2", "C10", c["active_qids"]),
        "F2_minus_C10_common_success": paired(metrics, records, "F2", "C10", common),
        "F2_minus_D0": paired(metrics, records, "F2", "D0", all_ids),
        "C10_minus_D0": paired(metrics, records, "C10", "D0", all_ids),
    }
    report["common_success_qids"] = common
    report["memory"] = {}
    for arm in ARMS:
        chunks = [read(p) for p in (root / "memories" / arm).glob("*/*.json")]
        events = [e for x in chunks for e in x["events"]]
        report["memory"][arm] = {
            "successful_windows": len(chunks),
            "nodes": len(events),
            "mean_support_seconds": float(np.mean([e["end"] - e["start"] for e in events]))
            if events
            else None,
            "accepted_links": sum(len(x["links"]) for x in chunks),
            "rejected_links": sum(len(x["rejected_links"]) for x in chunks),
            "cue_statuses": dict(
                Counter(v["dimension"] + ":" + v["status"] for e in events for v in e["values"])
            ),
        }
        if c.get("memory_contract"):
            report["memory"][arm].update(
                downgraded_cues=sum(len(x.get("downgraded_cues", [])) for x in chunks),
                windows_with_field_retry=sum(x["repair_attempted"] for x in chunks),
                window_quality_counts=dict(Counter(x["quality"] for x in chunks)),
                queries_with_complete_memory=sum(
                    bool(r[arm].get("memory_coverage_complete")) for r in records.values()
                ),
            )
    report["limitations"] = [
        "Explored training20; only14 had frozen candidates; six inherited fallbacks remain in full denominator",
        "C10/F2 use a shared new fixed-unit schema; original E1 is a historical reference, not the matched control",
        "Only selected-window memory is rebuilt; prior2 observations remain frozen coarse context in BOTH arms",
        "Observation supports are grid cells, not verified true event boundaries; sampling remains one frame/2s",
        "More nodes and tokens are intrinsic to finer representation; compute tokens are measured, not matched",
        "Bootstrap uses 10000 source-video cluster resamples, seed42; excludes systematic model bias",
        "No new source frames, retrieval reranking, E2/E3, held-out evaluation, or automatic expansion",
    ]
    if c.get("memory_contract"):
        report["limitations"].append(
            "v2 is a combined engineering repair (structured decoding, metadata, targeted retry, window fallback); "
            "before/after does not isolate any one change. Unknown downgrades remain explicit, not verified facts."
        )
    write_json(root / "per_query.json", details)
    write_json(root / "report.json", report)
    return report


def run(root):
    c, protocol_id = verify(root)
    if (root / "circuit_breaker.json").exists():
        raise RuntimeError("Engineering gate failed; inspect preserved attempts")
    served = Client(c["endpoint"], c["vlm"], root / "requests", protocol_id).check_model()
    rows = validate_manifest(read_jsonl(root / "data/manifests/pilot.jsonl"))
    started = time.time()
    session = {
        "started_at": started,
        "model": served,
        "workers": c["workers"],
        "gpu_start": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv"], text=True
        ),
    }
    session_path = root / "sessions" / (str(time.time_ns()) + ".json")
    write_json(session_path, session)
    completed = []
    for batch in (rows[:4], rows[4:]):
        with ThreadPoolExecutor(max_workers=c["workers"]) as pool:
            futures = {pool.submit(run_query, root, c, protocol_id, row): row for row in batch}
            while futures:
                done, _ = wait(futures, timeout=15, return_when=FIRST_COMPLETED)
                for future in done:
                    completed.append(future.result())
                    futures.pop(future)
                progress = {
                    "status": "running",
                    "completed": len(completed),
                    "total": len(rows),
                    "active_qids": [r["qid"] for r in futures.values()],
                    "updated_at": time.time(),
                    "elapsed_seconds": time.time() - started,
                    "statuses": {g: dict(Counter(r[g]["status"] for r in completed)) for g in ARMS},
                }
                write_json(root / "progress.json", progress)
                if done:
                    print(json.dumps(progress), flush=True)
                if time.time() - started > c["max_wall_seconds"]:
                    raise RuntimeError("Wall budget exhausted")
        gate = [r for r in completed if r["selected_windows"]]
        if len(completed) == 4 and not any(
            all(r[g]["status"] in {"ok", "partial"} for g in ARMS) for r in gate
        ):
            write_json(
                root / "circuit_breaker.json",
                {"reason": "No successful matched pair in first4", "at": time.time()},
            )
            raise RuntimeError("First4 engineering gate failed")
    report = evaluate(root)
    session.update(status="complete", finished_at=time.time())
    write_json(session_path, session)
    write_json(
        root / "progress.json",
        {
            "status": "complete",
            "completed": len(rows),
            "total": len(rows),
            "elapsed_seconds": time.time() - started,
            "updated_at": time.time(),
        },
    )
    return {g: report["groups"][g]["metrics"]["HL-min-VeryGood"] for g in GROUPS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "evaluate"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--robust", action="store_true", help="Freeze structured memory and window fallback v2"
    )
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == "prepare":
        result = prepare(args.source.resolve(), root, args.repo.resolve(), robust=args.robust)
    elif args.action == "evaluate":
        result = evaluate(root)
    else:
        with locked(root / "runner.lock", blocking=False):
            try:
                result = run(root)
            except Exception as error:
                progress = read(root / "progress.json") if (root / "progress.json").exists() else {}
                write_json(
                    root / "progress.json",
                    {
                        **progress,
                        "status": "stopped_with_error",
                        "error": str(error),
                        "updated_at": time.time(),
                    },
                )
                raise
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
