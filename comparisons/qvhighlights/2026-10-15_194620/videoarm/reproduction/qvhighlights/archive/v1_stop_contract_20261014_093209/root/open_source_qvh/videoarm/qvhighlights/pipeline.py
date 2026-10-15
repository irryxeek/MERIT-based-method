from __future__ import annotations

import json
import base64
import sqlite3
import time
from pathlib import Path

from . import prompts
from .common import atomic_json, digest, finite
from .contracts import clip_scores, windows
from .media import mapping, visual_message
from .state import BudgetExceeded, reserve, semantic_identity


def parse_events(p, clip=False, duration=None, start=0, end=None):
    keys = (
        {"events", "uncertainty", "candidates"} if clip else {"events", "uncertainty"}
    )
    if not isinstance(p, dict) or set(p) != keys:
        raise ValueError("event evidence schema mismatch")
    for key in ("events", "uncertainty"):
        if not isinstance(p[key], str) or len(p[key]) > 2000:
            raise ValueError("event evidence string invalid")
    if clip:
        p["candidates"] = windows(p["candidates"], duration, canonicalize=True)
        if any(w[0] < start or w[1] > end for w in p["candidates"]):
            raise ValueError("local candidate outside observed interval")
    return p


def parse_action(p, duration, task):
    if not isinstance(p, dict) or set(p) != {"action", "start", "end", "question"}:
        raise ValueError("controller action schema mismatch")
    if p["action"] not in {"scene_snapper", "clip_analyzer", "stop"}:
        raise ValueError("unknown tool")
    if not isinstance(p["question"], str) or len(p["question"]) > 1000:
        raise ValueError("invalid tool question")
    start, end = finite(p["start"]), finite(p["end"])
    if p["action"] == "stop":
        if start != 0 or end != 0:
            raise ValueError("stop must have zero bounds")
    elif not 0 <= start < end <= duration:
        raise ValueError("invalid tool time range")
    elif task == "hd" and p["action"] == "clip_analyzer":
        ids = intersecting_ids(start, end, int(duration / 2))
        if not ids or len(ids) > 24:
            raise ValueError("HD local clip range must contain 1..24 clips")
    return p


def intersecting_ids(start, end, n):
    return [i for i in range(n) if 2 * i < end and 2 * (i + 1) > start]


def memory_view(state, method):
    records = state.get("memory", [])
    if method == "A1":
        records = records[-1:]
    # Fixed extractive summary; full immutable records are checkpointed and traced.
    summary = []
    for r in records:
        p = r["observation"]
        summary.append(
            {
                "id": r["id"],
                "tier": r["tool"],
                "interval": [r["start"], r["end"]],
                "events": p.get("events", "")[:256],
                "uncertainty": p.get("uncertainty", "")[:128],
                "candidates": p.get("candidates", []),
                "clip_scores": p.get("clip_scores", []),
            }
        )
    objectives = state.get("actions", [])
    if method == "A1":
        objectives = objectives[-1:]
    return {
        "sensory": {
            "base_pts": [f["pts"] for f in state.get("initial_frames", [])],
            "current_focus": [
                {
                    "interval": [r["start"], r["end"]],
                    "pts": [f["pts"] for f in r["frames"]],
                }
                for r in records[-1:]
            ],
        },
        "result": summary,
        "working_objectives": [
            {
                "tool": a["action"],
                "interval": [a["start"], a["end"]],
                "objective": a["question"][:128],
            }
            for a in objectives
        ],
    }


class Engine:
    def __init__(self, row, config, store, backend, reader, info):
        self.row, self.config, self.store, self.backend = row, config, store, backend
        self.reader, self.info = reader, info
        self.qid = row["qid"]
        item = store.item(self.qid)
        self.state, self.budget = item["snapshot"], item["budget"]
        self.limits = config["limits"]
        self.frames_dir = store.directory / "frames" / str(self.qid)
        self.trace_dir = store.directory / "traces" / str(self.qid)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.started = time.monotonic()
        self.local_elapsed = 0
        self.shared = sqlite3.connect(
            store.directory.parent.parent / "initial_evidence_cache.sqlite",
            isolation_level=None,
        )
        self.shared.execute("PRAGMA journal_mode=WAL")
        self.shared.execute(
            "CREATE TABLE IF NOT EXISTS initial_requests(identity TEXT PRIMARY KEY, response TEXT, origin TEXT)"
        )

    def save(self):
        now = time.monotonic()
        self.budget["active_seconds"] += now - self.started - self.local_elapsed
        self.local_elapsed = now - self.started
        self.store.checkpoint(
            self.qid, snapshot=self.state, budget=self.budget, status="running"
        )
        atomic_json(
            self.trace_dir / "checkpoint.json",
            {"state": self.state, "budget": self.budget},
        )

    def sample(self, start, end, count=24, clip_ids=None):
        return self.reader.sample(
            start, end, count, clip_ids=clip_ids, output_dir=self.frames_dir
        )

    def current_images(self, initial):
        latest = self.state.get("memory", [])[-1:]
        if not latest:
            return initial
        saved = latest[0]["frames"]
        count = min(12, len(saved))
        local = []
        for i in range(count):
            f = saved[min(len(saved) - 1, int((i + 0.5) * len(saved) / count))]
            encoded = base64.b64encode(Path(f["path"]).read_bytes()).decode()
            if digest(encoded) != f["jpeg_sha256"]:
                raise ValueError("current visual evidence hash drift")
            local.append({**f, "url": "data:image/jpeg;base64," + encoded})
        return [{**f, "position": i} for i, f in enumerate(initial[::2][:12] + local)]

    def initial_view(self):
        initial = self.state["initial"]
        if self.config["task"] == "mr":
            return initial
        return [
            {"clip_scores": p["clip_scores"], "events": p["events"][:256]}
            for p in initial
        ]

    def request(self, stage, role, instruction, frames, parser, *, final=False):
        context = {
            "qid": self.qid,
            "query": self.row["query"],
            "vid": self.row["vid"],
            "duration": self.row["duration"],
            "task": self.config["task"],
            "available_media_duration": self.info["video_duration"],
            "mapping_audit": self.info.get("mapping_audit"),
        }
        text = json.dumps(context, ensure_ascii=False) + "\n" + instruction
        messages = [
            {"role": "system", "content": prompts.SYSTEM},
            visual_message(text, frames),
        ]
        max_output = self.limits[role + "_output"]
        rows = self.store.db.execute(
            "SELECT * FROM requests WHERE qid=? AND stage=? ORDER BY attempt",
            (self.qid, stage),
        ).fetchall()
        if rows:
            original = json.loads(rows[0]["request"])
            text = original["text"]
            frames = []
            for saved in original["frames"]:
                encoded = base64.b64encode(Path(saved["path"]).read_bytes()).decode()
                if digest(encoded) != saved["jpeg_sha256"]:
                    raise ValueError("saved request pixel hash drift")
                frames.append({**saved, "url": "data:image/jpeg;base64," + encoded})
            messages = [
                {"role": "system", "content": original["system"]},
                visual_message(original["text"], frames),
            ]
        cache_config = self.config
        if stage == "overview" or stage.startswith("initialize_"):
            # These local-refinement parameters do not affect initial pixels or messages.
            cache_config = {
                **self.config,
                "sampling": {
                    k: v
                    for k, v in self.config["sampling"].items()
                    if k not in {"per_tool_frames", "uniform_window_fraction"}
                },
            }
        key = semantic_identity(
            self.row, cache_config, self.info, messages, role, frames
        )
        last_error = "request not attempted"
        for attempt in range(2):
            previous = next((r for r in rows if r["attempt"] == attempt), None)
            if previous:
                if previous["identity"] != key:
                    raise ValueError("resumed request identity mismatch")
                if previous["status"] == "completed":
                    result = json.loads(previous["response"])
                    try:
                        if result["finish_reason"] != "stop":
                            raise ValueError("truncated/nonterminal model response")
                        parsed = parser(json.loads(result["raw"]))
                        self.budget["cache_hits"] += 1
                        self.save()
                        return parsed
                    except (ValueError, TypeError, KeyError) as e:
                        last_error = str(e)
                        continue
                last_error = previous["status"]
                continue
            self.save()
            new, input_reserved = reserve(
                self.budget,
                self.limits,
                frames=frames,
                output=max_output,
                controller=role == "controller",
                final=final,
            )
            if attempt:
                new["retries"] += 1
            # Store a replayable text + pixel mapping, not megabytes of encoded pixels.
            request_record = {
                "system": messages[0]["content"],
                "text": text,
                "frames": mapping(frames),
                "max_tokens": max_output,
                "seed": self.config["seed"],
                "temperature": 0,
            }
            self.store.db.execute("BEGIN IMMEDIATE")
            try:
                self.store.db.execute(
                    "INSERT INTO requests VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        self.qid,
                        stage,
                        attempt,
                        key,
                        "inflight",
                        json.dumps(request_record),
                        None,
                        input_reserved,
                        max_output,
                        len(frames),
                        None,
                    ),
                )
                self.store.checkpoint(self.qid, budget=new)
                self.store.db.execute("COMMIT")
            except BaseException:
                self.store.db.execute("ROLLBACK")
                raise
            self.budget = new
            started = time.monotonic()
            initial_stage = stage == "overview" or stage.startswith("initialize_")
            cached = (
                self.shared.execute(
                    "SELECT response,origin FROM initial_requests WHERE identity=?",
                    (key,),
                ).fetchone()
                if initial_stage
                else None
            )
            try:
                if cached:
                    result = json.loads(cached[0])
                    self.budget["shared_cache_hits"] += 1
                    request_record["shared_origin"] = cached[1]
                else:
                    self.budget["actual_requests"] += 1
                    self.budget["actual_images"] += len(frames)
                    self.store.checkpoint(self.qid, budget=self.budget)
                    result = self.backend.complete(
                        messages,
                        max_output,
                        self.limits["active_seconds"] - self.budget["active_seconds"],
                    )
                    result["model_elapsed_seconds"] = time.monotonic() - started
            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                self.store.db.execute(
                    "UPDATE requests SET status='error',response=?,elapsed=? WHERE qid=? AND stage=? AND attempt=?",
                    (
                        json.dumps({"error": last_error}),
                        time.monotonic() - started,
                        self.qid,
                        stage,
                        attempt,
                    ),
                )
                self.save()  # Keep conservative reservations for uncertain server failures.
                continue
            usage = result["usage"]
            self.budget["input_tokens"] += usage["prompt_tokens"] - input_reserved
            self.budget["output_tokens"] += usage["completion_tokens"] - max_output
            if not cached:
                self.budget["actual_input_tokens"] += usage["prompt_tokens"]
                self.budget["actual_output_tokens"] += usage["completion_tokens"]
            self.store.db.execute("BEGIN IMMEDIATE")
            self.store.db.execute(
                "UPDATE requests SET status='completed',response=?,request=?,elapsed=? WHERE qid=? AND stage=? AND attempt=?",
                (
                    json.dumps(result),
                    json.dumps(request_record),
                    time.monotonic() - started,
                    self.qid,
                    stage,
                    attempt,
                ),
            )
            self.store.checkpoint(self.qid, budget=self.budget)
            self.store.db.execute("COMMIT")
            atomic_json(
                self.trace_dir / f"{stage}_{attempt}.json",
                {"request": request_record, "response": result},
            )
            self.save()
            if (
                usage["prompt_tokens"] > input_reserved
                or usage["completion_tokens"] > max_output
            ):
                raise BudgetExceeded("backend exceeded reserved context/output tokens")
            try:
                if result["finish_reason"] != "stop":
                    raise ValueError("truncated/nonterminal model response")
                parsed = parser(json.loads(result["raw"]))
                result["parse_status"] = "succeeded"
                result["parsed"] = parsed
                self.store.db.execute(
                    "UPDATE requests SET response=? WHERE qid=? AND stage=? AND attempt=?",
                    (json.dumps(result), self.qid, stage, attempt),
                )
                if initial_stage and not cached:
                    self.shared.execute(
                        "INSERT OR IGNORE INTO initial_requests VALUES(?,?,?)",
                        (key, json.dumps(result), str(self.store.directory)),
                    )
                atomic_json(self.trace_dir / f"{stage}_{attempt}_parsed.json", parsed)
                return parsed
            except (ValueError, TypeError, KeyError) as e:
                last_error = str(e)
                result["parse_status"] = "failed"
                result["parse_error"] = last_error
                self.store.db.execute(
                    "UPDATE requests SET response=? WHERE qid=? AND stage=? AND attempt=?",
                    (json.dumps(result), self.qid, stage, attempt),
                )
                atomic_json(
                    self.trace_dir / f"{stage}_{attempt}_parse_error.json",
                    {"parse_status": "failed", "error": last_error},
                )
        raise ValueError(f"{stage} invalid after at most one retry: {last_error}")

    def initialize(self):
        if self.config["task"] == "mr":
            frames = self.sample(0, self.row["duration"])
            if "initial" not in self.state:
                self.state["initial"] = self.request(
                    "overview", "scene", prompts.OVERVIEW, frames, parse_events
                )
                self.state["initial_frames"] = mapping(frames)
                self.save()
            return frames
        n = self.row["num_clips"]
        self.state.setdefault("initial_batches", [])
        for start in range(0, n, 24):
            if start // 24 < len(self.state["initial_batches"]):
                continue
            ids = list(range(start, min(start + 24, n)))
            frames = self.sample(2 * ids[0], 2 * (ids[-1] + 1), clip_ids=ids)

            def parser(p):
                clip_scores(p, ids)
                return p

            p = self.request(
                f"initialize_{start}",
                "clip",
                prompts.HD + f"\nExpected IDs: {ids}",
                frames,
                parser,
            )
            self.state["initial_batches"].append(p)
            self.save()
        if "initial" not in self.state:
            self.state["initial"] = self.state["initial_batches"]
            self.state["scores"] = [
                s["score"]
                for p in self.state["initial_batches"]
                for s in p["clip_scores"]
            ]
            if len(self.state["scores"]) != n:
                raise ValueError("incomplete HD initialization")
            self.save()
        count = min(24, n)
        if count == 0:
            raise ValueError("HD duration contains no official clips")
        ids = [min(n - 1, int((i + 0.5) * n / count)) for i in range(count)]
        frames = self.sample(0, self.row["duration"], clip_ids=ids)
        self.state["initial_frames"] = mapping(frames)
        self.save()
        return frames

    def observe(self, action, index):
        if any(r["id"] == f"observation_{index}" for r in self.state.get("memory", [])):
            return
        if any(r["index"] == index for r in self.state.get("invalid_tools", [])):
            return
        start, end = action["start"], action["end"]
        tool = action["action"]
        hd_clip = self.config["task"] == "hd" and tool == "clip_analyzer"
        ids = intersecting_ids(start, end, self.row["num_clips"]) if hd_clip else None
        # Intersected HD clips are observed over their COMPLETE clip spans, explicitly traced.
        sample_start, sample_end = (
            (2 * ids[0], 2 * (ids[-1] + 1)) if ids else (start, end)
        )
        allocations = self.state.setdefault("tool_allocations", {})
        allocated = allocations.get(str(index))
        frame_cap = self.config["sampling"].get("per_tool_frames", 24)
        count = allocated or min(
            frame_cap, self.limits["local_frames"] - self.budget["local_frames"]
        )
        if count <= 0 or (
            allocated is None and self.budget["tool_actions"] >= self.limits["tools"]
        ):
            raise BudgetExceeded("local tool budget exhausted")
        if ids and len(ids) > count:
            raise BudgetExceeded(
                "HD request cannot cover every clip within remaining frame budget"
            )
        frames = self.sample(sample_start, sample_end, count, clip_ids=ids)
        if allocated is None:
            allocations[str(index)] = len(frames)
            self.budget["local_frames"] += len(frames)
            self.budget["tool_actions"] += 1
            self.save()
        if hd_clip:

            def parser(p):
                clip_scores(p, ids)
                return p

            prompt, role = prompts.HD + f"\nExpected IDs: {ids}", "clip"
        elif tool == "scene_snapper":
            prompt, role, parser = prompts.SCENE, "scene", parse_events
        else:
            prompt, role = prompts.CLIP, "clip"
            parser = lambda p: parse_events(
                p, clip=True, duration=self.row["duration"], start=start, end=end
            )
        instruction = (
            prompt
            + "\n"
            + json.dumps(
                {
                    "requested_interval": [start, end],
                    "sample_interval": [sample_start, sample_end],
                    "question": action["question"],
                }
            )
        )
        try:
            observation = self.request(
                f"tool_{index}", role, instruction, frames, parser
            )
        except ValueError as e:
            self.state.setdefault("invalid_tools", []).append(
                {"index": index, "action": action, "error": str(e)}
            )
            self.save()
            return
        record = {
            "id": f"observation_{index}",
            "tool": tool,
            "start": start,
            "end": end,
            "sample_interval": [sample_start, sample_end],
            "observation": observation,
            "frames": mapping(frames),
        }
        records = self.state.get("memory", []) + [record]
        self.state["memory"] = (
            records[-1:] if self.config["method"] == "A1" else records
        )
        if hd_clip:
            for row in observation["clip_scores"]:
                self.state["scores"][row["clip_id"]] = row["score"]
        self.save()

    def run(self):
        initial = self.initialize()
        method = self.config["method"]
        self.state.setdefault("actions", [])
        self.state.setdefault("next_action", 0)
        if method == "B1":
            for i in range(self.state["next_action"], 5):
                d = self.row["duration"]
                fraction = self.config["sampling"].get("uniform_window_fraction", 1)
                width = d / 5 * fraction
                center = (i + 0.5) * d / 5
                start, end = center - width / 2, center + width / 2
                if self.config["task"] == "hd":
                    max_clips = self.config["sampling"].get("per_tool_frames", 24)
                    first = max(0, int(start / 2))
                    last = min(
                        self.row["num_clips"],
                        int(end / 2 + 0.999999),
                        first + max_clips,
                    )
                    start, end = 2 * first, 2 * last
                action = {
                    "action": "clip_analyzer",
                    "start": start,
                    "end": end,
                    "question": "Inspect query event persistence, boundaries and highlight ranking.",
                }
                self.observe(action, i)
                self.state["actions"].append(action)
                self.state["next_action"] = i + 1
                self.save()
        elif method in {"M1", "A1"}:
            while self.state["next_action"] < self.limits[
                "tools"
            ] and not self.state.get("stopped"):
                i = self.state["next_action"]
                if self.budget["local_frames"] >= self.limits["local_frames"]:
                    self.state["stop_reason"] = "local_frames_exhausted"
                    self.state["budget_interrupted"] = True
                    break
                instruction = (
                    prompts.CONTROLLER
                    + "\n"
                    + json.dumps(
                        {
                            "initial": self.initial_view(),
                            "HM3": memory_view(self.state, method),
                            "budget": {
                                k: v for k, v in self.budget.items() if k != "frame_ids"
                            },
                            "limits": self.limits,
                            "coverage": self.state["actions"]
                            if method != "A1"
                            else [
                                {"start": a["start"], "end": a["end"]}
                                for a in self.state["actions"]
                            ],
                        },
                        ensure_ascii=False,
                    )
                )
                try:
                    if len(self.state["actions"]) > i:
                        action = self.state["actions"][i]
                    else:
                        action = self.request(
                            f"controller_{i}",
                            "controller",
                            instruction,
                            self.current_images(initial),
                            lambda p: parse_action(
                                p, self.row["duration"], self.config["task"]
                            ),
                        )
                        self.state["actions"].append(action)
                        self.save()
                    if action["action"] == "stop":
                        self.state["stopped"] = True
                        self.state["stop_reason"] = "model_stop"
                    else:
                        self.observe(action, i)
                except BudgetExceeded as e:
                    self.state["stop_reason"] = str(e)
                    self.state["budget_interrupted"] = True
                    break
                self.state["next_action"] = i + 1
                self.save()
            if self.state["next_action"] >= self.limits["tools"] and not self.state.get(
                "stopped"
            ):
                self.state["stop_reason"] = "tool_action_limit"
                self.state["budget_interrupted"] = True
        prediction = {k: self.row[k] for k in ("qid", "query", "vid")}
        if self.config["task"] == "mr":
            instruction = (
                prompts.FINAL
                + "\n"
                + json.dumps(
                    {
                        "initial": self.initial_view(),
                        "HM3": memory_view(self.state, method),
                    },
                    ensure_ascii=False,
                )
            )

            def parser(p):
                if not isinstance(p, dict) or set(p) != {"pred_relevant_windows"}:
                    raise ValueError("MR final output schema mismatch")
                return windows(
                    p["pred_relevant_windows"],
                    self.row["duration"],
                    allow_empty=False,
                    canonicalize=True,
                )

            prediction["pred_relevant_windows"] = self.request(
                "final",
                "final",
                instruction,
                self.current_images(initial),
                parser,
                final=True,
            )
            available = initial + [
                f for r in self.state.get("memory", []) for f in r["frames"]
            ]
            self.state["candidate_evidence_audit"] = [
                {
                    "window": w,
                    "direct_sample_pts": sorted(
                        {f["pts"] for f in available if w[0] <= f["pts"] < w[1]}
                    ),
                    "note": "PTS support diagnostic only; does not establish semantic correctness. A1 uses initial and latest record only.",
                }
                for w in prediction["pred_relevant_windows"]
            ]
        else:
            prediction["pred_saliency_scores"] = self.state["scores"]
        self.save()
        return prediction
