from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from .common import digest, write_jsonl

_UNSET = object()


class BudgetExceeded(RuntimeError):
    pass


class Store:
    def __init__(self, directory, identity, rows):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.directory / "state.sqlite", isolation_level=None, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
          CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT);
          CREATE TABLE IF NOT EXISTS items(qid INTEGER PRIMARY KEY, status TEXT NOT NULL,
            snapshot TEXT NOT NULL, budget TEXT NOT NULL, prediction TEXT, error TEXT, updated REAL);
          CREATE TABLE IF NOT EXISTS requests(qid INTEGER, stage TEXT, attempt INTEGER,
            identity TEXT NOT NULL, status TEXT NOT NULL, request TEXT NOT NULL, response TEXT,
            input_reserved INTEGER, output_reserved INTEGER, images INTEGER, elapsed REAL,
            PRIMARY KEY(qid, stage, attempt));
        """)
        old = self.db.execute(
            "SELECT value FROM metadata WHERE key='identity'"
        ).fetchone()
        if old and json.loads(old[0]) != identity:
            raise ValueError("resume refused: code/config/data/model identity changed")
        if not old:
            self.db.execute(
                "INSERT INTO metadata VALUES('identity',?)",
                (json.dumps(identity, sort_keys=True),),
            )
        for row in rows:
            budget = {
                "requests": 0,
                "controllers": 0,
                "images": 0,
                "frame_ids": [],
                "local_frames": 0,
                "tool_actions": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "active_seconds": 0,
                "cache_hits": 0,
                "retries": 0,
                "uncertain_requests": 0,
                "shared_cache_hits": 0,
                "actual_requests": 0,
                "actual_images": 0,
                "actual_input_tokens": 0,
                "actual_output_tokens": 0,
            }
            self.db.execute(
                "INSERT OR IGNORE INTO items VALUES(?,?,?,?,?,?,?)",
                (
                    row["qid"],
                    "pending",
                    "{}",
                    json.dumps(budget),
                    None,
                    None,
                    time.time(),
                ),
            )

    def item(self, qid):
        r = dict(self.db.execute("SELECT * FROM items WHERE qid=?", (qid,)).fetchone())
        for k in ("snapshot", "budget", "prediction"):
            r[k] = json.loads(r[k]) if r[k] else None
        return r

    def checkpoint(
        self,
        qid,
        snapshot=None,
        budget=None,
        status=None,
        prediction=None,
        error=_UNSET,
    ):
        row = self.item(qid)
        self.db.execute(
            "UPDATE items SET snapshot=?,budget=?,status=?,prediction=?,error=?,updated=? WHERE qid=?",
            (
                json.dumps(
                    snapshot if snapshot is not None else row["snapshot"],
                    allow_nan=False,
                ),
                json.dumps(
                    budget if budget is not None else row["budget"], allow_nan=False
                ),
                status or row["status"],
                json.dumps(prediction)
                if prediction is not None
                else json.dumps(row["prediction"])
                if row["prediction"] is not None
                else None,
                row["error"] if error is _UNSET else error,
                time.time(),
                qid,
            ),
        )

    def export(self):
        rows = [
            json.loads(r[0])
            for r in self.db.execute(
                "SELECT prediction FROM items WHERE status='succeeded' ORDER BY qid"
            )
        ]
        write_jsonl(self.directory / "predictions.jsonl", rows)
        statuses = {
            r["status"]: r["n"]
            for r in self.db.execute(
                "SELECT status,COUNT(*) AS n FROM items GROUP BY status"
            )
        }
        return statuses

    def recover(self):
        # Unknown requests already consumed their full conservative reservation.
        for row in self.db.execute(
            "SELECT qid,stage,attempt FROM requests WHERE status='inflight'"
        ).fetchall():
            item = self.item(row["qid"])
            item["budget"]["uncertain_requests"] += 1
            self.checkpoint(
                row["qid"],
                budget=item["budget"],
                status="paused",
                error="uncertain prior request; reservation retained",
            )
            self.db.execute(
                "UPDATE requests SET status='uncertain' WHERE qid=? AND stage=? AND attempt=?",
                tuple(row),
            )


def reserve(budget, limits, *, frames, output, controller=False, final=False):
    b = json.loads(json.dumps(budget))
    inp = limits["context_tokens"] - output
    terminal = 0 if final else 1
    if len(frames) > 24:
        raise BudgetExceeded("logical frame cap per request exceeded")
    for key, value, cap in (
        ("requests", 1, "requests"),
        ("images", len(frames), "images"),
        ("input_tokens", inp, "input_tokens"),
        ("output_tokens", output, "output_tokens"),
    ):
        margin = {
            "requests": terminal,
            "images": terminal * 24,
            "input_tokens": terminal
            * (limits["context_tokens"] - limits["final_output"]),
            "output_tokens": terminal * limits["final_output"],
        }[key]
        if b[key] + value + margin > limits[cap]:
            raise BudgetExceeded(f"{key} exhausted with terminal reservation")
        b[key] += value
    if controller:
        if b["controllers"] >= limits["controllers"]:
            raise BudgetExceeded("controller limit exhausted")
        b["controllers"] += 1
    if b["active_seconds"] >= limits["active_seconds"]:
        raise BudgetExceeded("active execution time exhausted")
    b["frame_ids"] = sorted(set(b["frame_ids"]) | {f["image_id"] for f in frames})
    return b, inp


def semantic_identity(row, config, info, messages, role, frames):
    return digest(
        {
            "task": config["task"],
            "qid": row["qid"],
            "query_sha256": digest(row["query"]),
            "vid": row["vid"],
            "media": info["media_sha256"],
            "timeline": info["timeline_sha256"],
            "model_revision": config["model_revision"],
            "sampling": config["sampling"],
            "protocol": config["protocol"],
            "role": role,
            "messages": messages,
            "generation": {
                "temperature": 0,
                "seed": config["seed"],
                "max_tokens": config["limits"][role + "_output"],
            },
            "frames": [
                {k: v for k, v in f.items() if k not in {"url", "path"}} for f in frames
            ],
        }
    )
