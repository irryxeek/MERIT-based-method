"""Frozen query-exposure ablation: same 28 observed videos, neutral observation question."""

import argparse
import fcntl
import os
import time
from datetime import datetime
from pathlib import Path

from backend import MODEL, Backend
from budget import BudgetStop
from dispatch import environment, idle_gpu
from io_utils import ROOT, compact, legacy, parents, read_json, sha256, utc_now, write_json
from memory import FactStore
from prompts import CAPS, OBSERVE
from r2_observer import read_frames, source_timeline
from r2_protocol import SharedLedger
from schema import ObservationFacts

NEUTRAL_QUESTION = (
    "What people, objects, actions, scene changes, and visible outcomes are directly shown in these frames? "
    "Describe only what is visible, without guessing an intended event."
)


def observation_prompt(trace):
    return OBSERVE.format(
        question=NEUTRAL_QUESTION,
        frames=compact([{"id": f["id"], "pts": f["pts"]} for f in trace["frames"]]),
    )


class TimedLedger(SharedLedger):
    def __init__(self, directory, limits, start, previous_seconds, maximum):
        super().__init__(directory, limits)
        self.start, self.previous_seconds, self.maximum = start, previous_seconds, maximum

    def reserve(self, *args, **kwargs):
        if time.monotonic() - self.start + self.previous_seconds >= self.maximum - 60:
            raise BudgetStop("Observation ablation GPU-time cap, including shutdown reserve")
        return super().reserve(*args, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    with (run / "worker.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        if (run / "complete.json").exists() or (run / "stopped.json").exists():
            return
        manifest = read_json(run / "manifest.json")
        if time.time() >= datetime.fromisoformat(manifest["queue_expires_at"]).timestamp():
            write_json(run / "stopped.json", {"status": "stopped", "reason": "Queue expired", "at": utc_now()})
            return
        for name, expected in manifest["source_sha256"].items():
            if sha256(ROOT / name) != expected:
                raise ValueError("Frozen observation ablation source drift")
        for name, expected in manifest["parent_receipts"].items():
            if sha256(name) != expected:
                raise ValueError("Frozen observation ablation parent drift")
        model = read_json(Path(manifest["parent_run"]) / "model_verification.json")
        for name, expected in model["stats"].items():
            stat = (Path(MODEL) / name).stat()
            if stat.st_size != expected["size"] or stat.st_mtime_ns != expected["mtime_ns"]:
                raise ValueError("Model drift")
        gpu, _ = idle_gpu("all")
        if gpu is None or idle_gpu([gpu["uuid"]])[0] is None:
            write_json(run / "status.json", {"status": "waiting_for_idle_gpu", "at": utc_now()})
            raise SystemExit(75)
        os.environ.update(environment(gpu))
        attempts_path = run / "worker_attempts.json"
        attempts = read_json(attempts_path) if attempts_path.exists() else []
        if any(a["status"] == "running" for a in attempts):
            raise RuntimeError("Interrupted worker needs elapsed-time reconciliation before resume")
        if len(attempts) >= manifest["maximum_worker_launches"]:
            raise BudgetStop("Worker launch cap")
        previous_seconds = sum(a["seconds"] for a in attempts)
        started = time.monotonic()
        attempt = {"status": "running", "gpu": gpu, "started_at": utc_now(), "pid": os.getpid(), "seconds": 0}
        attempts.append(attempt)
        write_json(attempts_path, attempts)
        try:
            import torch

            torch.set_num_threads(2)
            actual = "GPU-" + str(torch.cuda.get_device_properties(0).uuid).removeprefix("GPU-")
            if torch.cuda.device_count() != 1 or actual != gpu["uuid"]:
                raise ValueError("GPU UUID mismatch")
            directory = run / "probe"
            ledger = TimedLedger(directory, manifest["limits"], started, previous_seconds, manifest["maximum_gpu_seconds"])
            backend = Backend(directory, ledger)
            parent = Path(manifest["parent_run"])
            binding = read_json(parent / "binding.json")
            rows = {r["qid"]: r for r in legacy.read_jsonl(parent / "inference.jsonl")}
            completed = []
            for qid in manifest["observation_qids"]:
                row = rows[qid]
                baseline, _ = parents(binding, row)
                if sha256(row["video_path"]) != baseline["media"]["source_sha256"]:
                    raise ValueError("Source video bytes changed")
                original = read_json(parent / "R2/records" / f"{qid}.json")
                reader, timestamps, _ = source_timeline(row, baseline)
                for arm in ("E_uniform", "E_targeted"):
                    for number, old in enumerate(original["observations"][arm]):
                        ids = [f["frame_index"] for f in old["trace"]["frames"]]
                        video, trace = read_frames(
                            reader, timestamps, ids, row, old["action"], baseline["media"]["resized_height_width"], arm
                        )
                        if trace != old["trace"]:
                            raise ValueError("Observation source/resized pixels or timestamps changed")
                        refs = [f["id"] for f in trace["frames"]]

                        def validate(value, refs=refs):
                            ObservationFacts.model_validate(value)
                            FactStore().extract("observation", value, refs)
                            return value

                        key = f"{qid}/{arm}/neutral_observe/{number}"
                        value, raw = backend.call(
                            key, observation_prompt(trace), [video], [trace], ObservationFacts,
                            CAPS["observation"], [arm], validate,
                            references={"ObservationFact.frames": refs},
                        )
                        record = {
                            "qid": qid, "arm": arm, "number": number, "action": old["action"], "trace": trace,
                            "model_question": NEUTRAL_QUESTION, "original_question_not_shown": old["action"]["question"],
                            "result": value, "response_path": raw["response_path"],
                            "reference_response_path": old["response_path"], "attempt": raw["attempt"],
                        }
                        write_json(run / "observations" / f"{qid}_{arm}_{number}.json", record)
                        completed.append(key)
                        write_json(run / "status.json", {"status": "running", "completed_observations": len(completed), "expected": 28, "at": utc_now(), "gpu": gpu})
                        attempt["seconds"] = time.monotonic() - started
                        write_json(attempts_path, attempts)
                        print(f"Completed {len(completed)}/28: {key}", flush=True)
                del reader
            state = {
                "status": "complete", "completed_observations": len(completed), "at": utc_now(),
                "new_calls": len(ledger.state["attempts"]), "actual_tokens": ledger.cost(),
                "manifest_sha256": sha256(run / "manifest.json"), "ledger_sha256": sha256(ledger.path),
                "source_pixels_matched": True, "accuracy_evaluated": False,
                "semantic_review": "pending_assistant_review_not_independent_human_review",
                "comparisons_rerun": False,
            }
            write_json(run / "complete.json", state)
            write_json(run / "status.json", state)
            attempt["status"] = "complete"
        except Exception as error:
            state = {"status": "stopped", "at": utc_now(), "error": f"{type(error).__name__}: {error}"}
            write_json(run / "stopped.json", state)
            write_json(run / "status.json", state)
            attempt["status"] = "failed"
            raise
        finally:
            attempt.update(seconds=time.monotonic() - started, finished_at=utc_now())
            write_json(attempts_path, attempts)


if __name__ == "__main__":
    main()
