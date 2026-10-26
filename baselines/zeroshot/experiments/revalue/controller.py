"""Persisted bounded controller primitives for R0; R3's model policy is not enabled."""

from copy import deepcopy

from io_utils import read_json, write_json
from selector import select_events


class Controller:
    def __init__(self, path, initial_events, task):
        self.path = path
        if path.exists():
            self.state = read_json(path)
            if self.state["task"] != task or self.state["initial_events"] != initial_events:
                raise ValueError("Controller resume changed initial inputs")
        else:
            self.state = {
                "task": deepcopy(task),
                "initial_events": deepcopy(initial_events),
                "events": deepcopy(initial_events),
                "observations": [],
                "new_events": 0,
                "rounds": 0,
                "history": [],
                "seen_frames": [],
                "displayed_frames": 0,
            }
            write_json(path, self.state)

    def reserve_observation(self, action_id, frames, cached=False):
        state = self.state
        if len(state["observations"]) >= 4 or not 2 <= len(frames) <= 24 or len(set(frames)) != len(frames):
            raise ValueError("Observation budget exhausted or duplicated source frames")
        if state["displayed_frames"] + len(frames) > 96:
            raise ValueError("Display frame budget exhausted")
        if action_id in [o["id"] for o in state["observations"]]:
            raise ValueError("Repeated action adds no evidence")
        state["observations"].append({"id": action_id, "frames": frames, "cached": cached})
        state["displayed_frames"] += len(frames)  # Cache hits also pay the display budget.
        state["seen_frames"] = sorted(set(state["seen_frames"]) | set(frames))
        write_json(self.path, state)

    def add_event(self, event):
        if self.state["new_events"] >= 2 or not event.get("evidence"):
            raise ValueError("Candidate admission lacks evidence or exceeds budget")
        if event["event_id"] in [e["event_id"] for e in self.state["events"]]:
            raise ValueError("Duplicate event identity")
        self.state["events"].append(deepcopy(event))
        self.state["new_events"] += 1
        write_json(self.path, self.state)

    def decide(self, order, revisions=()):
        if self.state["rounds"] >= 3:
            raise ValueError("Decision round budget exhausted")
        for revision in revisions:
            for event in self.state["events"]:
                if event["event_id"] == revision["event_id"]:
                    event.update(deepcopy(revision))
                    break
            else:
                raise ValueError("Unknown event revision")
        result = select_events(self.state["events"], order, self.state["task"])
        self.state["rounds"] += 1
        self.state["history"].append(result)
        write_json(self.path, self.state)
        return result
