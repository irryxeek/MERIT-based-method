"""Monitor one frozen experiment and recover only its own processes."""

import argparse
import fcntl
import json
import os
import shutil
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

from merit_qvh.cli import REPO, validate_submission
from merit_qvh.core import digest, file_sha256, read_jsonl, write_json


def load_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def process_identity(pid):
    try:
        directory = Path("/proc") / str(pid)
        fields = (directory / "stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return {
            "pid": int(pid),
            "uid": directory.stat().st_uid,
            "pgid": int(fields[2]),
            "start_ticks": int(fields[19]),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        }
    except (OSError, ValueError, IndexError):
        return None


def same_process(identity):
    return bool(identity and process_identity(identity["pid"]) == identity)


def group_members(identity):
    if not same_process(identity):
        return []
    members = []
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            item = process_identity(int(entry.name))
            if item and item["uid"] == identity["uid"] and item["pgid"] == identity["pgid"]:
                members.append(item)
    return members


def stop_owned(record):
    """Never trust a PID alone: UID, boot ID and start time must all match."""
    identity = record.get("process")
    members = {item["pid"]: item for item in record.get("members", [])}
    for item in group_members(identity):
        members[item["pid"]] = item
    if identity:
        members[identity["pid"]] = identity
    members = {
        pid: item for pid, item in members.items() if item["uid"] == os.getuid() and pid != os.getpid()
    }
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid, item in members.items():
            if same_process(item):
                try:
                    os.kill(pid, sig)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + 10
        while any(same_process(item) for item in members.values()) and time.monotonic() < deadline:
            time.sleep(0.2)


def backoff(attempt):
    return min(1800, 60 * 2 ** min(max(attempt - 1, 0), 5))


class Supervisor:
    def __init__(self, config):
        self.config = config
        self.root = Path(config["run_root"])
        self.runtime = self.root / "runtime"
        self.state_path = self.runtime / "automation_status.json"
        self.state = load_json(self.state_path, {})
        self.children = []
        self.state.setdefault("high_water_completed", 0)
        self.state.setdefault("last_progress_at", time.time())
        self.state.setdefault("worker_attempts_without_progress", 0)
        self.state.setdefault("service_attempts_without_progress", 0)
        self.state.setdefault("health_failures", 0)
        self.state["supervisor_pid"] = os.getpid()

    def event(self, action, **details):
        item = {"time": time.time(), "action": action, **details}
        with (self.runtime / "automation_events.jsonl").open("a") as stream:
            stream.write(json.dumps(item) + "\n")
        print(json.dumps(item), flush=True)

    def status(self, phase, **details):
        self.state.update(phase=phase, checked_at=time.time(), **details)
        write_json(self.state_path, self.state)

    def record_path(self, kind):
        return self.runtime / ("active_runner.json" if kind == "worker" else "active_vlm_service.json")

    def active(self, kind):
        path = self.record_path(kind)
        record = load_json(path, {})
        if not record:
            return {}, False
        identity = record.get("process")
        if identity is None:
            identity = process_identity(record["pid"])
            if not identity:
                return record, False
            command = Path(f"/proc/{record['pid']}/cmdline").read_bytes().decode().rstrip("\0").split("\0")
            allowed = [self.config[f"{kind}_command"]]
            if kind == "worker":
                allowed.append(self.config["adopt_worker_command"])
            boot_seconds = next(
                int(line.split()[1])
                for line in Path("/proc/stat").read_text().splitlines()
                if line.startswith("btime ")
            )
            start_seconds = boot_seconds + identity["start_ticks"] / os.sysconf("SC_CLK_TCK")
            if (
                identity["uid"] != os.getuid()
                or identity["pgid"] != identity["pid"]
                or command not in allowed
                or abs(start_seconds - record["started_at"]) > 60
            ):
                raise RuntimeError(f"Refusing to adopt unverified {kind} PID {record['pid']}")
            record["process"] = identity
            self.event("adopt", kind=kind, pid=record["pid"])
        alive = same_process(identity)
        if alive:
            record["members"] = group_members(identity)
            write_json(path, record)
        return record, alive

    def verify_protocol(self):
        protocol = load_json(self.root / "protocol.json")
        if digest(protocol) != self.config["protocol_id"]:
            raise RuntimeError("Frozen protocol changed")
        for name, expected in protocol["source_sha256"].items():
            if file_sha256(REPO / name) != expected:
                raise RuntimeError(f"Frozen source changed: {name}")
        return protocol["config"]

    def health(self, model):
        try:
            with urllib.request.urlopen(self.config["endpoint"] + "/models", timeout=10) as response:
                models = json.load(response)["data"]
            matches = [item for item in models if item["id"] == model["model"]]
            if len(matches) != 1 or matches[0].get("root") != model["weight_path"]:
                return False, "Endpoint model identity differs"
            with urllib.request.urlopen(
                self.config["endpoint"].removesuffix("/v1") + "/health", timeout=10
            ) as response:
                if response.status != 200:
                    return False, "Service health check failed"
            return True, None
        except Exception as error:
            return False, f"{type(error).__name__}: {error}"

    def gpu_ready(self, records, require_embedding_space=False):
        inventory = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid,memory.free", "--format=csv,noheader,nounits"],
            text=True,
            timeout=15,
        )
        devices = {}
        for line in inventory.splitlines():
            index, uuid, free = [part.strip() for part in line.split(",")]
            devices[int(index)] = {"uuid": uuid, "free": int(free)}
        for index, uuid in self.config["gpu_uuids"].items():
            if devices[int(index)]["uuid"] != uuid:
                return False, "GPU UUID mapping changed"
        allowed = {
            item["pid"] for record in records for item in record.get("members", []) if same_process(item)
        }
        applications = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
            text=True,
            timeout=15,
        )
        for line in applications.splitlines():
            uuid, pid = [part.strip() for part in line.split(",")]
            if uuid in self.config["gpu_uuids"].values() and int(pid) not in allowed:
                return False, f"Assigned GPU is occupied by an unmanaged process {pid}"
        if require_embedding_space and devices[4]["free"] < 8500:
            return False, "Waiting for at least 8500 MiB for the embedding model on GPU 4"
        return True, None

    def spawn(self, kind):
        old = load_json(self.record_path(kind))
        if old:
            write_json(self.runtime / f"{kind}_previous_{time.time_ns()}.json", old)
        stamp = time.time_ns()
        log = self.root / "logs" / f"automatic_{kind}_{stamp}.log"
        command = self.config[f"{kind}_command"]
        environment = self.config[f"{kind}_environment"]
        with log.open("ab", buffering=0) as stream:
            child = subprocess.Popen(
                command,
                cwd=str(REPO),
                env={**os.environ, **environment},
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        self.children.append(child)
        identity = process_identity(child.pid)
        record = {
            "pid": child.pid,
            "started_at": time.time(),
            "log": str(log),
            "command": command,
            "environment": environment,
            "process": identity,
            "members": [identity] if identity else [],
            "launched_by": "experiment_supervisor",
            "endpoint": self.config["endpoint"],
        }
        write_json(self.record_path(kind), record)
        key = f"{kind}_attempts_without_progress"
        self.state[key] += 1
        self.state[f"{kind}_next_attempt_at"] = time.time() + backoff(self.state[key])
        if kind == "worker":
            self.state["last_progress_at"] = time.time()
        self.event("restart", kind=kind, pid=child.pid, log=str(log), attempt=self.state[key])

    def completed_report(self):
        report = load_json(self.root / "val_report.json")
        if report is None:
            return False
        rows = read_jsonl(self.root / "manifests/val.jsonl")
        predictions = read_jsonl(self.root / "val_predictions.jsonl")
        validate_submission(predictions, rows)
        return report["queries"] == len(rows) and sum(report["status_counts"].values()) == len(rows)

    def tick(self):
        self.children = [child for child in self.children if child.poll() is None]
        config = self.verify_protocol()
        worker, worker_alive = self.active("worker")
        service, service_alive = self.active("service")
        if self.completed_report():
            stop_owned(worker)
            if self.config["stop_service_on_complete"]:
                stop_owned(service)
            report = load_json(self.root / "val_report.json")
            execution = load_json(self.root / "execution_status.json", {})
            execution.update(
                stage="complete",
                updated_at=time.time(),
                validation_queries=report["queries"],
                validation_failures=len(report["failed_qids"]),
                metrics=report["official_metrics"]["brief"],
                completion_verified_by="supervisor",
            )
            write_json(self.root / "execution_status.json", execution)
            self.event("complete", report=str(self.root / "val_report.json"))
            self.status("complete")
            return True
        progress = load_json(self.root / "val_progress.json", {"completed": 0, "queries": []})
        completed = progress["completed"]
        now = time.time()
        if completed > self.state["high_water_completed"]:
            self.state.update(
                high_water_completed=completed,
                last_progress_at=now,
                worker_attempts_without_progress=0,
                service_attempts_without_progress=0,
            )
        self.state.update(
            completed=completed,
            total=config["full"]["expected_queries"],
            worker_alive=worker_alive,
            service_alive=service_alive,
        )
        trailing_failures = 0
        for row in reversed(progress["queries"]):
            if row["status"] == "ok":
                break
            trailing_failures += 1
        if trailing_failures >= 10:
            stop_owned(worker)
            self.status(
                "attention_required",
                reason="Ten consecutive failed observations; refusing silent mass fallback",
            )
            return False
        if shutil.disk_usage(self.root).free < 2 * 1024**3:
            stop_owned(worker)
            self.status("waiting_for_disk", reason="Less than 2 GiB free on artifact volume")
            return False
        healthy, error = self.health(config["vlm"])
        if not healthy:
            self.state["health_failures"] += 1
            in_startup = (
                service_alive and now - service["started_at"] < self.config["service_startup_seconds"]
            )
            if self.state["health_failures"] < 3 and service_alive or in_startup:
                self.status("checking_service", health_error=error)
                return False
            stop_owned(worker)
            stop_owned(service)
            if now < self.state.get("service_next_attempt_at", 0):
                self.status("service_backoff", health_error=error)
                return False
            ready, reason = self.gpu_ready([])
            if not ready:
                self.status("waiting_for_gpu", reason=reason)
                return False
            self.spawn("service")
            self.status("starting_service", health_error=error)
            return False
        self.state["health_failures"] = 0
        if not service_alive:
            self.status("attention_required", reason="Healthy endpoint has no verified owned service process")
            return False
        if worker_alive:
            if now - self.state["last_progress_at"] > self.config["stall_seconds"]:
                self.event("stalled_worker", pid=worker["pid"], seconds=now - self.state["last_progress_at"])
                stop_owned(worker)
                stop_owned(service)
                self.state["service_next_attempt_at"] = now + 60
                self.state["last_progress_at"] = now
                self.status("recovering_stall")
            else:
                self.status("running", last_error=None)
            return False
        # Clean verified leftover children before restarting a dead leader.
        stop_owned(worker)
        if now < self.state.get("worker_next_attempt_at", 0):
            self.status("worker_backoff")
            return False
        ready, reason = self.gpu_ready([service], require_embedding_space=True)
        if not ready:
            self.status("waiting_for_gpu", reason=reason)
            return False
        self.spawn("worker")
        self.status("resuming")
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    supervisor = Supervisor(json.loads(args.config.read_text()))
    with (supervisor.runtime / "supervisor.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        while True:
            try:
                if supervisor.tick():
                    return
            except Exception as error:
                supervisor.event("monitor_error", error=f"{type(error).__name__}: {error}")
                supervisor.status("attention_required", last_error=f"{type(error).__name__}: {error}")
            if args.once:
                return
            time.sleep(supervisor.config["poll_seconds"])


if __name__ == "__main__":
    main()
