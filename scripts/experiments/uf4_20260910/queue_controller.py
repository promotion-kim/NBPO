"""A dependency queue that keeps running after the conversation ends.

A job is RUNNING only while its process group actually exists; it is DONE only
when the process returned zero AND every declared artifact is on disk. A live
shell, a sleep timer or a reachable session is never evidence that work is
happening, so nothing here infers state from any of those.

Jobs are JSON files dropped into <root>/jobs/queue. The controller never edits a
spec, never deletes a log, and never touches a process it did not start.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

STATES = ("PENDING", "READY", "RUNNING", "DONE", "FAILED", "BLOCKED")


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Controller:
    def __init__(self, root, total_gpus, poll):
        self.root = Path(root)
        self.queue_dir = self.root / "jobs" / "queue"
        self.run_dir = self.root / "jobs" / "runs"
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "jobs" / "state.json"
        self.report_path = self.root / "logs" / "queue_report.jsonl"
        self.total_gpus = total_gpus
        self.poll = poll
        self.state = {}
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text()).get("jobs", {})

    # ---------------------------------------------------------------- specs
    def load_specs(self):
        specs = {}
        for path in sorted(self.queue_dir.glob("*.json")):
            spec = json.loads(path.read_text())
            for field in ("job_id", "command", "gpus"):
                if field not in spec:
                    raise ValueError(f"{path}: missing {field}")
            spec.setdefault("depends_on", [])
            spec.setdefault("cwd", str(self.root))
            spec.setdefault("env", {})
            spec.setdefault("timeout_s", 24 * 3600)
            spec.setdefault("artifacts", [])
            spec["spec_sha256"] = file_hash(path)
            spec["spec_path"] = str(path)
            if spec["job_id"] in specs:
                raise ValueError(f"duplicate job_id {spec['job_id']}")
            specs[spec["job_id"]] = spec
        return specs

    # ---------------------------------------------------------------- state
    def record(self, job_id, **fields):
        entry = self.state.setdefault(job_id, {})
        entry.update(fields)
        entry["updated"] = now()

    def flush(self, specs):
        payload = {"updated": now(), "total_gpus": self.total_gpus,
                   "gpus_in_use": self.gpus_in_use(),
                   "busy_devices": sorted(self.busy_devices()),
                   "reserved_devices": sorted(self.reserved_devices()),
                   "jobs": self.state,
                   "queued_specs": sorted(specs)}
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2) + "\n")
        tmp.replace(self.state_path)

    def gpus_in_use(self):
        return sum(entry.get("gpus", 0) for entry in self.state.values()
                   if entry.get("state") == "RUNNING")

    def busy_devices(self, threshold_mib=1024):
        """Device indices the driver reports as occupied, whoever owns them.

        Counting free slots is not enough: two jobs can be scheduled into the
        same slot count and land on the same card. Allocation is therefore by
        device index, and the driver is the authority because this controller
        cannot see a job it did not launch. Nothing here signals another process.
        """
        try:
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,memory.used",
                 "--format=csv,noheader,nounits"], text=True, timeout=30)
        except Exception:                                  # noqa: BLE001
            return set(range(self.total_gpus))             # unknown: assume none free
        busy = set()
        for line in output.strip().splitlines():
            index, memory = (part.strip() for part in line.split(","))
            if int(memory) >= threshold_mib:
                busy.add(int(index))
        return busy

    def reserved_devices(self):
        """Devices this controller has handed to jobs it still believes are running."""
        out = set()
        for entry in self.state.values():
            if entry.get("state") == "RUNNING":
                out.update(entry.get("devices", []))
        return out

    # ------------------------------------------------------------- lifecycle
    def alive(self, pid):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def poll_running(self):
        for job_id, entry in self.state.items():
            if entry.get("state") != "RUNNING":
                continue
            pid = entry.get("pid")
            exit_path = Path(entry["exit_path"])
            if exit_path.exists():
                result = json.loads(exit_path.read_text())
                self.finish(job_id, entry, result["returncode"])
                continue
            if pid and not self.alive(pid):
                # The process is gone without an exit record: report it as a
                # failure with that fact, never as a success.
                self.record(job_id, state="FAILED", returncode=None,
                            failure="process vanished before writing its exit record")
                self.log_event(job_id, "FAILED", "process vanished")
                continue
            if time.time() - entry.get("started_epoch", time.time()) > entry.get("timeout_s", 1e18):
                self.record(job_id, state="FAILED", failure="timeout")
                self.log_event(job_id, "FAILED", "timeout; process left running for inspection")

    def finish(self, job_id, entry, returncode):
        missing = [p for p in entry.get("artifacts", []) if not Path(p).exists()]
        if returncode == 0 and not missing:
            self.record(job_id, state="DONE", returncode=0,
                        artifact_hashes={p: file_hash(p) for p in entry.get("artifacts", [])
                                         if Path(p).is_file()})
            self.log_event(job_id, "DONE", "")
        else:
            self.record(job_id, state="FAILED", returncode=returncode,
                        missing_artifacts=missing)
            self.log_event(job_id, "FAILED",
                           f"returncode={returncode} missing={missing}")

    def log_event(self, job_id, state, detail):
        with self.report_path.open("a") as stream:
            stream.write(json.dumps({"time": now(), "job_id": job_id, "state": state,
                                     "detail": detail}) + "\n")

    def launch(self, spec, devices):
        job_id = spec["job_id"]
        run = self.run_dir / job_id
        run.mkdir(parents=True, exist_ok=True)
        log_path = run / "stdout.log"
        exit_path = run / "exit.json"
        if exit_path.exists():
            exit_path.unlink()
        env = dict(os.environ)
        env.update({str(k): str(v) for k, v in spec["env"].items()})
        # The controller owns device allocation. A spec may not pin its own
        # cards: two specs naming the same index is exactly how two jobs end up
        # on one GPU while the slot count still looks free.
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(d) for d in devices)
        env["VLLM_CACHE_ROOT"] = f"{self.root}/logs/vllm_cache_gpu{devices[0]}"
        # The wrapper writes the exit record even if the payload is killed, so a
        # missing record always means the wrapper itself died.
        inner = " ".join(shlex.quote(part) for part in spec["command"])
        wrapper = (f"{inner}; rc=$?; "
                   f"printf '{{\"returncode\": %d, \"finished\": \"%s\"}}\\n' "
                   f"$rc \"$(date -u +%FT%TZ)\" > {shlex.quote(str(exit_path))}; exit $rc")
        with log_path.open("a") as sink:
            sink.write(f"\n===== {now()} launching {job_id} =====\n")
            sink.flush()
            process = subprocess.Popen(["bash", "-lc", wrapper], cwd=spec["cwd"], env=env,
                                       stdout=sink, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        (run / "launch.json").write_text(json.dumps(
            {"job_id": job_id, "command": spec["command"], "cwd": spec["cwd"],
             "env_overrides": spec["env"], "gpus": spec["gpus"],
             "devices": devices, "pid": process.pid,
             "spec_sha256": spec["spec_sha256"], "started": now()}, indent=2) + "\n")
        self.record(job_id, state="RUNNING", pid=process.pid, gpus=spec["gpus"],
                    devices=devices,
                    log=str(log_path), exit_path=str(exit_path),
                    artifacts=spec["artifacts"], timeout_s=spec["timeout_s"],
                    started=now(), started_epoch=time.time(),
                    spec_sha256=spec["spec_sha256"])
        self.log_event(job_id, "RUNNING", f"pid={process.pid} devices={devices}")
        return process.pid

    def schedule(self, specs):
        taken = self.busy_devices() | self.reserved_devices()
        free_devices = [d for d in range(self.total_gpus) if d not in taken]
        for job_id in sorted(specs, key=lambda j: (specs[j].get("priority", 100), j)):
            spec = specs[job_id]
            entry = self.state.get(job_id, {})
            if entry.get("state") in ("RUNNING", "DONE"):
                continue
            if entry.get("state") == "FAILED":
                continue
            deps = spec["depends_on"]
            unmet = [d for d in deps if self.state.get(d, {}).get("state") != "DONE"]
            if any(self.state.get(d, {}).get("state") == "FAILED" for d in deps):
                self.record(job_id, state="BLOCKED", gpus=0,
                            failure=f"dependency failed: {deps}")
                continue
            if unmet:
                self.record(job_id, state="PENDING", gpus=0, waiting_on=unmet)
                continue
            if spec["gpus"] > self.total_gpus:
                self.record(job_id, state="BLOCKED", gpus=0,
                            failure="requests more GPUs than this controller owns")
                continue
            if spec["gpus"] > len(free_devices):
                self.record(job_id, state="READY", gpus=0,
                            waiting_on=[f"{spec['gpus']} GPUs, free devices {free_devices}"])
                continue
            devices = free_devices[:spec["gpus"]]
            self.launch(spec, devices)
            free_devices = free_devices[spec["gpus"]:]

    def run(self, once=False):
        while True:
            specs = self.load_specs()
            self.poll_running()
            self.schedule(specs)
            self.flush(specs)
            if once:
                return
            if all(self.state.get(j, {}).get("state") in ("DONE", "FAILED", "BLOCKED")
                   for j in specs) and specs:
                self.log_event("_controller", "IDLE",
                               "every queued job has reached a terminal state")
            time.sleep(self.poll)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="/work/uf4_20260910")
    ap.add_argument("--total-gpus", type=int, default=4)
    ap.add_argument("--poll", type=float, default=20.0)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    Controller(args.root, args.total_gpus, args.poll).run(once=args.once)


if __name__ == "__main__":
    main()
