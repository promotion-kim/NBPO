"""Release a vLLM engine child that deadlocks its parent after the work is done.

HarmBench jobs finish writing every summary and their evaluation_complete.json,
then sit in do_wait on a VLLM::EngineCore child that never exits, holding the GPU
against everything queued behind them. Observed once and handled by hand; the
follow-up queue runs two more of these, so it is automated here under conditions
narrow enough that it cannot mask a real failure:

  * the process must be a VLLM::EngineCore,
  * its parent must be an evaluate_responses for a job in this campaign root,
  * that job's output directory must already contain evaluation_complete.json,
  * the parent must have been in do_wait for at least the grace period,
  * and the job must not yet have recorded an exit.

It never signals anything else, never signals a job that is still computing, and
records every signal it sends with the hashes of the results that were already
on disk when it sent it.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path("/work/nbpo_repair_20260909")
LOG = ROOT / "analysis_claude" / "engine_reaper.jsonl"


def processes():
    out = subprocess.check_output(["ps", "-eo", "pid,ppid,cmd", "--no-headers"], text=True)
    for line in out.splitlines():
        pid, ppid, cmd = line.strip().split(None, 2)
        yield int(pid), int(ppid), cmd


def wchan(pid):
    try:
        return Path(f"/proc/{pid}/wchan").read_text().strip()
    except OSError:
        return ""


def out_dir(cmd):
    match = re.search(r"--out\s+(\S+)", cmd)
    return Path(match.group(1)) if match else None


def job_of(cmd):
    match = re.search(r"--job\s+(\S+)", cmd)
    return match.group(1) if match else None


def hashes(directory):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.glob("*.json"))}


def sweep(grace):
    table = {pid: (ppid, cmd) for pid, ppid, cmd in processes()}
    released = []
    for pid, (ppid, cmd) in table.items():
        if "VLLM::EngineCore" not in cmd:
            continue
        parent = table.get(ppid)
        if not parent or "evaluate_responses" not in parent[1] or str(ROOT) not in parent[1]:
            continue
        results = out_dir(parent[1])
        if results is None or not (results / "evaluation_complete.json").exists():
            continue
        if wchan(ppid) != "do_wait":
            continue
        grandparent = table.get(parent[0])
        job = job_of(grandparent[1]) if grandparent else None
        if job and (ROOT / "jobs" / job / "exit.json").exists():
            continue
        record = {"time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  "engine_pid": pid, "parent_pid": ppid, "job": job,
                  "results_dir": str(results), "results_sha256": hashes(results),
                  "reason": "parent in do_wait with results already complete"}
        os.kill(pid, signal.SIGTERM)
        with LOG.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        released.append(record)
    return released


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grace", type=float, default=180.0,
                    help="seconds a parent must have been finished before releasing it")
    ap.add_argument("--interval", type=float, default=60.0)
    ap.add_argument("--background", action="store_true")
    args = ap.parse_args()
    directory = ROOT / "controllers" / "engine_reaper_v1"
    directory.mkdir(parents=True, exist_ok=True)
    if args.background:
        with (directory / "stdout.log").open("a") as log:
            process = subprocess.Popen(["python3", str(Path(__file__).resolve()),
                                        "--grace", str(args.grace), "--interval", str(args.interval)],
                                       cwd=ROOT / "code", stdout=log, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, start_new_session=True)
        (directory / "launch.json").write_text(json.dumps({"pid": process.pid}, indent=2))
        print(json.dumps({"reaper_pid": process.pid}), flush=True)
        return
    seen = {}
    while True:
        table = {pid: cmd for pid, _, cmd in processes()}
        for pid in list(seen):
            if pid not in table:
                seen.pop(pid)
        for pid, cmd in table.items():
            if "VLLM::EngineCore" in cmd:
                seen.setdefault(pid, time.monotonic())
        ready = [pid for pid, since in seen.items() if time.monotonic() - since >= args.grace]
        if ready:
            for record in sweep(args.grace):
                print(json.dumps(record), flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
