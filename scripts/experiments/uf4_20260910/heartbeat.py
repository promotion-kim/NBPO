"""Append the real state of the campaign to a file every interval, forever.

Written for the case where no one is reading: the record has to be enough on its
own. Every line is measured -- process ids that exist, GPU utilisation the driver
reports, the artifacts actually on disk -- and no line is an intention.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

KST = timezone(timedelta(hours=9))


def gpu_rows():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"], text=True, timeout=30)
    except Exception as error:                              # noqa: BLE001
        return [{"error": str(error)[:120]}]
    rows = []
    for line in out.strip().splitlines():
        index, memory, util = (part.strip() for part in line.split(","))
        rows.append({"gpu": int(index), "mem_mib": int(memory), "util_pct": int(util)})
    return rows


def alive(pid):
    try:
        os.kill(int(pid), 0)
    except (ProcessLookupError, ValueError, TypeError):
        return False
    except PermissionError:
        return True
    return True


def tail_progress(path, patterns):
    try:
        text = Path(path).read_text(errors="ignore")[-200000:]
    except OSError:
        return None
    import re
    for pattern in patterns:
        found = re.findall(pattern, text)
        if found:
            return found[-1]
    return None


def snapshot(root):
    root = Path(root)
    state_path = root / "jobs" / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"jobs": {}}
    jobs = []
    for job_id, entry in state.get("jobs", {}).items():
        row = {"job_id": job_id, "state": entry.get("state"), "gpus": entry.get("gpus", 0)}
        if entry.get("state") == "RUNNING":
            row["pid"] = entry.get("pid")
            row["pid_alive"] = alive(entry.get("pid"))
            row["running_s"] = round(time.time() - entry.get("started_epoch", time.time()))
            row["progress"] = tail_progress(entry.get("log", ""),
                                            [r'"step": (\d+)', r"\| (\d+/\d+)"])
        elif entry.get("state") in ("PENDING", "READY", "BLOCKED"):
            row["waiting_on"] = entry.get("waiting_on") or entry.get("failure")
        elif entry.get("state") == "FAILED":
            row["failure"] = entry.get("failure") or entry.get("returncode")
        jobs.append(row)
    gpus = gpu_rows()
    return {"kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S"),
            "gpu": gpus,
            "gpu_idle": all(row.get("util_pct", 0) == 0 for row in gpus if "gpu" in row),
            "jobs": jobs,
            "queue_updated": state.get("updated")}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="/work/uf4_20260910")
    ap.add_argument("--interval", type=float, default=1800.0)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    out = Path(args.root) / "logs" / "heartbeat.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    while True:
        record = snapshot(args.root)
        with out.open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
