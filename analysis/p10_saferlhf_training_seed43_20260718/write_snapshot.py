#!/usr/bin/env python3
"""Write a measured P10 monitor snapshot without inferring progress."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path


KST = timezone(timedelta(hours=9))


def command(args: list[str]) -> str:
    return subprocess.run(args, text=True, capture_output=True, check=False).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--stage", required=True)
    args = parser.parse_args()
    arms = {}
    for path in sorted((args.experiment / "stage1").glob("*/train/*/job_status.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        arms.setdefault(data.get("arm", path.parents[2].name), {})[data.get("stage", path.parent.name)] = {
            "status": data.get("status"), "returncode": data.get("returncode"),
            "elapsed_seconds": data.get("elapsed_seconds"), "wandb_run_id": data.get("wandb_run_id"),
        }
    stamp = datetime.now(KST).strftime("%Y%m%dT%H%M%S%z")
    payload = {
        "timestamp_kst": datetime.now(KST).isoformat(),
        "host": args.host,
        "stage": args.stage,
        "gpu_snapshot": command(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader"]),
        "compute_processes": command(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader"]),
        "stage1_statuses": arms,
    }
    out = args.experiment / "hourly" / f"{stamp}_{args.host}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
