#!/usr/bin/env python3
import argparse
import json
import socket
import subprocess
import time
from pathlib import Path


def output(command):
    return subprocess.check_output(command, text=True).splitlines()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--interval", type=int, default=1800)
    args = parser.parse_args()
    host = socket.gethostname()
    folder = args.root / "hourly"
    folder.mkdir(parents=True, exist_ok=True)
    while True:
        stamp = time.strftime("%Y%m%dT%H%M%S%z")
        scheduler = args.root / "scheduler"
        payload = {
            "timestamp": time.time(),
            "host": host,
            "gpus": output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ]
            ),
            "processes": output(
                [
                    "nvidia-smi",
                    "--query-compute-apps=gpu_uuid,pid,used_memory",
                    "--format=csv,noheader,nounits",
                ]
            ),
            "done": sorted(path.name for path in scheduler.glob("*.DONE.json")),
            "failed": sorted(path.name for path in scheduler.glob("*.FAILED.json")),
            "claims": sorted(path.name for path in (scheduler / "claims").glob("*")),
        }
        (folder / f"{stamp}_{host}.json").write_text(json.dumps(payload, indent=2) + "\n")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
