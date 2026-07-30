#!/usr/bin/env python3
import argparse
import json
import os
import socket
import subprocess
import time
from pathlib import Path


ARMS = [
    "ronpo_os", "inpo_avg", "sppo_avg", "simpo", "ipo", "dpo",
    "ht_mnpo_harmless", "ht_mnpo_helpfulness",
]


def passed(path):
    try:
        data = json.loads(path.read_text())
        return data.get("status") == "passed" and data.get("passed") is True
    except (FileNotFoundError, json.JSONDecodeError):
        return False


def gate(root, seed, stage, arm):
    seed_root = root / f"seed{seed}"
    base = seed_root / ("stage12" if stage <= 2 else f"stage{stage}")
    return base / f"stage{stage}_stability_p8_locked_panel/gates/{arm}.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--venv", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    scheduler = args.root / "scheduler"
    (scheduler / "claims").mkdir(parents=True, exist_ok=True)
    (scheduler / "logs").mkdir(parents=True, exist_ok=True)
    worker_id = f"{socket.gethostname()}-g{args.gpu}-{os.getpid()}"
    tasks = [
        (stage, seed, arm)
        for stage in range(1, 5)
        for arm in ARMS
        for seed in (45, 46)
    ]
    while True:
        claimed = False
        for stage, seed, arm in tasks:
            name = f"s{seed}__stage{stage}__{arm}"
            done = scheduler / f"{name}.DONE.json"
            failed = scheduler / f"{name}.FAILED.json"
            if done.exists() or failed.exists():
                continue
            if stage > 1 and not passed(gate(args.root, seed, stage - 1, arm)):
                continue
            claim = scheduler / "claims" / name
            try:
                claim.mkdir()
            except FileExistsError:
                continue
            (claim / "owner.json").write_text(
                json.dumps({"worker": worker_id, "pid": os.getpid(), "at": time.time()}, indent=2) + "\n"
            )
            command = [
                str(args.venv / "bin/python"),
                str(args.project / "analysis/saferlhf_seed45_46_replication_20260723/run_stage_task.py"),
                "--project", str(args.project), "--venv", str(args.venv),
                "--root", str(args.root), "--base", args.base,
                "--seed", str(seed), "--stage", str(stage), "--arm", arm,
                "--gpu", str(args.gpu), "--log",
                str(scheduler / "logs" / f"{name}_{worker_id}.log"),
            ]
            started = time.time()
            result = subprocess.run(command)
            payload = {
                "worker": worker_id, "seed": seed, "stage": stage, "arm": arm,
                "returncode": result.returncode, "started": started,
                "finished": time.time(), "elapsed_seconds": time.time() - started,
            }
            target = done if result.returncode == 0 else failed
            target.write_text(json.dumps(payload, indent=2) + "\n")
            claimed = True
            break
        if claimed:
            continue
        terminal = sum(
            (scheduler / f"s{seed}__stage{stage}__{arm}.DONE.json").exists()
            or (scheduler / f"s{seed}__stage{stage}__{arm}.FAILED.json").exists()
            for stage in range(1, 5) for arm in ARMS for seed in (45, 46)
        )
        active = [
            path for path in (scheduler / "claims").iterdir()
            if not (scheduler / f"{path.name}.DONE.json").exists()
            and not (scheduler / f"{path.name}.FAILED.json").exists()
        ]
        if terminal == len(tasks) or (not active and list(scheduler.glob("*.FAILED.json"))):
            break
        time.sleep(20)


if __name__ == "__main__":
    main()
