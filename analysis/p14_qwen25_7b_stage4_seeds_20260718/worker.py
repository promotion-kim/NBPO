#!/usr/bin/env python3
"""Dependency-aware single-GPU worker for all seeds, stages, and arms."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import time
from pathlib import Path

from common import ARMS, SEEDS, gate_path, stage_root


def passed(path: Path) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("passed") is True and data.get("status") == "passed"
    except (FileNotFoundError, json.JSONDecodeError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--train-python", type=Path, required=True)
    parser.add_argument("--infer-python", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY must be supplied ephemerally")
    scheduler = args.root / "scheduler"
    (scheduler / "claims").mkdir(parents=True, exist_ok=True)
    (scheduler / "logs").mkdir(parents=True, exist_ok=True)
    worker_id = f"{socket.gethostname()}-g{args.gpu}"
    tasks = [(stage, seed, arm) for stage in range(1, 5) for arm in ARMS for seed in SEEDS]
    while True:
        claimed = False
        for stage, seed, arm in tasks:
            name = f"s{seed}__stage{stage}__{arm}"
            done, failed, blocked = (scheduler / f"{name}.{suffix}.json" for suffix in ("DONE", "FAILED", "BLOCKED"))
            if done.exists() or failed.exists() or blocked.exists():
                continue
            if stage > 1 and not passed(gate_path(args.root, seed, stage - 1, arm)):
                parent = f"s{seed}__stage{stage - 1}__{arm}"
                if (scheduler / f"{parent}.FAILED.json").exists() or (scheduler / f"{parent}.BLOCKED.json").exists():
                    blocked.write_text(json.dumps({"status": "blocked_by_failed_parent", "parent": parent, "at": time.time()}, indent=2) + "\n", encoding="utf-8")
                continue
            claim = scheduler / "claims" / name
            try:
                claim.mkdir()
            except FileExistsError:
                continue
            (claim / "owner.json").write_text(json.dumps({"worker": worker_id, "pid": os.getpid(), "claimed": time.time()}, indent=2) + "\n", encoding="utf-8")
            task_log = scheduler / "logs" / f"{name}_{worker_id}.log"
            started = time.time()
            commands: list[list[str]] = []
            if stage > 1 and not (stage_root(args.root, seed, stage) / arm / "pool/PREPARED").is_file():
                commands.append(["bash", str(args.project / "analysis/p14_qwen25_7b_stage4_seeds_20260718/prepare_continuation_pool.sh"), str(args.project), str(args.train_python), str(args.infer_python), str(args.root), str(args.cache), str(seed), str(stage), arm, str(args.gpu)])
            commands.append([str(args.train_python), str(args.project / "analysis/p14_qwen25_7b_stage4_seeds_20260718/train_arm.py"), "--project", str(args.project), "--train-python", str(args.train_python), "--root", str(args.root), "--cache", str(args.cache), "--seed", str(seed), "--stage", str(stage), "--arm", arm, "--gpu", str(args.gpu)])
            commands.append(["bash", str(args.project / "analysis/p14_qwen25_7b_stage4_seeds_20260718/decode_and_gate.sh"), str(args.project), str(args.train_python), str(args.infer_python), str(args.root), str(seed), str(stage), arm, str(args.gpu)])
            returncode = 0
            with task_log.open("a", encoding="utf-8") as handle:
                for command in commands:
                    handle.write("command=" + json.dumps(command) + "\n")
                    handle.flush()
                    returncode = subprocess.run(command, env=os.environ.copy(), stdout=handle, stderr=subprocess.STDOUT).returncode
                    if returncode:
                        break
            payload = {"worker": worker_id, "seed": seed, "stage": stage, "arm": arm, "returncode": returncode, "started": started, "finished": time.time(), "elapsed_seconds": time.time() - started, "log": str(task_log)}
            (done if returncode == 0 and passed(gate_path(args.root, seed, stage, arm)) else failed).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            claimed = True
            break
        if claimed:
            continue
        terminal = sum(any((scheduler / f"s{seed}__stage{stage}__{arm}.{suffix}.json").exists() for suffix in ("DONE", "FAILED", "BLOCKED")) for stage, seed, arm in tasks)
        active = [path for path in (scheduler / "claims").iterdir() if not any((scheduler / f"{path.name}.{suffix}.json").exists() for suffix in ("DONE", "FAILED", "BLOCKED"))]
        if terminal == len(tasks):
            break
        if not active:
            time.sleep(5)
        else:
            time.sleep(20)


if __name__ == "__main__":
    main()
