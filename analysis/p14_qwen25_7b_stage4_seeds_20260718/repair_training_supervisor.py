#!/usr/bin/env python3
"""Run repaired continuation training as each locked pool becomes ready."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--python", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--lock", type=Path, required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--arm", choices=["inpo_avg", "ipo"], required=True)
    p.add_argument("--gpu", type=int, required=True)
    args = p.parse_args()
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY must be supplied ephemerally")
    trainer = args.project / "analysis/p14_qwen25_7b_stage4_seeds_20260718/train_repaired_stage.py"
    log = args.root / f"seeds/s42/logs/{args.arm}_stage3_stage4_supervisor.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    for stage in (3, 4):
        prepared = args.root / f"seeds/s42/stage{stage}/{args.arm}/pool/PREPARED"
        while not prepared.is_file():
            time.sleep(30)
        command = [str(args.python), str(trainer), "--project", str(args.project), "--python", str(args.python),
                   "--root", str(args.root), "--cache", str(args.cache), "--lock", str(args.lock),
                   "--candidate", args.candidate, "--arm", args.arm, "--stage", str(stage), "--gpu", str(args.gpu)]
        with log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"stage": stage, "started": time.time(), "command": command}) + "\n")
            handle.flush()
            if subprocess.run(command, env=os.environ.copy(), stdout=handle, stderr=subprocess.STDOUT).returncode:
                raise SystemExit(1)


if __name__ == "__main__":
    main()
