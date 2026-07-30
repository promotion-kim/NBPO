#!/usr/bin/env python3
"""Start SPPO Stage 4 after reward-blind candidate selection and pool build."""

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
    p.add_argument("--gpu", type=int, required=True)
    args = p.parse_args()
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY must be supplied ephemerally")
    selection = args.root / "seeds/s42/stage3/sppo_avg/repair_selection.json"
    prepared = args.root / "seeds/s42/stage4/sppo_avg/pool/PREPARED"
    while not selection.is_file() or not prepared.is_file():
        time.sleep(30)
    candidate = json.loads(selection.read_text())["candidate"]
    command = [str(args.python), str(args.project / "analysis/p14_qwen25_7b_stage4_seeds_20260718/train_repaired_stage.py"),
               "--project", str(args.project), "--python", str(args.python), "--root", str(args.root),
               "--cache", str(args.cache), "--lock", str(args.root / "sppo_extension/baseline_repair_extension_lock.json"),
               "--candidate", candidate, "--arm", "sppo_avg", "--stage", "4", "--gpu", str(args.gpu)]
    raise SystemExit(subprocess.run(command, env=os.environ.copy()).returncode)


if __name__ == "__main__":
    main()
