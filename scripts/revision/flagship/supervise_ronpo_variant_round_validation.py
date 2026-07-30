#!/usr/bin/env python3
"""Resume-safe handoff from one training round to validation-only RM selection."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path


def run(command: list[str], cwd: Path, log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        subprocess.run(command, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--round-id", required=True)
    parser.add_argument("--round-root", type=Path, required=True)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--flagship-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--evaluator-lock", type=Path, required=True)
    parser.add_argument("--general-rm-cache", type=Path, required=True)
    parser.add_argument("--deadline", required=True, help="ISO datetime with offset")
    args = parser.parse_args()
    manifest_path = args.round_root / "training_manifest.json"
    deadline = datetime.fromisoformat(args.deadline)
    while True:
        if datetime.now().astimezone() >= deadline:
            raise RuntimeError("deadline reached before training round completed")
        try:
            status = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            time.sleep(20); continue
        if status.get("status") == "completed":
            break
        if status.get("status") == "failed":
            raise RuntimeError(f"training round failed: {status.get('failures')}")
        time.sleep(20)
    work = args.run_dir / f"{args.round_id}_validation"
    checkpoint_manifest = work / "checkpoint_manifest.json"
    run([args.python, str(args.project / "scripts/revision/flagship/build_ronpo_variant_checkpoint_manifest.py"),
         "--round", f"{args.round_id}={args.round_root}", "--grid", f"{args.round_id}={args.grid}",
         "--output", str(checkpoint_manifest)], args.project, work / "logs/build_manifest.log")
    run([args.python, str(args.project / "scripts/revision/flagship/run_ronpo_variant_validation_decode_gate.py"),
         "--project", str(args.project), "--python", args.python,
         "--flagship-root", str(args.flagship_root), "--manifest", str(checkpoint_manifest),
         "--work", str(work)], args.project, work / "logs/decode_gate_driver.log")
    run([args.python, str(args.project / "scripts/revision/flagship/run_ronpo_variant_validation_rewards.py"),
         "--project", str(args.project), "--python", args.python,
         "--flagship-root", str(args.flagship_root), "--manifest", str(checkpoint_manifest),
         "--evaluator-lock", str(args.evaluator_lock),
         "--general-rm-cache", str(args.general_rm_cache), "--work", str(work)],
        args.project, work / "logs/reward_selection_driver.log")
    print(json.dumps({"status": "completed", "round": args.round_id,
                      "selection_lock": str(work / "results/selection_lock.json"),
                      "spent_sealed_split_touched": False}, indent=2))


if __name__ == "__main__":
    main()
