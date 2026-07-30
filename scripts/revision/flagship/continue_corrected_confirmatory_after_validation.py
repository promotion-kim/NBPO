#!/usr/bin/env python3
"""Advance to the confirmatory holdout only after protocol validation passes."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--decode-python", required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--base-revision", required=True)
    args = parser.parse_args()

    validation_work = args.eval_root / "protocol_validation_v2"
    validation_status = validation_work / "status.json"
    while True:
        try:
            status = json.loads(validation_status.read_text())
        except (OSError, json.JSONDecodeError):
            time.sleep(15)
            continue
        if status.get("status") in {"completed", "failed"}:
            break
        time.sleep(15)
    if status.get("status") != "completed" or status.get("all_passed") is not True:
        raise RuntimeError(f"corrected protocol validation failed; holdout remains unopened: {status}")

    lock = args.eval_root / "corrected_protocol_lock.json"
    subprocess.run([
        args.python, str(args.project / "scripts/revision/flagship/lock_corrected_confirmatory_protocol.py"),
        "--split-manifest", str(args.eval_root / "input/split_manifest.json"),
        "--protocol-candidate", str(validation_work / "protocol_candidate.json"),
        "--validation-summary", str(validation_work / "stability_gates/summary.json"),
        "--selection-lock", str(args.eval_root / "input/selection_lock.json"),
        "--confirmatory-prompts", str(args.eval_root / "input/confirmatory_holdout_prompts.jsonl"),
        "--output", str(lock),
    ], cwd=args.project, check=True)

    subprocess.run([
        args.python, str(args.project / "scripts/revision/flagship/run_corrected_confirmatory_reward_eval.py"),
        "--root", str(args.root), "--project", str(args.project), "--python", args.python,
        "--decode-python", args.decode_python, "--work", str(args.eval_root / "confirmatory"),
        "--prompts", str(args.eval_root / "input/confirmatory_holdout_prompts.jsonl"),
        "--protocol-lock", str(lock), "--models-tsv", str(args.eval_root / "input/models.tsv"),
        "--expected-prompts", "1736", "--base-revision", args.base_revision,
    ], cwd=args.project, check=True)


if __name__ == "__main__":
    main()
