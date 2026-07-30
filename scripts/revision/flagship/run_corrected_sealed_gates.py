#!/usr/bin/env python3
"""Run the corrected sealed gate uniformly and preserve a combined summary."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


METHODS = (
    "base", "ronpo_full_expect", "ronpo_k_only", "dpo", "ipo", "simpo",
    "sppo_avg", "inpo_avg", "ht_mnpo_helpfulness", "ht_mnpo_safety",
    "ht_mnpo_conciseness",
)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    output_dir = args.work / "stability_gates_corrected"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.work / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    base = args.work / "generations/base/output_42.json"
    models = {}
    returncodes = {}
    for method in METHODS:
        output = output_dir / f"{method}.json"
        command = [
            args.python,
            str(args.project / "scripts/revision/flagship/stability_gate_corrected.py"),
            "--base", str(base),
            "--candidate", str(args.work / "generations" / method / "output_42.json"),
            "--output", str(output),
            "--min-length-ratio", "0.33",
            "--max-length-ratio", "2.0",
            "--max-repeat-run", "20",
            "--expected-records", "604",
        ]
        completed = subprocess.run(command, cwd=args.project, text=True, capture_output=True)
        (log_dir / f"stability_corrected_{method}.log").write_text(
            completed.stdout + completed.stderr
        )
        returncodes[method] = completed.returncode
        if not output.is_file():
            raise RuntimeError(f"corrected gate produced no artifact for {method}")
        models[method] = json.loads(output.read_text())
    payload = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "fail_closed": True,
        "detector": "corrected_nonempty_paired_span_v1",
        "thresholds": {
            "records": 604,
            "empty_count": 0,
            "think_leak_count": 0,
            "min_length_ratio": 0.33,
            "max_length_ratio": 2.0,
            "max_repeat_run": 20,
        },
        "models": models,
        "returncodes": returncodes,
        "all_passed": all(model["passed"] for model in models.values()),
        "eligible_models": [method for method, model in models.items() if model["passed"]],
        "failed_models": [method for method, model in models.items() if not model["passed"]],
    }
    atomic_json(output_dir / "summary.json", payload)


if __name__ == "__main__":
    main()
