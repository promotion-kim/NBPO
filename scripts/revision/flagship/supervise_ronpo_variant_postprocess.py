#!/usr/bin/env python3
"""Wait for the measured search outcome and upload only an authorized winner."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_env(path: Path, environment: dict[str, str]) -> bool:
    if not path.is_file():
        return False
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        environment[key.strip()] = value.strip().strip("'\"")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--hf-env", type=Path, required=True)
    parser.add_argument("--deadline", required=True)
    args = parser.parse_args()
    deadline = datetime.fromisoformat(args.deadline)
    summary_path = args.run_dir / "summary.json"
    while True:
        if datetime.now().astimezone() >= deadline:
            atomic_json(args.run_dir / "postprocess_status.json", {
                "status": "deadline_before_final_summary",
                "spent_sealed_split_touched": False,
            })
            return
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            time.sleep(20)
            continue
        if summary.get("status") == "completed":
            break
        time.sleep(20)
    if not summary.get("upload_authorized"):
        atomic_json(args.run_dir / "postprocess_status.json", {
            "status": "completed_no_upload",
            "outcome": summary.get("outcome"),
            "reason": "Measured success rule did not authorize an upload.",
            "spent_sealed_split_touched": False,
        })
        return
    environment = os.environ.copy()
    env_loaded = load_env(args.hf_env, environment)
    environment["PYTHONPATH"] = str(args.project)
    command = [
        args.python,
        str(args.project / "scripts/revision/flagship/upload_ronpo_variant_winner.py"),
        "--run-dir", str(args.run_dir), "--namespace", "promotion",
    ]
    log_path = args.run_dir / "hf_upload.log"
    with log_path.open("a", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=args.project, env=environment,
                                stdout=handle, stderr=subprocess.STDOUT)
    payload = {
        "status": "completed_verified_upload" if result.returncode == 0 else "failed_upload",
        "outcome": summary.get("outcome"),
        "returncode": result.returncode,
        "hf_env_loaded": env_loaded,
        "auth_fallback": None if env_loaded else "verified cached promotion login",
        "log": str(log_path),
        "spent_sealed_split_touched": False,
    }
    atomic_json(args.run_dir / "postprocess_status.json", payload)
    if result.returncode:
        raise RuntimeError(f"winner upload failed; see {log_path}")


if __name__ == "__main__":
    main()
