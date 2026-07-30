#!/usr/bin/env python3
"""Import one cross-container task and close its original scheduler record."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--destination", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--stage", type=int, required=True)
    p.add_argument("--arm", required=True)
    args = p.parse_args()
    rc_path = args.source / "dispatch/train_and_gate.rc"
    while not rc_path.is_file():
        time.sleep(20)
    name = f"s{args.seed}__stage{args.stage}__{args.arm}"
    scheduler = args.destination / "scheduler"
    if int(rc_path.read_text().strip()):
        payload = {"status": "failed_on_dispatched_host", "source": str(args.source), "at": time.time()}
        (scheduler / f"{name}.FAILED.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        raise SystemExit(1)
    source_stage = args.source / f"seeds/s{args.seed}/stage{args.stage}"
    gate = load(source_stage / f"gates/{args.arm}.json")
    if not (gate.get("passed") is True and gate.get("status") == "passed"):
        raise RuntimeError("dispatched model did not pass the locked stability gate")
    destination_stage = args.destination / f"seeds/s{args.seed}/stage{args.stage}"
    copies = [
        (source_stage / args.arm, destination_stage / args.arm),
        (source_stage / f"generations/{args.arm}", destination_stage / f"generations/{args.arm}"),
        (source_stage / f"gates/{args.arm}.json", destination_stage / f"gates/{args.arm}.json"),
    ]
    for source, destination in copies:
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["rsync", "-a", str(source), str(destination)], check=True)
    payload = {
        "status": "completed_on_dispatched_host",
        "source": str(args.source),
        "destination": str(args.destination),
        "gate": gate,
        "at": time.time(),
    }
    (args.destination / f"dispatch_import_{name}.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (scheduler / f"{name}.DONE.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
