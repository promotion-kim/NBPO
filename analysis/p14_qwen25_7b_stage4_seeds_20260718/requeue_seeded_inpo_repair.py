#!/usr/bin/env python3
"""Audit the collapsed INPO Stage-1 artifacts and requeue the locked repair."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def move(source: Path, target: Path) -> None:
    if not source.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise RuntimeError(f"audit target exists: {target}")
    shutil.move(str(source), str(target))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    scheduler = args.root / "scheduler"
    audit = args.root / "audit_seeded_inpo_repair"
    repaired = []
    for seed in (43, 44):
        gate = args.root / f"seeds/s{seed}/stage1/gates/inpo_avg.json"
        measured = json.loads(gate.read_text(encoding="utf-8"))
        if measured.get("status") != "failed" or measured.get("passed") is not False:
            raise RuntimeError(f"INPO is not a measured gate failure: seed {seed}")
        checks = measured.get("checks", {})
        if checks.get("length_near_base") is not False:
            raise RuntimeError(f"unexpected INPO failure mechanism: seed {seed}")
        move(args.root / f"seeds/s{seed}/stage1/inpo_avg/train", audit / f"seeds/s{seed}/stage1/inpo_avg/train")
        move(args.root / f"seeds/s{seed}/stage1/generations/inpo_avg", audit / f"seeds/s{seed}/stage1/generations/inpo_avg")
        move(gate, audit / f"seeds/s{seed}/stage1/gates/inpo_avg.json")
        for stage, suffix in ((1, "FAILED"), (2, "BLOCKED"), (3, "BLOCKED"), (4, "BLOCKED")):
            status = scheduler / f"s{seed}__stage{stage}__inpo_avg.{suffix}.json"
            move(status, audit / "scheduler" / status.name)
        move(scheduler / "claims" / f"s{seed}__stage1__inpo_avg", audit / "scheduler/claims" / f"s{seed}__stage1__inpo_avg")
        repaired.append(seed)
    print(json.dumps({"status": "requeued", "arm": "inpo_avg", "seeds": repaired}, indent=2))


if __name__ == "__main__":
    main()
