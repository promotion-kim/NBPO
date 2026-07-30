#!/usr/bin/env python3
"""Replace only false gate-prerequisite failures after a measured gate pass."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--seed", type=int, choices=(43, 44), required=True)
    p.add_argument("--arm", required=True)
    args = p.parse_args()
    scheduler = args.root / "scheduler"
    name = f"s{args.seed}__stage1__{args.arm}"
    failed = scheduler / f"{name}.FAILED.json"
    gate = args.root / f"seeds/s{args.seed}/stage1/gates/{args.arm}.json"
    status = args.root / f"seeds/s{args.seed}/stage1/{args.arm}/train/full/job_status.json"
    task = load(failed)
    if task.get("returncode") != 1 or "gate prerequisites missing" not in Path(task["log"]).read_text(encoding="utf-8"):
        raise RuntimeError(f"not a gate-prerequisite failure: {failed}")
    if load(status).get("status") != "completed" or load(status).get("finite_metrics") is not True:
        raise RuntimeError(f"training is not valid: {status}")
    measured = load(gate)
    if measured.get("status") != "passed" or measured.get("passed") is not True:
        print(json.dumps({"status": "kept_failed", "seed": args.seed, "arm": args.arm, "gate": measured}))
        return
    audit = scheduler / "audit_gate_prereq_failure"
    audit.mkdir(parents=True, exist_ok=True)
    failed.rename(audit / failed.name)
    for stage in (2, 3, 4):
        blocked = scheduler / f"s{args.seed}__stage{stage}__{args.arm}.BLOCKED.json"
        if blocked.exists():
            blocked.rename(audit / blocked.name)
    task.update({
        "returncode": 0,
        "finished": time.time(),
        "repair": "restored missing shared/eval/base/output_42.json, then reran unchanged gate",
        "gate": str(gate),
    })
    (scheduler / f"{name}.DONE.json").write_text(json.dumps(task, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "repaired", "seed": args.seed, "arm": args.arm, "gate": str(gate)}))


if __name__ == "__main__":
    main()
