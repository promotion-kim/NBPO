#!/usr/bin/env python3
"""Audit and requeue tasks whose container disappeared mid-stage."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path


def completed(path: Path) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("status") == "completed" and data.get("finite_metrics") is True
    except (FileNotFoundError, json.JSONDecodeError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--owners", nargs="+", required=True)
    args = parser.parse_args()
    scheduler = args.root / "scheduler"
    stamp = time.strftime("%Y%m%dT%H%M%S")
    audit = args.root / "audit_container_rerun" / stamp
    records = []
    for claim in sorted((scheduler / "claims").iterdir()):
        if not claim.is_dir():
            continue
        name = claim.name
        if any((scheduler / f"{name}.{state}.json").exists() for state in ("DONE", "FAILED", "BLOCKED")):
            continue
        owner = json.loads((claim / "owner.json").read_text(encoding="utf-8"))
        owner_host = str(owner.get("worker", "")).rsplit("-g", 1)[0]
        if owner_host not in args.owners:
            continue
        fields = name.split("__")
        seed = int(fields[0][1:])
        stage = int(fields[1].replace("stage", ""))
        arm = fields[2]
        task = args.root / f"seeds/s{seed}/stage{stage}/{arm}"
        moved = []
        full = task / "train/full"
        if full.exists() and not completed(full / "job_status.json"):
            target = audit / f"seeds/s{seed}/stage{stage}/{arm}/train/full"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(full), str(target))
            moved.append({"source": str(full), "audit": str(target), "reason": "incomplete_full_training"})
        pool = task / "pool"
        if pool.exists() and not (pool / "PREPARED").is_file():
            target = audit / f"seeds/s{seed}/stage{stage}/{arm}/pool"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(pool), str(target))
            moved.append({"source": str(pool), "audit": str(target), "reason": "incomplete_pool"})
        claim_target = audit / "scheduler/claims" / name
        claim_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(claim), str(claim_target))
        records.append({"task": name, "owner": owner, "moved": moved, "claim_audit": str(claim_target)})
    audit.mkdir(parents=True, exist_ok=True)
    report = {"status": "requeued", "at": time.time(), "owners": args.owners, "tasks": records}
    (audit / "recovery.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
