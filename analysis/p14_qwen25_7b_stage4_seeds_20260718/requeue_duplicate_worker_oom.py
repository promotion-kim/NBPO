#!/usr/bin/env python3
"""Requeue only continuation tasks that failed from a verified duplicate-worker OOM."""

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
    audit = scheduler / "audit_duplicate_worker_oom"
    repaired = []
    for seed in (43, 44):
        for arm in ("ronpo_os", "sppo_avg", "dpo"):
            name = f"s{seed}__stage2__{arm}"
            failed = scheduler / f"{name}.FAILED.json"
            payload = json.loads(failed.read_text(encoding="utf-8"))
            pool_log = args.root / f"seeds/s{seed}/stage2/{arm}/pool/logs/decode_parent_seed42.log"
            text = pool_log.read_text(encoding="utf-8", errors="replace")
            marker = "Free memory on device cuda:0"
            if payload.get("returncode") != 1 or marker not in text or "less than desired GPU memory utilization" not in text:
                raise RuntimeError(f"not the verified duplicate-worker OOM: {name}")
            move(args.root / f"seeds/s{seed}/stage2/{arm}/pool", audit / f"partial/seeds/s{seed}/stage2/{arm}/pool")
            for stage in (3, 4):
                blocked = scheduler / f"s{seed}__stage{stage}__{arm}.BLOCKED.json"
                move(blocked, audit / "status" / blocked.name)
            move(scheduler / "claims" / name, audit / "claims" / name)
            move(failed, audit / "status" / failed.name)
            repaired.append(name)
    print(json.dumps({"status": "requeued", "tasks": repaired}, indent=2))


if __name__ == "__main__":
    main()
