#!/usr/bin/env python3
"""Requeue verified Stage-2 smoke failures caused only by the polymer tokenizer parser."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


ARMS = ("dpo", "ht_mnpo_harmless", "ht_mnpo_helpfulness", "ipo", "simpo", "sppo_avg")
MARKER = "data did not match any variant of untagged enum ModelWrapper"


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
    audit = args.root / "audit_polymer_tokenizer_parser"
    tasks = []
    for seed in (43, 44):
        for arm in ARMS:
            name = f"s{seed}__stage2__{arm}"
            failed = scheduler / f"{name}.FAILED.json"
            payload = json.loads(failed.read_text(encoding="utf-8"))
            status = args.root / f"seeds/s{seed}/stage2/{arm}/train/smoke/job_status.json"
            measured = json.loads(status.read_text(encoding="utf-8"))
            log = Path(measured["log"])
            if payload.get("returncode") != 1 or measured.get("status") != "failed" or MARKER not in log.read_text(encoding="utf-8", errors="replace"):
                raise RuntimeError(f"not the locked tokenizer-parser failure: {name}")
            if not (args.root / f"seeds/s{seed}/stage2/{arm}/pool/PREPARED").is_file():
                raise RuntimeError(f"pool is not complete: {name}")
            if (args.root / f"seeds/s{seed}/stage2/{arm}/train/full/job_status.json").exists():
                raise RuntimeError(f"full training unexpectedly started: {name}")
            move(args.root / f"seeds/s{seed}/stage2/{arm}/train", audit / f"seeds/s{seed}/stage2/{arm}/train")
            for stage in (3, 4):
                blocked = scheduler / f"s{seed}__stage{stage}__{arm}.BLOCKED.json"
                move(blocked, audit / "scheduler" / blocked.name)
            move(scheduler / "claims" / name, audit / "scheduler/claims" / name)
            move(failed, audit / "scheduler" / failed.name)
            tasks.append(name)
    print(json.dumps({"status": "requeued", "cause": "polymer_tokenizer_parser", "tasks": tasks}, indent=2))


if __name__ == "__main__":
    main()
