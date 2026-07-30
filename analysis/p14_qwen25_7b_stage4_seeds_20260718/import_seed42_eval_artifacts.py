#!/usr/bin/env python3
"""Import verified seed-42 gate generations into the shared three-seed run."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--main-root", type=Path, required=True)
    parser.add_argument("--repair-root", type=Path, required=True)
    args = parser.parse_args()
    sources = {
        **{arm: args.main_root for arm in ("ronpo_os", "ht_mnpo_helpfulness", "ht_mnpo_harmless", "simpo", "dpo")},
        **{arm: args.repair_root for arm in ("inpo_avg", "ipo")},
    }
    records = []
    for arm, source_root in sources.items():
        source_gen = source_root / f"seeds/s42/stage4/generations/{arm}/output_42.json"
        source_gate = source_root / f"seeds/s42/stage4/gates/{arm}.json"
        responses = json.loads(source_gen.read_text(encoding="utf-8"))
        gate = json.loads(source_gate.read_text(encoding="utf-8"))
        if len(responses) != 1000 or gate.get("passed") is not True:
            raise RuntimeError(f"invalid seed-42 artifact: {arm}")
        target_gen = args.root / f"seeds/s42/stage4/generations/{arm}/output_42.json"
        target_gate = args.root / f"seeds/s42/stage4/gates/{arm}.json"
        target_gen.parent.mkdir(parents=True, exist_ok=True)
        target_gate.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_gen, target_gen)
        shutil.copy2(source_gate, target_gate)
        if sha(source_gen) != sha(target_gen) or sha(source_gate) != sha(target_gate):
            raise RuntimeError(f"copy verification failed: {arm}")
        records.append({
            "arm": arm, "records": len(responses), "gate_passed": True,
            "generation_source": str(source_gen), "generation_sha256": sha(target_gen),
            "gate_source": str(source_gate), "gate_sha256": sha(target_gate),
        })
    output = args.root / "evaluation/s42_import_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"status": "verified", "models": records}, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
