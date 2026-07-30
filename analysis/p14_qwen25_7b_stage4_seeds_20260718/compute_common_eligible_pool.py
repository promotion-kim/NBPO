#!/usr/bin/env python3
"""Lock the identical Stage-4 evaluation pool shared by all training seeds."""

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=(42, 43, 44))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    details, common = {}, []
    for arm in args.arms:
        seed_status = {}
        for seed in args.seeds:
            gate = args.root / f"seeds/s{seed}/stage4/gates/{arm}.json"
            generation = args.root / f"seeds/s{seed}/stage4/generations/{arm}/output_42.json"
            passed, records, error = False, None, None
            try:
                gate_payload = json.loads(gate.read_text(encoding="utf-8"))
                records = len(json.loads(generation.read_text(encoding="utf-8")))
                passed = gate_payload.get("passed") is True and records == 1000
                if not passed:
                    error = f"gate_passed={gate_payload.get('passed')!r}, records={records}"
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            seed_status[str(seed)] = {
                "eligible": passed,
                "records": records,
                "gate": str(gate),
                "generation": str(generation),
                "gate_sha256": sha256(gate) if gate.is_file() else None,
                "generation_sha256": sha256(generation) if generation.is_file() else None,
                "error": error,
            }
        details[arm] = seed_status
        if all(item["eligible"] for item in seed_status.values()):
            common.append(arm)
    base = args.root / "shared/eval/base/output_42.json"
    base_gate = args.root / "shared/eval/base/gate.json"
    base_ok = (
        len(json.loads(base.read_text(encoding="utf-8"))) == 1000
        and json.loads(base_gate.read_text(encoding="utf-8")).get("passed") is True
    )
    if not base_ok:
        raise RuntimeError("shared base generation failed integrity or stability gate")
    if not common:
        raise RuntimeError("no trained method is eligible across all seeds")
    payload = {
        "status": "locked",
        "rule": "intersection of 1000-record, unchanged stability-gate passers across training seeds",
        "training_seeds": args.seeds,
        "eligible_models": ["base", *common],
        "common_trained_arms": common,
        "excluded_arms": [arm for arm in args.arms if arm not in common],
        "per_seed_evidence": details,
        "base": {"generation_sha256": sha256(base), "gate_sha256": sha256(base_gate)},
        "spent_sealed_split_touched": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("\n".join(common))


if __name__ == "__main__":
    main()
