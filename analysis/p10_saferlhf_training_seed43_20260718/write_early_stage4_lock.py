#!/usr/bin/env python3
"""Lock one gate-passing Stage-4 arm before launching it on an idle GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--loss-type", required=True)
    parser.add_argument("--target-column")
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--parent-gate", type=Path, required=True)
    args = parser.parse_args()
    status_path = args.parent / "job_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    gate = json.loads(args.parent_gate.read_text(encoding="utf-8"))
    if status.get("status") != "completed" or status.get("finite_metrics") is not True:
        raise RuntimeError("parent training is not complete and finite")
    if gate.get("status") != "passed" or gate.get("passed") is not True:
        raise RuntimeError("parent did not pass the frozen reward-blind gate")
    payload = {
        "status": "locked_before_stage4_pool_or_training",
        "scope": "early utilization of an idle authorized GPU; no outcome-dependent arm or recipe change",
        "stage": "stage4",
        "seed": 43,
        "arm": args.arm,
        "loss_type": args.loss_type,
        "target_column": None if args.target_column in (None, "-", "") else args.target_column,
        "parent": str(args.parent),
        "parent_status": {"path": str(status_path), "sha256": sha(status_path)},
        "required_parent_gate": {"path": str(args.parent_gate), "sha256": sha(args.parent_gate)},
        "matched_budget": {
            "steps": 900, "effective_batch": 16, "microbatch": 1,
            "learning_rate": 5e-7, "ronpo_alpha": 1.0, "ronpo_tau": 0.05,
            "eta": 0.0075, "reference_anchor_weight": 0.05,
            "preference_sft_weight": 0.005, "cosine": True,
            "warmup_ratio": 0.1, "gradient_checkpointing": True,
            "sdpa": True, "cudnn_sdpa": False,
        },
        "pool_rule": "base plus own Stage-3 parent on the locked 2,500 SafeRLHF conflict prompts",
        "checkpoint_rule": "20-step smoke followed by the final 900-step checkpoint; no per-arm retry or metric selection",
        "wandb": {"mode": "online", "entity": "promotion-kim", "project": "mnpo"},
        "spent_sealed_split_touched": False,
    }
    if args.output.exists():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(f"existing early lock differs from the requested lock: {args.output}")
        print(json.dumps({"lock": str(args.output), "sha256": sha(args.output), "reused": True}, indent=2))
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output.with_suffix(".sha256").write_text(f"{sha(args.output)}  {args.output.name}\n", encoding="utf-8")
    print(json.dumps({"lock": str(args.output), "sha256": sha(args.output)}, indent=2))


if __name__ == "__main__":
    main()
