#!/usr/bin/env python3
"""Freeze one matched seed-43 continuation before any pool or training work."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ARMS = {
    "ronpo_os": {"loss_type": "ronpo", "target_column": "target_os_k0p1"},
    "ronpo_topmass": {"loss_type": "ronpo", "target_column": "target_topmass_k0p1"},
    "inpo_avg": {"loss_type": "inpo", "target_column": None},
    "sppo_avg": {"loss_type": "sppo", "target_column": None},
    "simpo": {"loss_type": "simpo", "target_column": None},
    "ipo": {"loss_type": "ipo", "target_column": None},
    "dpo": {"loss_type": "dpo", "target_column": None},
    "ht_mnpo_harmless": {"loss_type": "ht_mnpo", "target_column": "ht_target"},
    "ht_mnpo_helpfulness": {"loss_type": "ht_mnpo", "target_column": "ht_target_helpfulness"},
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=["stage3", "stage4"], required=True)
    parser.add_argument("--parent-experiment", type=Path, required=True)
    parser.add_argument("--parent-stage", choices=["stage2", "stage3"], required=True)
    parser.add_argument("--required-gates", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite existing lock: {args.output}")
    parents = {
        arm: str(args.parent_experiment / args.parent_stage / arm / "train" / "full")
        for arm in ARMS
    }
    gate_records = {}
    if args.required_gates is not None:
        for arm in ARMS:
            gate = args.required_gates / "gates" / f"{arm}.json"
            if not gate.is_file():
                raise RuntimeError(f"missing required reward-blind stability gate: {gate}")
            data = json.loads(gate.read_text(encoding="utf-8"))
            if data.get("status") != "passed" or data.get("passed") is not True:
                raise RuntimeError(f"parent did not pass stability gate: {arm}")
            gate_records[arm] = {"path": str(gate), "sha256": digest(gate)}
    payload = {
        "status": "locked_before_pool_or_training",
        "stage": args.stage,
        "seed": 43,
        "arms": ARMS,
        "parent_stage": args.parent_stage,
        "parents": parents,
        "required_parent_stability_gates": gate_records,
        "matched_budget": {
            "steps": 900,
            "effective_batch": 16,
            "microbatch": 1,
            "learning_rate": 5e-7,
            "ronpo_alpha": 1.0,
            "ronpo_tau": 0.05,
            "eta": 0.0075,
            "reference_anchor_weight": 0.05,
            "preference_sft_weight": 0.005,
            "cosine": True,
            "warmup_ratio": 0.1,
            "gradient_checkpointing": True,
            "sdpa": True,
            "cudnn_sdpa": False,
        },
        "pool_rule": "Each arm refreshes a base-plus-own-parent response mixture on the same 2,500 SafeRLHF conflict training prompts. The base remains the reference anchor and the own parent is history0.",
        "checkpoint_rule": "Final 900-step checkpoint only. No outcome-dependent checkpoint choice, retry, or per-method hyperparameter change.",
        "wandb": {"mode": "online", "entity": "promotion-kim", "project": "mnpo"},
        "hf_policy": "After Stage-4 only, upload final stability-passing paper-critical checkpoints and verify their public revisions before removing only redundant local optimizer or intermediate artifacts.",
        "spent_sealed_split_touched": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output.with_suffix(".sha256").write_text(
        f"{digest(args.output)}  {args.output.name}\n", encoding="utf-8"
    )
    print(json.dumps({"lock": str(args.output), "sha256": digest(args.output)}, indent=2))


if __name__ == "__main__":
    main()
