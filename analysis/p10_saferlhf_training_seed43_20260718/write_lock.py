#!/usr/bin/env python3
"""Freeze the Stage-1 part of the SafeRLHF training-seed replication."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ARMS = {
    "ronpo_os": ("ronpo", "target_os_k0p1"),
    "ronpo_topmass": ("ronpo", "target_topmass_k0p1"),
    "inpo_avg": ("inpo", None),
    "sppo_avg": ("sppo", None),
    "simpo": ("simpo", None),
    "ipo": ("ipo", None),
    "dpo": ("dpo", None),
    "ht_mnpo_harmless": ("ht_mnpo", "ht_target"),
    "ht_mnpo_helpfulness": ("ht_mnpo", "ht_target_helpfulness"),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p4", type=Path, required=True)
    parser.add_argument("--p8", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("refusing to overwrite a locked P10 protocol")
    pairs = args.p4 / "train_pool"
    required = [
        pairs / "pairs_train.jsonl",
        pairs / "pairs_test.jsonl",
        args.p4 / "dataset_manifest" / "train_conflict.jsonl",
    ]
    if not all(path.is_file() for path in required):
        raise RuntimeError("missing frozen P4 Stage-1 source artifacts")
    fresh = args.p8 / "dataset_manifest" / "fresh_default_test_1000.jsonl"
    if not fresh.is_file():
        raise RuntimeError("missing P8 held-out panel manifest")
    payload = {
        "status": "locked_before_training",
        "purpose": "training-seed-43 replication of the SafeRLHF Stage-1 comparison, followed by stage-matched continuations when complete",
        "scope_limit": "The fixed P4 pair dataset is reused. This tests training stochasticity, not a resampled-data effect. The P8 1,000-prompt panel was already opened by P8 and may only be reused as a preregistered held-out seed-comparison panel; it is not a fresh confirmation and cannot select configurations.",
        "base": "meta-llama/Llama-3.1-8B-Instruct",
        "objectives": {
            "helpfulness": "PKU-Alignment/beaver-7b-v1.0-reward@375cd6a9f0d7e339d2199b05ba129a4a8906596d",
            "harmlessness": "negative PKU-Alignment/beaver-7b-v1.0-cost@c1bd343d2ddc2cb810bd736563c7ad0bf38f6b28",
        },
        "source_dataset": {str(path): sha(path) for path in required},
        "heldout_panel": {"path": str(fresh), "sha256": sha(fresh), "records": 1000, "fresh_for_this_seed": False},
        "arms": [{"arm": name, "loss_type": values[0], "target_column": values[1]} for name, values in ARMS.items()],
        "matched_budget": {
            "stages": [1, 2, 3, 4], "stage1_seed": 43, "steps_per_stage": 900,
            "effective_batch": 16, "learning_rate": 5e-7, "ronpo_alpha": 1.0,
            "ronpo_tau": 0.05, "eta": 0.0075, "reference_anchor_weight": 0.05,
            "preference_sft_weight": 0.005, "cosine": True, "warmup_ratio": 0.1,
            "microbatch": 1, "gradient_checkpointing": True, "sdpa": True, "cudnn_sdpa": False,
        },
        "selection": {"checkpoint_rule": "final 900-step checkpoint for every arm", "no_per_arm_retry": True, "no_metric_based_checkpoint_selection": True},
        "stability_gate": {"corrected_detector": "scripts/revision/flagship/stability_gate_corrected.py", "fail_closed": True, "full_panel_required": True},
        "wandb": {"entity": "promotion-kim", "project": "mnpo", "mode": "online", "required": True},
        "hf_policy": "Only final stability-passing RONPO models with completed paper-critical evaluations are public-uploaded and verified before local pruning.",
        "spent_sealed_split_touched": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output.with_suffix(".sha256").write_text(f"{sha(args.output)}  {args.output.name}\n", encoding="utf-8")
    args.output.with_name("PREREG.md").write_text(
        "# P10 SafeRLHF training-seed-43 replication\n\n"
        "This protocol was locked before any P10 optimization. All nine P8 Table-4 methods begin anew from the common Llama-3.1-8B base policy with training seed 43, using the byte-identical P4 Stage-1 pairs and log-probability procedure. Each receives the same final-checkpoint 900-step budget and then, if resources permit, advances through its own stage-matched parent/base pool for stages 2--4. The fixed pair dataset makes this a replication of optimizer and trajectory randomness, not a data-resampling study. The P8 1,000-prompt panel has already been opened and will be reused only for a preregistered held-out seed comparison after all compared Stage-4 models are complete; it cannot guide selection.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
