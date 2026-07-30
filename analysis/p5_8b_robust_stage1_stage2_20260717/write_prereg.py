#!/usr/bin/env python3
"""Write the reward-blind Stage-1/Stage-2 lock before new optimisation."""

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
    parser.add_argument("--existing-precompute", type=Path, required=True)
    parser.add_argument("--topmass-column", required=True)
    parser.add_argument("--softmin-column", required=True)
    parser.add_argument("--softmin-tau", type=float, required=True)
    args = parser.parse_args()
    arms = [
        {"name": "ronpo_topmass_stage1_replicate", "target_column": args.topmass_column, "role": "exact hard-argmax Stage-1 replication"},
        {"name": "ronpo_softmin_lb_stage1", "target_column": args.softmin_column, "role": "smooth minimum lower-bound ablation"},
    ]
    payload = {
        "status": "locked_before_training",
        "base": "meta-llama/Llama-3.1-8B-Instruct",
        "objective_protocol": "SafeRLHF Beaver reward versus negated Beaver cost, identical to p4",
        "source_precompute": str(args.existing_precompute),
        "source_precompute_dataset_dict_sha256": sha(args.existing_precompute / "dataset_dict.json"),
        "matched_budget": {"steps": 900, "seed": 42, "effective_batch": 16, "learning_rate": 5e-7, "ronpo_alpha": 1.0, "ronpo_tau": 0.05, "reference_anchor_weight": 0.05, "preference_sft_weight": 0.005},
        "arms": arms,
        "softmin_lower_bound": {"tau": args.softmin_tau, "formula": "-tau*log(sum_k exp(-g_k/tau))", "bound": "min(g)-tau*log(K) <= target <= min(g)", "scope": "exact target inequality; no global non-convex optimisation claim"},
        "stage2": {"rule": "Each method refreshes a stage-matched pool from its own eligible Stage-1 parent, retains base as reference, and uses its own Stage-1 parent as history0. No Stage-2 model is compared against Stage-1 models."},
        "selection": {"checkpoint_rule": "final model only", "evaluation": "frozen 49-prompt p4 validation used only for descriptive comparison; any new headline needs a separately preregistered disjoint test"},
        "spent_sealed_split_touched": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output.with_suffix(".sha256").write_text(sha(args.output) + "  " + args.output.name + "\n", encoding="utf-8")
    args.output.parent.joinpath("arms.json").write_text(json.dumps({"arms": arms}, indent=2) + "\n", encoding="utf-8")
    args.output.parent.joinpath("PREREG.md").write_text(
        "# Stage-1 robustness repair and stage-matched Stage-2 continuation\n\n"
        "This lock was written before new training. The top-mass arm is an exact 900-step replication on the frozen p4 SafeRLHF precompute. The only new Stage-1 method is the smooth worst-objective lower-bound target. For each pair let `g_k` be the uniform-opponent expected RONPO advantage for objective `k`; its target is `-tau log sum_k exp(-g_k/tau)`, with `tau=0.1`. For finite `g`, `min(g)-tau log K <= target <= min(g)`. This is an algebraic lower-bound guarantee, not a claim of globally solving the neural optimisation problem.\n\n"
        "Stage-2 must advance every eligible comparator from its own Stage-1 parent, using a refreshed parent/base response mixture and history0 equal to that parent. Results are reported only among stage-matched models. The spent 604-prompt sealed split is excluded.\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
