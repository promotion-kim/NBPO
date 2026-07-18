#!/usr/bin/env python3
"""Lock a matched Stage-4 continuation before any new optimization."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ARMS = {
    "ronpo_os_stage4": ("ronpo", "target_os_k0p1", "ronpo_os_stage3"),
    "ronpo_topmass_stage4": ("ronpo", "target_topmass_k0p1", "ronpo_topmass_stage3"),
    "inpo_avg_stage4": ("inpo", None, "inpo_avg_stage3"),
    "sppo_avg_stage4": ("sppo", None, "sppo_avg_stage3"),
    "simpo_stage4": ("simpo", None, "simpo_stage3"),
    "ipo_stage4": ("ipo", None, "ipo_stage3"),
    "dpo_stage4": ("dpo", None, "dpo_stage3"),
    "ht_mnpo_harmless_stage4": ("ht_mnpo", "ht_target", "ht_mnpo_harmless_stage3"),
    "ht_mnpo_helpfulness_stage4": ("ht_mnpo", "ht_target_helpfulness", "ht_mnpo_helpfulness_stage3"),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p7", type=Path, required=True)
    parser.add_argument("--fresh-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite Stage-4 run lock: {args.output}")
    stage3_audit = json.loads((args.p7 / "stage3_eval" / "pool_audit.json").read_text())
    parents = {parent for _, _, parent in ARMS.values()}
    if not parents.issubset(set(stage3_audit["eligible_models"])):
        raise RuntimeError("all Stage-3 parents must have passed the fresh full-panel stability gate")
    records = []
    for arm, (loss_type, target, parent) in ARMS.items():
        config = args.p7 / "stage3" / parent / "train" / "full" / "config.yaml"
        model = args.p7 / "stage3" / parent / "train" / "full"
        if not (model / "config.json").is_file() or not config.is_file():
            raise RuntimeError(f"missing parent artifact for {arm}")
        records.append({"arm": arm, "loss_type": loss_type, "target_column": target, "parent_stage3_arm": parent, "parent_model": str(model), "parent_config_sha256": sha(config)})
    payload = {
        "status": "locked_before_training",
        "stage": "stage4",
        "continuation_reason": "P7 Stage-3's RONPO-OS point estimate ranked first on its fresh panel, but its paired Worst confidence interval versus IPO crossed zero. The user requested one further stage-matched continuation before considering new settings.",
        "base": "meta-llama/Llama-3.1-8B-Instruct",
        "objective_protocol": "SafeRLHF Beaver reward helpfulness and negated Beaver cost harmlessness, unchanged from P5/P6b/P7.",
        "arms": records,
        "matched_budget": {"steps": 900, "effective_batch": 16, "seed": 42, "learning_rate": 5e-7, "ronpo_alpha": 1.0, "ronpo_tau": 0.05, "eta": 0.0075, "reference_anchor_weight": 0.05, "preference_sft_weight": 0.005, "cosine": True, "warmup_ratio": 0.1, "microbatch": 1, "gradient_checkpointing": True, "sdpa": True, "cudnn_sdpa": False},
        "pool_rule": "Each arm refreshes a base-plus-own-Stage-3-parent mixture on the unchanged 2,500 training conflict prompts. Base remains the fixed reference and the arm's Stage-3 parent is history0.",
        "checkpoint_rule": "final 900-step checkpoint only; no per-arm metric checkpoint selection or retry",
        "fresh_evaluation_manifest": str(args.fresh_manifest),
        "fresh_evaluation_manifest_sha256": sha(args.fresh_manifest),
        "decode": {"backend": "vllm", "seed": 42, "temperature": 0.7, "top_p": 0.9, "max_new_tokens": 512, "bfloat16": True, "enable_thinking": False},
        "stability_gate": {"detector": "corrected_nonempty_paired_span_v1", "records": 1000, "empty": 0, "think_leakage": 0, "length_ratio": [0.33, 2.0], "max_repeat_run": 20, "fail_closed": True},
        "goal_rule": {"worst": "RONPO is eligible and has paired prompt-level Worst minus the best eligible non-RONPO trained arm bootstrap-95% lower bound > 0", "average": "RONPO Avg minus the best eligible non-RONPO trained arm has bootstrap-95% lower bound > -0.02"},
        "wandb": {"entity": "promotion-kim", "project": "mnpo", "mode": "online", "required_for_training": True},
        "hf_policy": "Only final, full-panel stability-passing RONPO checkpoints are public uploaded and remotely verified before any local optimizer/intermediate cleanup.",
        "spent_sealed_split_touched": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output.with_suffix(".sha256").write_text(f"{sha(args.output)}  {args.output.name}\n", encoding="utf-8")
    args.output.with_name("PREREG.md").write_text(
        "# P8 Stage-4 matched continuation\n\n"
        "This protocol was locked before Stage-4 optimization. All nine methods advance from their own full-panel stability-passing Stage-3 parent. They receive identical 900-step, seed-42, effective-batch-16 budgets, the same training prompts, base reference, decode settings, gates, final-checkpoint rule, and evaluator. P7 was used only to trigger the predeclared Stage-4 continuation; the prompt-disjoint P8 fresh panel is the sole Stage-4 evaluation panel.\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
