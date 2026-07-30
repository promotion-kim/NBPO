#!/usr/bin/env python3
"""Lock P10's stage-matched continuation before any Stage-2 pool is built."""

from __future__ import annotations

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


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    root = Path("/NHNHOME/AIPR/sjkim/MNPO_rev_20260710")
    exp = root / "results/p10_saferlhf_training_seed43_20260718"
    source = root / "results/p4_8b_saferlhf_table4_20260717"
    payload = {
        "status": "locked_before_stage2_pool_or_training",
        "scope": "P10 seed-43 Stage-2 continuation. No reward ranking was consulted to choose an arm, parent, pool, or hyperparameter.",
        "stage_match": "Every listed method advances from its own completed P10 Stage-1 parent. The base model remains the reference anchor and that same Stage-1 parent is history0 during precompute.",
        "arms": ARMS,
        "seed": 43,
        "budget": {"steps": 900, "effective_batch": 16, "learning_rate": 5e-7, "ronpo_alpha": 1.0, "eta": 0.0075, "ronpo_tau": 0.05, "reference_anchor_weight": 0.05, "preference_sft_weight": 0.005},
        "pool": {
            "manifest": str(source / "dataset_manifest/train_conflict.jsonl"),
            "rows": 2500,
            "parent_responses": [42, 43],
            "base_responses": [44, 45],
            "decode": {"temperature": 0.7, "top_p": 0.9, "max_new_tokens": 512, "thinking": False},
            "method": "Each method gets its own stage-matched parent-plus-base pool; no Stage-2 model will be compared to a Stage-1 model.",
        },
        "wandb": {"mode": "online", "entity": "promotion-kim", "project": "mnpo"},
        "test_integrity": {"spent_604_prompt_split_touched": False, "p8_1000_prompt_panel": "not used for Stage-2 model selection"},
    }
    path = exp / "stage2_lock.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (exp / "stage2_lock.sha256").write_text(sha(path) + "  stage2_lock.json\n", encoding="utf-8")
    print(json.dumps({"lock": str(path), "sha256": sha(path)}, indent=2))


if __name__ == "__main__":
    main()
