#!/usr/bin/env python3
"""Freeze the Qwen2.5-7B three-seed Table-4 replication before training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import ARMS, BASE_ID, BASE_REVISION, COST_ID, COST_REVISION, REWARD_ID, REWARD_REVISION, SEEDS, sha256


def write_once(path: Path, text: str) -> None:
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise RuntimeError(f"locked file differs: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.with_suffix(path.suffix + ".sha256").write_text(f"{sha256(path)}  {path.name}\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--eval-manifest", type=Path, required=True)
    args = parser.parse_args()
    expected = {
        args.train_manifest: (2500, "6296a9efa506e6b5fde1786f6c6c58d9df2d35e913f61e2f8bf671bbadad5f39"),
        args.eval_manifest: (1000, "c7b5d42f5b866d6c8fce8667cfb22d27541fa86eea61d4b9d2ad0dad7a12eec2"),
    }
    for path, (rows, digest) in expected.items():
        if sum(1 for line in path.open(encoding="utf-8") if line.strip()) != rows or sha256(path) != digest:
            raise RuntimeError(f"manifest integrity failure: {path}")
    payload = {
        "status": "locked_before_training",
        "study": "Table-4 SafeRLHF base-backbone replication",
        "base": {"id": BASE_ID, "revision": BASE_REVISION},
        "seeds": list(SEEDS),
        "stages": [1, 2, 3, 4],
        "arms": {name: {"loss_type": spec[0], "target_family": spec[1]} for name, spec in ARMS.items()},
        "objectives": {
            "helpfulness": {"id": REWARD_ID, "revision": REWARD_REVISION, "definition": "Beaver reward"},
            "harmlessness": {"id": COST_ID, "revision": COST_REVISION, "definition": "negative Beaver cost"},
        },
        "data": {
            "train_manifest": str(args.train_manifest), "train_sha256": sha256(args.train_manifest),
            "train_prompts": 2500, "pairs_per_prompt": 3,
            "evaluation_manifest": str(args.eval_manifest), "evaluation_sha256": sha256(args.eval_manifest),
            "evaluation_prompts": 1000,
        },
        "decode": {"pool_seeds": [42, 43, 44, 45], "evaluation_seed": 42, "temperature": 0.7, "top_p": 0.9, "max_new_tokens": 512, "dtype": "bfloat16"},
        "training": {"steps_per_stage": 900, "effective_batch": 16, "learning_rate": 5e-7, "eta": 0.0075, "ronpo_alpha": 1.0, "ronpo_tau": 0.05, "reference_anchor_weight": 0.05, "preference_sft_weight": 0.005, "warmup_ratio": 0.1, "scheduler": "cosine", "checkpoint_rule": "20-step smoke then final 900-step checkpoint; no metric selection or per-arm retry"},
        "ronpo_kappa": {"candidates": [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5], "selection": "nearest mean normalized sigma entropy to 0.55 on the shared Qwen base pool before training"},
        "stability_gate": {"records": 1000, "empty": 0, "nonempty_think_spans": 0, "length_ratio": [0.33, 2.0], "max_repeat_run": 20, "fail_closed": True},
        "wandb": {"mode": "online", "entity": "promotion-kim", "project": "mnpo", "required": True},
        "hf": {"visibility": "public", "upload": "every gate-passing stage", "verify_lfs_sha256_before_prune": True},
        "paper_edit": False,
        "spent_sealed_split_touched": False,
    }
    write_once(args.root / "run_lock.json", json.dumps(payload, indent=2, sort_keys=True) + "\n")
    prereg = (
        "# Qwen2.5-7B Table-4 replication\n\n"
        "Before training, this run fixes Qwen/Qwen2.5-7B-Instruct at the recorded revision, training seeds 42/43/44, "
        "the existing 2,500-prompt SafeRLHF conflict pool and 1,000-prompt Stage-4 evaluation panel, Beaver reward/cost objectives, "
        "nine matched arms, four 900-step stages, and the final-checkpoint rule. Qwen-specific generations and reference/history "
        "log-probabilities are recomputed. The RONPO kappa is selected once from sigma entropy on the shared base pool before any "
        "training outcome exists. W&B online logging and verified public Hugging Face uploads are mandatory.\n"
    )
    write_once(args.root / "PREREG.md", prereg)
    print(json.dumps({"run_lock": str(args.root / "run_lock.json"), "sha256": sha256(args.root / "run_lock.json")}, indent=2))


if __name__ == "__main__":
    main()
