#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
from pathlib import Path


ARMS = {
    "ronpo_os": {"loss_type": "ronpo", "target_column": "target_os_k0p1"},
    "inpo_avg": {"loss_type": "inpo", "target_column": None},
    "sppo_avg": {"loss_type": "sppo", "target_column": None},
    "simpo": {"loss_type": "simpo", "target_column": None},
    "ipo": {"loss_type": "ipo", "target_column": None},
    "dpo": {"loss_type": "dpo", "target_column": None},
    "ht_mnpo_harmless": {"loss_type": "ht_mnpo", "target_column": "ht_target"},
    "ht_mnpo_helpfulness": {
        "loss_type": "ht_mnpo",
        "target_column": "ht_target_helpfulness",
    },
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_once(path, payload):
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != text:
        raise RuntimeError(f"locked file differs: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(text)
        path.with_suffix(".sha256").write_text(f"{sha(path)}  {path.name}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage1-source", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    args = parser.parse_args()
    required = (
        args.stage1_source / "precompute_history_base/targets/dataset_dict.json"
    )
    if sha(required) != "65fae6c706b1546da1530d427b62b39df6dd156cce4b92a208091410c52a4362":
        raise RuntimeError("Stage-1 shared dataset hash mismatch")
    for seed in args.seeds:
        seed_root = args.root / f"seed{seed}"
        stage12 = seed_root / "stage12"
        link = stage12 / "stage1/shared_pool"
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink():
            if link.resolve() != args.stage1_source.resolve():
                raise RuntimeError(f"Stage-1 link mismatch: {link}")
        elif link.exists():
            raise RuntimeError(f"Stage-1 link path already exists: {link}")
        else:
            os.symlink(args.stage1_source.resolve(), link)
        common = {
            "status": "locked_before_training",
            "seed": seed,
            "base": "meta-llama/Llama-3.1-8B-Instruct",
            "arms": ARMS,
            "stages": [1, 2, 3, 4],
            "steps_per_stage": 900,
            "effective_batch": 16,
            "learning_rate": 5e-7,
            "ronpo_alpha": 1.0,
            "ronpo_tau": 0.05,
            "eta": 0.0075,
            "reference_anchor_weight": 0.05,
            "preference_sft_weight": 0.005,
            "checkpoint_rule": "20-step smoke then final 900-step checkpoint",
            "stage1_dataset_dict_sha256": sha(required),
            "report_regardless_of_outcome": True,
            "spent_sealed_split_touched": False,
        }
        write_once(stage12 / "run_lock.json", common)
        write_once(
            seed_root / "stage3/continuation_lock.json",
            {**common, "stage": "stage3", "parent_root": str(stage12)},
        )
        write_once(
            seed_root / "stage4/continuation_lock.json",
            {
                **common,
                "stage": "stage4",
                "parent_root": str(seed_root / "stage3"),
            },
        )
    print(json.dumps({"status": "initialized", "seeds": args.seeds, "root": str(args.root)}))


if __name__ == "__main__":
    main()
