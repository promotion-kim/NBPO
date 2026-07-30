#!/usr/bin/env python3
"""Write the eight locked covariance training configs."""

import argparse
from pathlib import Path

import yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()
    for condition in ("clean", "noise03"):
        for rule in ("nbs", "ks", "unif", "maxmin"):
            name = f"{condition}_{rule}"
            out = args.root / "train" / name
            out.mkdir(parents=True, exist_ok=True)
            suffix = "" if condition == "clean" else "_noise03"
            config = {
                "model_name_or_path": args.base,
                "attn_implementation": "sdpa",
                "dataset_mixer": {args.dataset: 1.0},
                "dataset_splits": ["train", "test"],
                "preprocessing_num_workers": 4,
                "bf16": True,
                "loss_type": "ht_mnpo",
                "eta": 0.0075,
                "ht_target_column": f"bpo_target_{rule}{suffix}",
                "ht_target_scale": 1.0,
                "max_history_t": 1,
                "history_weights": [1.0],
                "beta": 10.0,
                "reference_anchor_weight": 0.05,
                "preference_sft_weight": 0.005,
                "learning_rate": 5e-7,
                "lr_scheduler_type": "cosine",
                "warmup_ratio": 0.1,
                "optim": "adamw_torch",
                "weight_decay": 0.0,
                "max_grad_norm": 1.0,
                "seed": 42,
                "gradient_accumulation_steps": 16,
                "gradient_checkpointing": True,
                "gradient_checkpointing_kwargs": {"use_reentrant": False},
                "num_train_epochs": 100,
                "max_steps": 900,
                "per_device_train_batch_size": 1,
                "per_device_eval_batch_size": 1,
                "max_length": 1536,
                "max_prompt_length": 1024,
                "do_eval": False,
                "eval_strategy": "no",
                "logging_steps": 10,
                "save_strategy": "no",
                "save_only_model": True,
                "save_safetensors": True,
                "push_to_hub": False,
                "report_to": ["wandb"],
                "output_dir": str(out),
                "run_name": f"nbpo-cov-hh-{name}-s42",
            }
            (out / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))


if __name__ == "__main__":
    main()
