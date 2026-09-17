#!/usr/bin/env python3
"""Materialize and launch ONE arm of the Section-7 neural-realization sweep.

Thin on purpose: everything that decides whether the run is legitimate lives in
``scripts/nbpo/run_nbpo_stage.materialize_run_config`` -- the NBPO trainer
invariants (``loss_type=nbpo``, ``logp_reduction=sum``, one history term, no
auxiliary losses), the artifact hashes that bind this fit to this stage's pairs,
solver and precompute, and a parse with run_mnpo's own dataclasses before any
weights load. This script only chooses eta, the learning rate and the step count.

``dataset_splits`` is deliberately ``["train"]`` even though the precomputed
artifact also holds the held-out split: the trainer must never see it, and a
checkpoint must never be selected on it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from mnpo_scripts.precompute_provenance import sha256_file_hex  # noqa: E402
from scripts.nbpo.run_nbpo_stage import materialize_run_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs-dir", type=Path, required=True)
    ap.add_argument("--solver-dir", type=Path, required=True)
    ap.add_argument("--precomputed", type=Path, required=True)
    ap.add_argument("--parent", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--run-config", type=Path, required=True)
    ap.add_argument("--eta", type=float, required=True)
    ap.add_argument("--learning-rate", type=float, default=5.0e-7)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--max-grad-norm", type=float, default=1.0,
                    help="the frozen config uses 1.0; raising it is a DIAGNOSTIC, "
                         "because pre-clip norms here are ~1e4 and a clip that tight "
                         "makes every update a fixed-size normalized step")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()

    meta = json.loads((args.precomputed / "precompute_meta.json").read_text())
    stage_cfg = {
        "stage": 0,
        "trainer": {
            "eta": args.eta,
            "optim": "adafactor",
            "learning_rate": args.learning_rate,
            "lr_scheduler_type": "cosine",
            # transformers 5.x dropped `warmup_ratio`. For a fixed `max_steps`
            # the two are exactly equivalent, and the config's 0.1 is written
            # out as a step count rather than silently dropped.
            "warmup_steps": round(0.1 * args.max_steps),
            "weight_decay": 0.0,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 8,
            "max_steps": args.max_steps,
            "max_length": 2048,
            "max_prompt_length": 1024,
            "max_grad_norm": args.max_grad_norm,
            "bf16": True,
            "gradient_checkpointing": True,
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
            "attn_implementation": "sdpa",
            "beta": 10.0,
            "seed": args.seed,
            "logging_steps": 10,
            "run_name": args.run_name or args.out_dir.name,
        },
    }
    expected = {
        "nbpo_expected_pair_artifact_sha256":
            sha256_file_hex(str(args.pairs_dir / "pairs_train.jsonl")),
        "nbpo_expected_solver_artifact_sha256":
            sha256_file_hex(str(args.solver_dir / "solution.json")),
        "nbpo_expected_precompute_manifest_sha256":
            meta.get("precompute_manifest_sha256"),
        "nbpo_expected_tokenization_config_sha256":
            meta.get("tokenization_config_sha256"),
    }
    run = materialize_run_config(stage_cfg, args.parent, args.precomputed,
                                args.out_dir, args.run_config,
                                dataset_splits=["train"], expected_hashes=expected)
    print(json.dumps({k: run[k] for k in
                      ("model_name_or_path", "loss_type", "logp_reduction", "eta",
                       "learning_rate", "max_steps", "warmup_steps", "output_dir",
                       "dataset_splits", "nbpo_target_column")}, indent=2))


if __name__ == "__main__":
    main()
