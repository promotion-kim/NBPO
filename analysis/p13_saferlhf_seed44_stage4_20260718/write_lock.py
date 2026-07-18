#!/usr/bin/env python3
"""Lock and initialize the matched seed-44 SafeRLHF Stage-1-to-4 run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


ARMS = {
    "ronpo_os": ["ronpo", "target_os_k0p1"],
    "ronpo_topmass": ["ronpo", "target_topmass_k0p1"],
    "inpo_avg": ["inpo", None],
    "sppo_avg": ["sppo", None],
    "simpo": ["simpo", None],
    "ipo": ["ipo", None],
    "dpo": ["dpo", None],
    "ht_mnpo_harmless": ["ht_mnpo", "ht_target"],
    "ht_mnpo_helpfulness": ["ht_mnpo", "ht_target_helpfulness"],
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_once(path: Path, payload: dict) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise RuntimeError(f"locked file differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.with_suffix(".sha256").write_text(f"{sha(path)}  {path.name}\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--source-seed43", type=Path, required=True)
    p.add_argument("--stage12", type=Path, required=True)
    p.add_argument("--stage3", type=Path, required=True)
    p.add_argument("--stage4", type=Path, required=True)
    a = p.parse_args()
    source = a.source_seed43 / "stage1/shared_pool"
    required = source / "precompute_history_base/targets/dataset_dict.json"
    if not required.is_file():
        raise RuntimeError(f"missing byte-identical Stage-1 pool: {required}")
    link = a.stage12 / "stage1/shared_pool"
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() and link.resolve() == source.resolve():
        pass
    elif link.exists() or link.is_symlink():
        raise RuntimeError(f"refusing to replace Stage-1 pool path: {link}")
    else:
        os.symlink(source.resolve(), link)
    common = {
        "status": "locked_before_training",
        "seed": 44,
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
        "checkpoint_rule": "20-step smoke followed by final 900-step checkpoint; no metric selection and no per-arm retry",
        "stage1_pool": {"path": str(source), "dataset_dict_sha256": sha(required)},
        "continuation_pool": "same 2500 SafeRLHF conflict prompts; base plus own preceding-stage policy; base reference anchor",
        "stability_gate": "full locked 1000-prompt panel, fail-closed corrected detector",
        "wandb": {"mode": "online", "entity": "promotion-kim", "project": "mnpo", "required": True},
        "hf": {"visibility": "public", "upload": "every completed gate-passing stage under one repo per arm", "remote_hash_verification_before_prune": True},
        "spent_sealed_split_touched": False,
    }
    write_once(a.stage12 / "run_lock.json", {**common, "root": str(a.stage12)})
    write_once(a.stage3 / "continuation_lock.json", {**common, "stage": "stage3", "parent_root": str(a.stage12)})
    write_once(a.stage4 / "continuation_lock.json", {**common, "stage": "stage4", "parent_root": str(a.stage3)})
    prereg = (
        "# SafeRLHF seed-44 Stage-1-to-4 replication\n\n"
        "Before optimization, this run fixes the nine seed-42/43 Table-4 arms, seed 44, four 900-step stages, "
        "the byte-identical Stage-1 pair/log-probability dataset, each arm's own on-policy continuation pools, "
        "the final-checkpoint rule, and the full-panel reward-blind stability gate. W&B online logging is mandatory. "
        "Every gate-passing stage is uploaded to a public Hugging Face repository and verified before exact local "
        "weight files are pruned. The spent 604-prompt split is never touched.\n"
    )
    prereg_path = a.stage12 / "PREREG.md"
    if prereg_path.exists() and prereg_path.read_text(encoding="utf-8") != prereg:
        raise RuntimeError("existing PREREG differs")
    prereg_path.write_text(prereg, encoding="utf-8")
    print(json.dumps({"lock": str(a.stage12 / "run_lock.json"), "sha256": sha(a.stage12 / "run_lock.json")}, indent=2))


if __name__ == "__main__":
    main()
