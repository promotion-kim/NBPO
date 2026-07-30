#!/usr/bin/env python3
"""Lock the reward-blind P10 Stage-2 seed-43 evaluation before scoring."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


PROJECT = Path("/NHNHOME/AIPR/sjkim/MNPO_rev_20260710")
EXPERIMENT = PROJECT / "results/p10_saferlhf_training_seed43_20260718"
P8 = PROJECT / "results/p8_stage4_fresh_default_test_20260718"
MODELS = ["base", "ronpo_os", "inpo_avg", "sppo_avg", "ipo", "dpo"]
EXCLUDED = {
    "ronpo_topmass": "Stage-2 was not launched in the locked P10 continuation queue.",
    "simpo": "Stage-2 pool/training was not completed when this evaluation lock was written.",
    "ht_mnpo_harmless": "Stage-2 pool/training was not completed when this evaluation lock was written.",
    "ht_mnpo_helpfulness": "Stage-2 pool/training was not completed when this evaluation lock was written.",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    target = EXPERIMENT / "stage2_eval_lock.json"
    if target.exists():
        raise RuntimeError(f"refusing to overwrite existing lock: {target}")
    manifest = P8 / "dataset_manifest/fresh_default_test_1000.jsonl"
    base = P8 / "stage4_eval/generations/base/output_42.json"
    if not manifest.is_file() or not base.is_file():
        raise RuntimeError("missing already-open P8 locked panel or base generation")
    paths = {"base": str(base)}
    for arm in MODELS[1:]:
        paths[arm] = str(EXPERIMENT / "stage2" / arm / "train/full")
    payload = {
        "status": "locked_before_reward_scoring",
        "scope": "P10 optimizer-seed-43, Stage-2, same 1,000-prompt P8 panel already opened by P8. This is a preregistered seed comparison, not model selection or a fresh confirmation.",
        "training_seed": 43,
        "models": paths,
        "excluded_stage2_arms": EXCLUDED,
        "manifest": str(manifest),
        "manifest_sha256": sha(manifest),
        "decode": {"backend": "vllm", "seed": 42, "temperature": 0.7, "top_p": 0.9, "max_new_tokens": 512, "bfloat16": True, "enable_thinking": False},
        "scoring": {"helpfulness": "PKU-Alignment/beaver-7b-v1.0-reward@375cd6a9f0d7e339d2199b05ba129a4a8906596d", "harmlessness": "-PKU-Alignment/beaver-7b-v1.0-cost@c1bd343d2ddc2cb810bd736563c7ad0bf38f6b28"},
        "stability_gate": {"detector": "corrected_nonempty_paired_span_v1", "records": 1000, "empty": 0, "think_leakage": 0, "length_ratio": [0.33, 2.0], "max_repeat_run": 20, "fail_closed": True},
        "primary_metric": "mean_prompt_worst_norm_score; per-prompt min-max over exactly the gate-passing locked pool",
        "bootstrap": {"resamples": 2000, "seed": 42, "unit": "prompt", "paired": True},
        "spent_sealed_split_touched": False,
    }
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    target.with_suffix(".sha256").write_text(f"{sha(target)}  {target.name}\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
