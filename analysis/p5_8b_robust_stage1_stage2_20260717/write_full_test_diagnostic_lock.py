#!/usr/bin/env python3
"""Freeze the retrospective full-test diagnostic protocol before decode."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.data_manifest.read_text(encoding="utf-8"))
    if data["status"] != "retrospective_diagnostic_manifest_locked_before_decode":
        raise RuntimeError("unexpected manifest status")
    models = [
        "base", "ronpo_os_stage2", "ronpo_topmass_stage2", "inpo_avg_stage2", "simpo_stage2",
        "ipo_stage2", "dpo_stage2", "sppo_avg_stage2", "ht_mnpo_harmless_stage2", "ht_mnpo_helpfulness_stage2",
    ]
    payload = {
        "status": "locked_before_decode",
        "scope": "retrospective diagnostic only; raw test overlaps prior held-out panels and is not a fresh evaluation",
        "manifest": data["manifest"],
        "manifest_sha256": data["manifest_sha256"],
        "unique_prompts": data["unique_conflict_prompts"],
        "prior_overlap": data["unique_conflicts_overlapping_prior_heldout"],
        "models": models,
        "decode": {
            "vllm": True, "seed": 42, "temperature": 0.7, "top_p": 0.9,
            "max_new_tokens": 512, "max_model_len": 8192, "dtype": "bfloat16",
            "enable_thinking": False,
        },
        "scorers": {
            "helpfulness": "PKU-Alignment/beaver-7b-v1.0-reward@375cd6a9f0d7e339d2199b05ba129a4a8906596d",
            "harmlessness": "negative PKU-Alignment/beaver-7b-v1.0-cost@c1bd343d2ddc2cb810bd736563c7ad0bf38f6b28",
        },
        "metrics": "raw per-objective rewards; per-prompt min-max normalized Avg/Worst over this diagnostic model pool; paired 2000-resample prompt bootstrap, seed 42",
        "stability_policy": "Gate every generation and retain failures only as diagnostic rows. No failed row is paper eligible.",
        "decision_policy": "No checkpoint selection, tuning, paper edit, upload, or model claim may use this diagnostic.",
        "spent_sealed_split_touched": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock = args.output_dir / "DIAGNOSTIC_LOCK.json"
    lock.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "DIAGNOSTIC_LOCK.sha256").write_text(sha(lock) + "  DIAGNOSTIC_LOCK.json\n", encoding="utf-8")
    print(json.dumps({"lock": str(lock), "sha256": sha(lock), "models": models}, indent=2))


if __name__ == "__main__":
    main()
