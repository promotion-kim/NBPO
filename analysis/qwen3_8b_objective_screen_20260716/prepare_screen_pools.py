#!/usr/bin/env python3
"""Assemble the locked general and conflict response pools."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


TRAINED = [
    "ronpo_full_expect", "ronpo_k_only", "ipo", "simpo", "sppo_avg", "inpo_avg",
    "ht_mnpo_helpfulness", "ht_mnpo_safety", "ht_mnpo_conciseness",
]
ALL = ["base", "weak_small", "verbose", "terse", "less_aligned"] + TRAINED


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--python", required=True)
    args = parser.parse_args()
    existing = args.repo / "results/p1_8b_ronpo_os_only_20260716/fixed647_exact_table4/generations"
    script = args.repo / "analysis/qwen3_8b_objective_screen_20260716/assemble_screen_pool.py"
    outputs = {}
    for split in ["general", "conflict_curated"]:
        command = [args.python, str(script), "--manifest", str(args.root / "prompt_manifests" / f"{split}.jsonl")]
        for name in ALL:
            if split == "general" and name in ["base"] + TRAINED:
                source = existing / name / "output_42.json"
            else:
                source = args.root / "generations" / split / name / "output_42.json"
            command.extend(["--policy", f"{name}={source}"])
        output = args.root / "pools" / f"{split}.jsonl"
        command.extend(["--output", str(output)])
        subprocess.run(command, cwd=args.repo, check=True)
        outputs[split] = {"path": str(output), "sha256": sha(output), "prompts": 647 if split == "general" else 128}

    combined = args.root / "pools/combined.jsonl"
    with combined.open("wb") as handle:
        for split in ["general", "conflict_curated"]:
            handle.write(Path(outputs[split]["path"]).read_bytes())
    payload = {"policy_order": ALL, "pools": outputs, "combined": str(combined), "combined_sha256": sha(combined)}
    (args.root / "pool_manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
