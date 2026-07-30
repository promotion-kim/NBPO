#!/usr/bin/env python3
"""Write the immutable Stage-4 ARC-Challenge cohort lock before evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from run_p8_stage4_arc_challenge import MODELS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / "capability_lock.json"
    if target.exists():
        raise RuntimeError(f"refusing to overwrite existing lock: {target}")
    requested = list(MODELS)
    target.write_text(json.dumps({
        "status": "locked_before_evaluation",
        "scope": "Stage-4 appendix cohort capability measurement; not a selection criterion.",
        "task": "arc_challenge",
        "lm_eval_version": "0.4.12",
        "num_fewshot": 25,
        "apply_chat_template": True,
        "enable_thinking": False,
        "seed": 42,
        "requested_models": requested,
        "model_paths": MODELS,
        "spent_sealed_split_touched": False,
    }, indent=2) + "\n", encoding="utf-8")
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    (args.output / "capability_lock.sha256").write_text(digest + "  capability_lock.json\n", encoding="utf-8")
    print(digest)


if __name__ == "__main__":
    main()
