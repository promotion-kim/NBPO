#!/usr/bin/env python3
"""Regenerate the deterministic nbpo_toy fixture data (committed; run from repo root).

Six training prompts, two monitoring prompts, two final-eval prompts (pairwise
disjoint, as run_nbpo_stage enforces); four policy seeds and four reference
seeds per pool. All text is synthetic -- the mock judge scores it by hashed
response-id strengths, so content only needs to be distinct.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE / "assets"

TRAIN = [f"p{i:02d}" for i in range(6)]
MONITORING = ["m00", "m01"]
FINAL = ["f00", "f01"]


def rows(prompt_ids, tag, seed):
    return [
        {
            "prompt_id": pid,
            "prompt": f"Toy prompt {pid}: describe the object in two sentences.",
            "generated_text": f"[{tag}:{seed}] deterministic toy response for {pid}.",
            "seed": seed,
        }
        for pid in prompt_ids
    ]


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    for s in range(4):
        (DATA / f"train_policy_s{s}.json").write_text(
            json.dumps(rows(TRAIN, "policy", f"s{s}"), indent=1) + "\n")
        (DATA / f"train_ref_r{s}.json").write_text(
            json.dumps(rows(TRAIN, "ref", f"r{s}"), indent=1) + "\n")
        (DATA / f"mon_candidate_s{s}.json").write_text(
            json.dumps(rows(MONITORING, "candidate", f"s{s}"), indent=1) + "\n")
        (DATA / f"mon_ref_r{s}.json").write_text(
            json.dumps(rows(MONITORING, "ref", f"r{s}"), indent=1) + "\n")
    (DATA / "monitoring_prompts.json").write_text(json.dumps(MONITORING) + "\n")
    (DATA / "final_eval_prompts.json").write_text(json.dumps(FINAL) + "\n")
    print(f"fixture data regenerated under {DATA}")


if __name__ == "__main__":
    main()
