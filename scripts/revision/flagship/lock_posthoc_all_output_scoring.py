#!/usr/bin/env python3
"""Lock the user-requested post-hoc scoring of all existing raw outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

from run_seed42_sealed_reward_eval import METHODS


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    work = args.eval_root / "confirmatory"
    gates = json.loads((work / "stability_gates/summary.json").read_text())
    protocol = json.loads((args.eval_root / "corrected_protocol_lock.json").read_text())
    failure = json.loads((work / "results/failure_diagnostics.json").read_text())
    if failure.get("reward_scoring_started") is not False:
        raise RuntimeError("reward scoring had already started before the post-hoc lock")
    for ronpo in ("ronpo_full_expect", "ronpo_k_only"):
        if gates["models"][ronpo].get("passed") is not True:
            raise RuntimeError(f"strict RONPO stability requirement failed: {ronpo}")

    generations = {}
    for method in METHODS:
        path = work / "generations" / method / "output_42.json"
        rows = json.loads(path.read_text())
        if not isinstance(rows, list) or len(rows) != 1736:
            raise RuntimeError(f"incomplete existing raw generation: {method}")
        generations[method] = {
            "path": str(path), "sha256": sha256_file(path), "records": len(rows),
            "stability_status": gates["models"][method]["status"],
            "max_repeat_run": gates["models"][method]["candidate"]["max_repeat_run"],
        }
    payload = {
        "schema_version": 1,
        "status": "locked_posthoc_scoring_only",
        "locked_at_kst": datetime.now().astimezone().isoformat(timespec="seconds"),
        "reason": "User instructed that RONPO remains strict fail-closed while baseline raw outputs may still be reward-scored with their stability failures disclosed.",
        "scope": "score all 11 already-generated 1736-prompt raw-output files; no generation, cleaning, retry, model selection, or hyperparameter change",
        "ronpo_strict_gate_passed": True,
        "baseline_failures_retained_and_disclosed": True,
        "paper_eligible_as_preregistered_flagship": False,
        "analysis_label": "posthoc_all_output_sensitivity",
        "fairness_note": "The reward scoring pipeline is identical for all raw outputs, but stability eligibility is asymmetric and post-hoc; do not describe this as the preregistered fair flagship comparison.",
        "prompt_count": 1736,
        "prompt_file_sha256": protocol["confirmatory_file_sha256"],
        "decode": protocol["decode"],
        "reward_scoring": {
            "model": "RLHFlow/ArmoRM-Llama3-8B-v0.1",
            "objectives": ["helpfulness", "safety", "conciseness"],
            "normalization": "per-prompt min-max across all 11 raw-output model responses",
            "bootstrap_resamples": 2000,
            "bootstrap_seed": 42,
        },
        "generations": generations,
        "reward_data_consulted_before_lock": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        previous = json.loads(args.output.read_text())
        current = dict(payload)
        current.pop("locked_at_kst")
        previous.pop("locked_at_kst", None)
        if previous != current:
            raise RuntimeError("a different post-hoc scoring amendment is already locked")
        return
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
