#!/usr/bin/env python3
"""Merge Beaver score shards and aggregate a Stage-4 seed panel."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


DISPLAY = {
    "base": "Base",
    "ronpo_os_stage4": "RONPO (OS)",
    "ronpo_topmass_stage4": "RONPO (top-mass)",
    "inpo_avg_stage4": "INPO-avg",
    "sppo_avg_stage4": "SPPO-avg",
    "simpo_stage4": "SimPO",
    "ipo_stage4": "IPO",
    "dpo_stage4": "DPO",
    "ht_mnpo_harmless_stage4": "HT-MNPO (harml.)",
    "ht_mnpo_helpfulness_stage4": "HT-MNPO (help.)",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-seed", type=int, default=43)
    parser.add_argument("--num-shards", type=int, default=6)
    parser.add_argument("--expected-records", type=int, default=1000)
    args = parser.parse_args()

    audit = json.loads((args.output / "pool_audit.json").read_text(encoding="utf-8"))
    model_count = len(audit["eligible_models"])
    if model_count != len(DISPLAY):
        raise RuntimeError(f"expected {len(DISPLAY)} eligible models, found {model_count}")
    merge = args.project / "analysis/p2_8b_hh_multiobjective_20260717/shard_score_input.py"
    for objective in ("helpfulness", "harmlessness"):
        inputs = [args.output / f"score_shards/{objective}_{index}.jsonl" for index in range(args.num_shards)]
        missing = [str(path) for path in inputs if not path.is_file()]
        if missing:
            raise RuntimeError(f"missing score shards for {objective}: {missing}")
        subprocess.run(
            [
                sys.executable, str(merge), "merge", "--inputs", *map(str, inputs),
                "--output", str(args.output / f"scores/{objective}.jsonl"),
                "--audit", str(args.output / f"scores/{objective}_audit.json"),
                "--expected-records", str(args.expected_records),
                "--expected-scores-per-row", str(model_count),
                "--strip-responses",
            ],
            check=True,
        )
    display_path = args.output / "display_names.json"
    display_path.write_text(json.dumps(DISPLAY, indent=2) + "\n", encoding="utf-8")
    aggregate = args.project / "analysis/p5_8b_robust_stage1_stage2_20260717/aggregate_stage1_comparison.py"
    subprocess.run(
        [
            sys.executable, str(aggregate),
            "--helpfulness", str(args.output / "scores/helpfulness.jsonl"),
            "--harmlessness", str(args.output / "scores/harmlessness.jsonl"),
            "--pool-audit", str(args.output / "pool_audit.json"),
            "--output-dir", str(args.output / "results"),
            "--bootstrap", "2000", "--seed", "42",
            "--scope", f"fresh 1000-prompt PKU-SafeRLHF default-test panel; Stage-4 training seed {args.training_seed}; decode seed fixed at 42",
            "--report-title", f"Stage-4 seed-{args.training_seed} SafeRLHF comparison",
            "--ronpo-arm", "ronpo_os_stage4",
            "--ronpo-arm", "ronpo_topmass_stage4",
            "--comparison-label", "best eligible non-RONPO trained Stage-4 arm",
            "--display-name-map", str(display_path),
        ],
        check=True,
    )
    print(json.dumps({"status": "complete", "results": str(args.output / "results")}, indent=2))


if __name__ == "__main__":
    main()
