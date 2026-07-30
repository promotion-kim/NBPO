#!/usr/bin/env python3
"""Merge P10 Stage-2 scorer shards and recompute the frozen metrics."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    args = parser.parse_args()
    output = args.experiment / "stage2_eval_p8_locked_panel"
    audit = json.loads((output / "pool_audit.json").read_text(encoding="utf-8"))
    count = len(audit["eligible_models"])
    display_names = {
        "base": "Base",
        "ronpo_os": "RONPO (OS, Stage-2)",
        "inpo_avg": "INPO (avg, Stage-2)",
        "sppo_avg": "SPPO (avg, Stage-2)",
        "ipo": "IPO (Stage-2)",
        "dpo": "DPO (Stage-2)",
    }
    display_map = output / "stage2_display_name_map.json"
    display_map.write_text(json.dumps(display_names, indent=2) + "\n", encoding="utf-8")
    merge = args.project / "analysis/p2_8b_hh_multiobjective_20260717/shard_score_input.py"
    for objective in ("helpfulness", "harmlessness"):
        inputs = [output / "score_shards" / f"{objective}_{index}.jsonl" for index in range(6)]
        subprocess.run([sys.executable, str(merge), "merge", "--inputs", *map(str, inputs), "--output", str(output / "scores" / f"{objective}.jsonl"), "--audit", str(output / "scores" / f"{objective}_audit.json"), "--expected-records", "1000", "--expected-scores-per-row", str(count), "--strip-responses"], check=True)
    aggregate = args.project / "analysis/p5_8b_robust_stage1_stage2_20260717/aggregate_stage1_comparison.py"
    subprocess.run([sys.executable, str(aggregate), "--helpfulness", str(output / "scores/helpfulness.jsonl"), "--harmlessness", str(output / "scores/harmlessness.jsonl"), "--pool-audit", str(output / "pool_audit.json"), "--output-dir", str(output / "results"), "--bootstrap", "2000", "--seed", "42", "--scope", "P10 optimizer-seed-43 Stage-2 comparison on the already-open P8 1,000-prompt panel; not a fresh confirmation and not used for model selection", "--report-title", "P10 seed-43 Stage-2 SafeRLHF comparison", "--ronpo-arm", "ronpo_os", "--comparison-label", "best eligible non-RONPO trained Stage-2 arm", "--display-name-map", str(display_map)], check=True)
    summary = json.loads((output / "results/ranked_validation_summary.json").read_text(encoding="utf-8"))
    (output / "evaluation_status.json").write_text(json.dumps({"status": "completed", "eligible_models": audit["eligible_models"], "failed_models": audit["failed_models"], "summary": str(output / "results/ranked_validation_summary.json"), "spent_sealed_split_touched": False}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
