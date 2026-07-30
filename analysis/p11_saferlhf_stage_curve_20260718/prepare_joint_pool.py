#!/usr/bin/env python3
"""Merge every locked gate-passing stage point and shard the joint score pool."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with (args.output / "models.tsv").open(encoding="utf-8") as handle:
        models = [row["key"] for row in csv.DictReader(handle, delimiter="\t")]
    merge = args.project / "analysis/p2_8b_hh_multiobjective_20260717/merge_eval_pool.py"
    shard = args.project / "analysis/p2_8b_hh_multiobjective_20260717/shard_score_input.py"
    subprocess.run([
        sys.executable, str(merge), "--generation-root", str(args.output / "generations"),
        "--models", ",".join(models), "--seed", "42", "--expected-records", "1000",
        "--gate-root", str(args.output / "gates"), "--output", str(args.output / "response_pool.jsonl"),
        "--audit", str(args.output / "pool_audit.json"),
    ], check=True)
    subprocess.run([
        sys.executable, str(shard), "split", "--input", str(args.output / "response_pool.jsonl"),
        "--output-dir", str(args.output / "shards"), "--num-shards", "6", "--expected-records", "1000",
    ], check=True)
    (args.output / "SCORE_READY").write_text("ready\n", encoding="utf-8")


if __name__ == "__main__":
    main()
