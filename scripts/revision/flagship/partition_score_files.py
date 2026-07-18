#!/usr/bin/env python3
"""Partition completed S0 score files into frozen train/validation membership."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def ids(path: Path) -> set[str]:
    with path.open(encoding="utf-8") as handle:
        return {str(json.loads(line)["prompt_id"]) for line in handle if line.strip()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--scored-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--objectives", nargs="+", default=["helpfulness", "safety", "conciseness"])
    args = parser.parse_args()
    data, scored, output = map(Path, (args.data_dir, args.scored_dir, args.output_dir))
    membership = {name: ids(data / f"pool_{name}.jsonl") for name in ("train", "validation")}
    if membership["train"] & membership["validation"]:
        raise RuntimeError("train/validation overlap")
    summary = {}
    for objective in args.objectives:
        handles = {}
        try:
            for split in membership:
                path = output / split / f"{objective}.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                handles[split] = path.open("w", encoding="utf-8")
            counts = {split: 0 for split in membership}
            with (scored / f"{objective}.jsonl").open(encoding="utf-8") as source:
                for line in source:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    prompt_id = str(row["prompt_id"])
                    destinations = [split for split, values in membership.items() if prompt_id in values]
                    if len(destinations) != 1:
                        raise RuntimeError(f"{objective}: unexpected membership for {prompt_id}: {destinations}")
                    split = destinations[0]
                    handles[split].write(json.dumps(row, ensure_ascii=False) + "\n")
                    counts[split] += 1
        finally:
            for handle in handles.values():
                handle.close()
        expected = {split: len(values) for split, values in membership.items()}
        if counts != expected:
            raise RuntimeError(f"{objective}: counts={counts}, expected={expected}")
        summary[objective] = counts
    (output / "partition_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
