#!/usr/bin/env python3
"""Collect verified RMOD K summaries into the Figure-3 JSONL input."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--k", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--expected-n", type=int, default=1000)
    parser.add_argument("--expected-block-size", type=int, default=16)
    args = parser.parse_args()
    rows = []
    for k in args.k:
        path = args.summary_dir / f"k{k}_summary.json"
        if not path.exists():
            raise FileNotFoundError(path)
        row = json.loads(path.read_text(encoding="utf-8"))
        if (row["k"] != k or not row.get("chat_template")
                or row.get("n") != args.expected_n
                or row.get("block_size") != args.expected_block_size):
            raise ValueError(f"unverified summary: {path}")
        rows.append(row)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(f"wrote {len(rows)} points to {args.out}")


if __name__ == "__main__":
    main()
