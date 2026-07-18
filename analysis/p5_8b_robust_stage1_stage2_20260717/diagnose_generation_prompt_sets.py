#!/usr/bin/env python3
"""Compare response-pool prompt identities without emitting prompt text."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def prompt_hashes(path: Path) -> tuple[int, set[str], set[str]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    ids = {str(row.get("prompt_id", row["prompt"])) for row in rows}
    prompts = {hashlib.sha256(row["prompt"].encode("utf-8")).hexdigest() for row in rows}
    return len(rows), ids, prompts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--models", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    names = [name for name in args.models.split(",") if name]
    base_rows, base_ids, base_prompts = prompt_hashes(args.generation_root / "base" / f"output_{args.seed}.json")
    payload = {"base": {"rows": base_rows, "ids": len(base_ids), "prompt_hashes": len(base_prompts)}}
    for name in names:
        rows, ids, prompts = prompt_hashes(args.generation_root / name / f"output_{args.seed}.json")
        payload[name] = {
            "rows": rows,
            "ids": len(ids),
            "prompt_hashes": len(prompts),
            "id_overlap_with_base": len(ids & base_ids),
            "prompt_overlap_with_base": len(prompts & base_prompts),
            "prompt_set_sha256": hashlib.sha256("\n".join(sorted(prompts)).encode("utf-8")).hexdigest(),
        }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
