#!/usr/bin/env python3
"""Build one frozen response matrix for common-context RM scoring."""

import argparse
import hashlib
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--generations", required=True)
    p.add_argument("--pool", required=True, help="Any frozen scored test_*.jsonl; response text is shared")
    p.add_argument("--lock", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    lock = json.loads(Path(args.lock).read_text())
    names = lock["normalization_policies"]
    by_policy = {}
    for name in names:
        path = Path(args.generations) / name / "output_42.json"
        rows = json.loads(path.read_text())
        if len(rows) != 647:
            raise ValueError(f"{name}: expected 647 rows, found {len(rows)}")
        by_policy[name] = {row["prompt"]: row["generated_text"] for row in rows}
        if len(by_policy[name]) != 647:
            raise ValueError(f"{name}: duplicate prompt")

    prompts = sorted(by_policy[names[0]])
    for name in names[1:]:
        if sorted(by_policy[name]) != prompts:
            raise ValueError(f"{name}: prompt set differs")

    pool = {row["prompt"]: row["all_generated_responses"] for row in read_jsonl(Path(args.pool))}
    if len(pool) != 646:
        raise ValueError(f"expected frozen pool for 646 prompts, found {len(pool)}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as out:
        for prompt in prompts:
            response_names = list(names)
            responses = [by_policy[name][prompt] for name in names]
            stored = pool.get(prompt, [])
            response_names.extend(f"stage2_pool_{idx}" for idx in range(len(stored)))
            responses.extend(stored)
            row = {
                "prompt_id": hashlib.sha256(prompt.encode()).hexdigest(),
                "prompt": prompt,
                "response_model_names": response_names,
                "all_generated_responses": responses,
                "stored_pool_present": bool(stored),
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"rows": len(prompts), "policies": len(names), "pool_rows": len(pool), "output": str(output)}))


if __name__ == "__main__":
    main()
