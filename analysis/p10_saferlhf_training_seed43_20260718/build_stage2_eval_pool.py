#!/usr/bin/env python3
"""Create the score input only from P10 Stage-2 models that pass the frozen gate."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> list[dict]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or len(value) != 1000:
        raise RuntimeError(f"expected 1,000 generation rows: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    args = parser.parse_args()
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    output = args.experiment / "stage2_eval_p8_locked_panel"
    if (output / "pool_audit.json").exists():
        raise RuntimeError("refusing to overwrite frozen evaluation pool")
    generations = {"base": Path(lock["models"]["base"])}
    eligible = ["base"]
    failed: dict[str, str] = {}
    for arm in lock["models"]:
        if arm == "base":
            continue
        gate = output / "gates" / f"{arm}.json"
        gen = output / "generations" / arm / "output_43.json"
        if not gate.is_file() or not gen.is_file():
            failed[arm] = "missing gate or generation"
            continue
        result = json.loads(gate.read_text(encoding="utf-8"))
        if result.get("passed") is not True:
            failed[arm] = "failed corrected stability gate"
            continue
        eligible.append(arm)
        generations[arm] = gen
    if "ronpo_os" not in eligible:
        raise RuntimeError("RONPO-OS failed the frozen gate; fail closed before reward scoring")
    ordered = []
    for model in eligible:
        rows = read(generations[model])
        row_map = {str(row["prompt_id"]): row for row in rows}
        if len(row_map) != 1000:
            raise RuntimeError(f"duplicate prompt ids: {model}")
        ordered.append((model, row_map))
    prompt_ids = sorted(ordered[0][1])
    if any(set(rows) != set(prompt_ids) for _, rows in ordered):
        raise RuntimeError("generation prompt sets differ")
    response_pool = output / "response_pool.jsonl"
    response_pool.parent.mkdir(parents=True, exist_ok=True)
    with response_pool.open("w", encoding="utf-8") as handle:
        for prompt_id in prompt_ids:
            first = ordered[0][1][prompt_id]
            responses = [str(rows[prompt_id].get("generated_text", "")) for _, rows in ordered]
            if any(not response.strip() for response in responses):
                raise RuntimeError(f"empty generation at prompt {prompt_id}")
            handle.write(json.dumps({"prompt_id": prompt_id, "prompt": str(first["prompt"]), "response_model_names": eligible, "all_generated_responses": responses}, ensure_ascii=False) + "\n")
    audit = {"status": "completed", "records": len(prompt_ids), "eligible_models": eligible, "failed_models": failed, "generations": {model: {"path": str(path), "sha256": sha(path)} for model, path in generations.items()}, "response_pool": {"path": str(response_pool), "sha256": sha(response_pool)}}
    (output / "pool_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    shard = args.project / "analysis/p2_8b_hh_multiobjective_20260717/shard_score_input.py"
    subprocess.run([sys.executable, str(shard), "split", "--input", str(response_pool), "--output-dir", str(output / "shards"), "--num-shards", "6", "--expected-records", "1000"], check=True)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
