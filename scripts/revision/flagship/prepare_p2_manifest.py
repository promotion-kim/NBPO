#!/usr/bin/env python3
"""Freeze the P2 model set without consulting P1 reward outcomes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path


METHODS = (
    "base",
    "ronpo_full_expect",
    "dpo",
    "ipo",
    "simpo",
    "kto",
    "sppo_avg",
    "inpo_avg",
    "ht_mnpo_helpfulness",
    "ht_mnpo_safety",
    "ht_mnpo_conciseness",
)
SEED = 42
TASK_DIRS = (
    "ifeval", "mmlu", "mmlu_pro", "gpqa", "arc", "hellaswag", "truthfulqa",
    "winogrande", "gsm8k", "minerva_math", "benchmarks", "aime", "humaneval",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p1-ledger", required=True)
    parser.add_argument("--lm-eval-task-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--provenance", required=True)
    args = parser.parse_args()

    ledger = json.loads(Path(args.p1_ledger).read_text())
    by_key = {(row["method"], row.get("seed")): row for row in ledger["models"]}
    selected = []
    missing = []
    for method in METHODS:
        key = ("base", None) if method == "base" else (method, SEED)
        row = by_key.get(key)
        if row is None:
            missing.append({"method": method, "seed": key[1]})
        else:
            selected.append({
                "name": "base" if method == "base" else f"{method}__s{SEED}",
                "model": row["model"],
                "method": method,
                "seed": key[1],
                "wandb_run_id": row.get("wandb_run_id"),
                "stability_gate": row.get("stability_gate"),
            })
    if not selected or selected[0]["method"] != "base":
        raise SystemExit("P2 cannot run without the frozen base model")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("name", "model", "method", "seed"), delimiter="\t")
        writer.writeheader()
        writer.writerows([{key: row[key] for key in writer.fieldnames} for row in selected])

    task_root = Path(args.lm_eval_task_root)
    files = []
    for directory in TASK_DIRS:
        root = task_root / directory
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix in {".yaml", ".yml", ".py"}:
                files.append({"path": str(path.relative_to(task_root)), "sha256": sha256(path)})
    provenance = {
        "selection_policy": "seed42 primary RONPO full-expect plus every non-ablation baseline plus base; frozen before P1 results",
        "models": selected,
        "missing_stability_eligible_models": missing,
        "complete_preregistered_set": not missing,
        "lm_eval_version": importlib.metadata.version("lm_eval"),
        "task_source_hashes": files,
    }
    Path(args.provenance).write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"models": len(selected), "missing": missing, "task_source_files": len(files)}, indent=2))


if __name__ == "__main__":
    main()
