#!/usr/bin/env python3
"""Resolve exactly one stability-passing attempt for each method/seed."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METHODS = (
    "ronpo_full_expect", "ronpo_k_only", "dpo", "ipo", "simpo", "kto", "sppo_avg", "inpo_avg",
    "ht_mnpo_helpfulness", "ht_mnpo_safety", "ht_mnpo_conciseness",
)
SEEDS = (42, 43, 44)


def has_weights(path: Path) -> bool:
    return (path / "model.safetensors").is_file() or (path / "model.safetensors.index.json").is_file()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ledger", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    rows = [{"name": "base", "model": args.base_model, "kind": "model", "method": "base", "seed": None}]
    missing = []
    for method in METHODS:
        for seed in SEEDS:
            candidates = []
            for status_path in sorted((root / "full" / method / f"seed{seed}").glob("attempt*/job_status.json")):
                status = json.loads(status_path.read_text())
                model = status_path.parent
                gate = model / "stability/gate.json"
                if status.get("status") == "completed" and has_weights(model) and gate.exists():
                    gate_data = json.loads(gate.read_text())
                    if gate_data.get("passed") is True:
                        candidates.append((int(model.name.removeprefix("attempt")), model, status, gate_data))
            if len(candidates) != 1:
                missing.append({"method": method, "seed": seed, "eligible_attempts": len(candidates)})
                continue
            attempt, model, status, gate_data = candidates[0]
            rows.append({
                "name": f"{method}__s{seed}", "model": str(model), "kind": "model",
                "method": method, "seed": seed, "attempt": attempt,
                "wandb_run_id": status["wandb_run_id"], "wandb_url": status["wandb_url"],
                "stability_gate": str(model / "stability/gate.json"),
            })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("name", "model", "kind"), delimiter="\t", extrasaction="ignore")
        writer.writerows(rows)
    ledger = {"models": rows, "missing": missing, "complete": not missing, "required_methods": list(METHODS), "seeds": list(SEEDS)}
    Path(args.ledger).write_text(json.dumps(ledger, indent=2) + "\n")
    print(json.dumps(ledger, indent=2))


if __name__ == "__main__":
    main()
