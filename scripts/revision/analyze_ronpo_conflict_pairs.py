#!/usr/bin/env python3
"""Summarize the adversarial distribution in a RONPO JSONL pair file."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "q05": float(np.quantile(array, 0.05)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.50)),
        "q75": float(np.quantile(array, 0.75)),
        "q95": float(np.quantile(array, 0.95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    targets: list[float] = []
    entropies: list[float] = []
    effective_atoms: list[float] = []
    max_masses: list[float] = []
    prompt_pair_counts: Counter[str] = Counter()
    seen_sigma: set[str] = set()
    objective_mass = defaultdict(float)
    top_objective = Counter()
    sigma_prompts = 0

    with Path(args.input).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            prompt_key = str(row.get("prompt_id") or row["prompt"])
            prompt_pair_counts[prompt_key] += 1
            target = float(row["ronpo_target"])
            targets.append(target)

            if prompt_key in seen_sigma:
                continue
            seen_sigma.add(prompt_key)
            sigma = row.get("ronpo_sigma") or {}
            if not sigma:
                continue
            marginals = {
                objective: float(sum(float(value) for value in masses))
                for objective, masses in sigma.items()
            }
            total = sum(marginals.values())
            if total <= 0:
                continue
            marginals = {name: value / total for name, value in marginals.items()}
            for name, value in marginals.items():
                objective_mass[name] += value
            top_objective[max(marginals, key=lambda name: (marginals[name], name))] += 1
            entropies.append(float(row["ronpo_sigma_entropy"]))
            effective_atoms.append(float(row["ronpo_sigma_effective_atoms"]))
            max_masses.append(float(row["ronpo_sigma_max_mass"]))
            sigma_prompts += 1

    if not targets or not prompt_pair_counts or sigma_prompts == 0:
        raise RuntimeError("input did not contain complete RONPO target/sigma records")

    target_array = np.asarray(targets, dtype=np.float64)
    pair_counts = list(prompt_pair_counts.values())
    result = {
        "input": str(Path(args.input).resolve()),
        "rows": len(targets),
        "prompts": len(prompt_pair_counts),
        "pairs_per_prompt": quantiles([float(value) for value in pair_counts]),
        "target": {
            **quantiles(targets),
            "mean_abs": float(np.abs(target_array).mean()),
            "positive_fraction": float((target_array > 0).mean()),
            "negative_fraction": float((target_array < 0).mean()),
            "zero_fraction": float((target_array == 0).mean()),
        },
        "adversary": {
            "sigma_prompts": sigma_prompts,
            "entropy": quantiles(entropies),
            "effective_atoms": quantiles(effective_atoms),
            "max_atom_mass": quantiles(max_masses),
            "mean_objective_marginal": {
                name: objective_mass[name] / sigma_prompts for name in sorted(objective_mass)
            },
            "top_objective_fraction": {
                name: top_objective[name] / sigma_prompts for name in sorted(objective_mass)
            },
        },
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    marginal = result["adversary"]["mean_objective_marginal"]
    top = result["adversary"]["top_objective_fraction"]
    lines = [
        "# RONPO Conflict-Pair Diagnostics",
        "",
        f"- Rows / prompts: {result['rows']:,} / {result['prompts']:,}",
        f"- Mean pairs per prompt: {result['pairs_per_prompt']['mean']:.3f}",
        f"- Mean absolute regression target: {result['target']['mean_abs']:.4f}",
        f"- Median adversary entropy / effective atoms: "
        f"{result['adversary']['entropy']['median']:.4f} / "
        f"{result['adversary']['effective_atoms']['median']:.3f}",
        f"- Median / 95th-percentile maximum atom mass: "
        f"{result['adversary']['max_atom_mass']['median']:.4f} / "
        f"{result['adversary']['max_atom_mass']['q95']:.4f}",
        "",
        "| Objective | Mean adversary marginal | Fraction ranked most exposed |",
        "|---|---:|---:|",
    ]
    for name in sorted(marginal):
        lines.append(f"| {name} | {marginal[name]:.4f} | {top[name]:.4f} |")
    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
