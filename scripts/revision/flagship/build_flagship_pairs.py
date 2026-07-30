#!/usr/bin/env python3
"""Build all matched P1 pair datasets from a frozen three-objective pool."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from mnpo_scripts.build_multi_objective_dataset import (
    build_pairs_for_record,
    load_objective_records,
    merge_prompt_records,
)


OBJECTIVES = ("helpfulness", "safety", "conciseness")


def messages(prompt: str, response: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]


def normalize(values: list[float]) -> list[float]:
    lo, hi = min(values), max(values)
    if abs(hi - lo) < 1e-12:
        return [0.5] * len(values)
    return [(float(value) - lo) / (hi - lo) for value in values]


def ht_pair(record: dict[str, Any], objective: str) -> dict[str, Any] | None:
    scores = normalize([float(value) for value in record["objective_scores"][objective]])
    chosen = max(range(len(scores)), key=scores.__getitem__)
    rejected = min(range(len(scores)), key=scores.__getitem__)
    if chosen == rejected or scores[chosen] - scores[rejected] <= 1e-12:
        return None
    out = dict(record)
    prompt = str(record["prompt"])
    responses = record["all_generated_responses"]
    out["normalized_objective_scores"] = {
        name: normalize([float(value) for value in record["objective_scores"][name]]) for name in OBJECTIVES
    }
    out["prompt"] = [{"role": "user", "content": prompt}]
    out["chosen"] = messages(prompt, str(responses[chosen]))
    out["rejected"] = messages(prompt, str(responses[rejected]))
    out["chosen_index"] = chosen
    out["rejected_index"] = rejected
    out["pair_source"] = "flagship_ht_mnpo_single_armorm_head"
    out["ht_target"] = float(scores[chosen] - scores[rejected])
    out["ht_objective_name"] = objective
    out["ht_score_normalization"] = "per_prompt_minmax"
    return out


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_split(scored_root: Path, output_root: Path, split: str, kappa: float) -> dict[str, int]:
    named = [(name, str(scored_root / split / f"{name}.jsonl")) for name in OBJECTIVES]
    merged = merge_prompt_records(load_objective_records(named), list(OBJECTIVES))
    outputs: dict[str, list[dict[str, Any]]] = {
        "avg": [], "ronpo_full_expect": [], "ronpo_k_only": [],
        **{f"ht_mnpo_{name}": [] for name in OBJECTIVES},
    }
    for record in merged:
        avg, full = build_pairs_for_record(
            record, normalization="minmax", ronpo_pair_strategy="sigma", tie_threshold=0.0,
            adversary_steps=25, adversary_alpha=1.0, adversary_kappa=kappa,
            preference_scale=8.0, policy_mode="uniform", policy_temperature=0.2,
            pairs_per_prompt=3, adversary_selection="all",
            ronpo_policy_pair_mode="expected_relative_policy_vs_policy",
            ronpo_policy_samples_per_atom=1, k_only_fixed_atom="avg_worst",
            k_only_response_mode="uniform", common_pair_seed="flagship-common-pairs-v1",
        )
        _, konly = build_pairs_for_record(
            record, normalization="minmax", ronpo_pair_strategy="sigma_k_only", tie_threshold=0.0,
            adversary_steps=25, adversary_alpha=1.0, adversary_kappa=kappa,
            preference_scale=8.0, policy_mode="uniform", policy_temperature=0.2,
            pairs_per_prompt=3, adversary_selection="all",
            ronpo_policy_pair_mode="expected_relative_policy_vs_policy",
            ronpo_policy_samples_per_atom=1, k_only_fixed_atom="avg_worst",
            k_only_response_mode="uniform", common_pair_seed="flagship-common-pairs-v1",
        )
        if avg is not None:
            outputs["avg"].append(avg)
        outputs["ronpo_full_expect"].extend(full)
        outputs["ronpo_k_only"].extend(konly)
        for name in OBJECTIVES:
            pair = ht_pair(record, name)
            if pair is not None:
                outputs[f"ht_mnpo_{name}"].append(pair)
    for method, rows in outputs.items():
        write_jsonl(output_root / method / f"{split}.jsonl", rows)
    return {method: len(rows) for method, rows in outputs.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--kappa", type=float, default=0.05)
    args = parser.parse_args()
    scored, output = Path(args.scored_root), Path(args.output_root)
    summary = {split: build_split(scored, output, split, args.kappa) for split in ("train", "validation")}
    metadata = {
        "schema_version": 1,
        "objectives": list(OBJECTIVES),
        "normalization": "per_prompt_minmax",
        "adversary": {"steps": 25, "alpha": 1.0, "kappa": args.kappa, "preference_scale": 8.0},
        "ronpo_estimator": "Rao-Blackwellized full sigma expectation with three common response pairs per prompt",
        "counts": summary,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "pair_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
