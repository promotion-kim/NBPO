#!/usr/bin/env python3
"""Build precomputed RONPO support-K variants while reusing frozen log probabilities."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from datasets import Dataset, DatasetDict, load_from_disk

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from mnpo_scripts.build_multi_objective_dataset import build_pairs_for_record


VARIANTS = {
    "ronpo_full_expect_k6": ("ronpo_full_expect", "sigma", 6),
    "ronpo_k_only_k1": ("ronpo_k_only", "sigma_k_only", 1),
    "ronpo_k_only_k2": ("ronpo_k_only", "sigma_k_only", 2),
}


def prompt_record(row: dict) -> dict:
    return {
        "prompt_id": row["prompt_id"],
        "prompt": row["prompt"],
        "all_generated_responses": row["all_generated_responses"],
        "objective_names": row["objective_names"],
        "objective_scores": row["objective_scores"],
    }


def update_row(old: dict, new: dict) -> dict:
    result = dict(old)
    for key, value in new.items():
        if key.startswith("ronpo_") or key in {
            "normalized_objective_scores", "avg_objective_scores", "min_objective_scores",
            "avg_oracle_win_probs", "pair_source", "chosen_index", "rejected_index",
        }:
            result[key] = value
    return result


def build_split(source, strategy: str, support_k: int) -> Dataset:
    rows = [source[index] for index in range(len(source))]
    output = []
    start = 0
    while start < len(rows):
        prompt_id = rows[start]["prompt_id"]
        end = start + 1
        while end < len(rows) and rows[end]["prompt_id"] == prompt_id:
            end += 1
        old_group = rows[start:end]
        _avg, new_group = build_pairs_for_record(
            prompt_record(old_group[0]), normalization="minmax", ronpo_pair_strategy=strategy,
            tie_threshold=0.0, adversary_steps=25, adversary_alpha=1.0,
            adversary_kappa=0.05, preference_scale=8.0, policy_mode="uniform",
            policy_temperature=0.2, pairs_per_prompt=3, adversary_selection="all",
            ronpo_policy_pair_mode="expected_relative_policy_vs_policy",
            ronpo_policy_samples_per_atom=1, k_only_fixed_atom="avg_worst",
            k_only_response_mode="uniform", common_pair_seed="flagship-common-pairs-v1",
            expected_support_k=support_k,
        )
        if len(old_group) != len(new_group):
            raise RuntimeError(f"pair count changed for {prompt_id}: {len(old_group)} != {len(new_group)}")
        for old, new in zip(old_group, new_group):
            old_pair = (int(old["chosen_index"]), int(old["rejected_index"]))
            new_pair = (int(new["chosen_index"]), int(new["rejected_index"]))
            if old_pair != new_pair:
                raise RuntimeError(f"common pair changed for {prompt_id}: {old_pair} != {new_pair}")
            output.append(update_row(old, new))
        start = end
    return Dataset.from_list(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--precomputed-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = {"status": "building", "variants": {}, "spent_sealed_split_touched": False}
    args.output_root.mkdir(parents=True, exist_ok=True)
    for name, (source_name, strategy, support_k) in VARIANTS.items():
        destination = args.output_root / name
        if (destination / "dataset_dict.json").is_file():
            dataset = load_from_disk(str(destination))
            manifest["variants"][name] = {"status": "reused", "counts": {k: len(v) for k, v in dataset.items()}}
            continue
        source = load_from_disk(str(args.precomputed_root / source_name))
        built = DatasetDict({split: build_split(source[split], strategy, support_k) for split in source})
        built.save_to_disk(str(destination))
        manifest["variants"][name] = {
            "status": "completed", "source": source_name, "strategy": strategy,
            "expected_support_k": support_k, "counts": {k: len(v) for k, v in built.items()},
        }
    manifest["status"] = "completed"
    manifest_path = args.output_root / "variant_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    print(json.dumps({**manifest, "manifest_sha256": digest}, indent=2))


if __name__ == "__main__":
    main()
