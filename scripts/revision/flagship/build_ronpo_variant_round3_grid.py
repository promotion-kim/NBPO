#!/usr/bin/env python3
"""Freeze a final trust-region refinement round from Round 2 RM-only selection."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path


TUNABLE = ["learning_rate", "warmup_ratio", "ronpo_alpha", "ronpo_tau", "eta",
           "reference_anchor_weight", "preference_sft_weight", "ronpo_target_column",
           "ronpo_target_schedule_columns", "ronpo_target_schedule_boundaries"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--evaluator-lock", type=Path, required=True)
    parser.add_argument("--round2-grid", type=Path, required=True)
    parser.add_argument("--output-grid", type=Path, required=True)
    parser.add_argument("--output-lock", type=Path, required=True)
    args = parser.parse_args()
    selection = json.loads(args.selection_lock.read_text(encoding="utf-8"))
    if selection.get("status") != "locked_after_validation_rm_scoring_before_any_variant_panel_judgment":
        raise RuntimeError("Round 2 selection is not locked")
    if selection.get("panel_judgments_used_for_selection") is not False:
        raise RuntimeError("panel leakage into Round 2 selection")
    best = selection["overall_validation_incumbent"]
    source = {key: best["candidate_config"][key] for key in TUNABLE
              if key in best["candidate_config"] and best["candidate_config"][key] is not None}
    recipes = []
    combined = dict(source); combined.update({"learning_rate": 2.5e-8,
                                               "reference_anchor_weight": 0.5,
                                               "preference_sft_weight": 0.05})
    recipes.append(("r3_anchor050_lr025", combined,
                    "Combine the two dominant alignment-tax controls: half step and a strong base trust region."))
    near_base = dict(source); near_base.update({"learning_rate": 2.5e-8,
                                                "reference_anchor_weight": 0.8,
                                                "preference_sft_weight": 0.08})
    recipes.append(("r3_anchor080_lr025", near_base,
                    "Test the near-base regularized Nash limit with a tighter trust region."))
    high_tau = dict(source); high_tau.update({"learning_rate": 2.5e-8, "ronpo_tau": 0.2,
                                              "ronpo_alpha": 0.1,
                                              "reference_anchor_weight": 0.3,
                                              "preference_sft_weight": 0.03})
    recipes.append(("r3_tau020_anchor030_lr025", high_tau,
                    "Increase policy KL regularization while reducing target sharpness and step size."))
    sharp_gentle = dict(source); sharp_gentle.update({"learning_rate": 2.5e-8,
                                                      "ronpo_target_column": "target_os_k0p005",
                                                      "ronpo_target_schedule_columns": [],
                                                      "ronpo_target_schedule_boundaries": [],
                                                      "ronpo_alpha": 0.1,
                                                      "reference_anchor_weight": 0.5,
                                                      "preference_sft_weight": 0.05})
    recipes.append(("r3_os_k0005_anchor050_lr025", sharp_gentle,
                    "Pair a hard worst-case OS target with a gentle policy update and strong anchoring."))
    candidates = [{"id": identifier, **recipe, "theory_note": theory,
                   "source_round2_candidate": best["candidate_id"],
                   "source_round2_checkpoint": best["model_id"]}
                  for identifier, recipe, theory in recipes]
    round2 = json.loads(args.round2_grid.read_text(encoding="utf-8"))
    grid = {"status": "frozen_before_round_launch_and_ranking", "round": 3,
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "adaptive_input": "Round 2 locked validation RM metric only; no panel judgment",
            "source_selection_lock_sha256": sha256(args.selection_lock),
            "source_validation_incumbent": {key: best[key] for key in
                ["candidate_id", "model_id", "step", "selection_metric", "selection_metric_ci95"]},
            "common": round2["common"], "candidates": candidates,
            "spent_sealed_split_touched": False}
    write(args.output_grid, grid)
    lock = {"status": "locked_before_round3_training_and_ranking",
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "grid_sha256": sha256(args.output_grid),
            "evaluator_lock_sha256": sha256(args.evaluator_lock),
            "source_selection_lock_sha256": sha256(args.selection_lock),
            "panel_judgments_used_for_adaptation": False, "spent_sealed_split_touched": False}
    write(args.output_lock, lock)
    print(json.dumps({"grid": grid, "lock": lock}, indent=2))


if __name__ == "__main__":
    main()
