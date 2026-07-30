#!/usr/bin/env python3
"""Adaptively freeze Round 2 from the locked, RM-only Round 1 selection."""

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


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def retarget_k005(config: dict) -> dict:
    result = dict(config)
    column = str(result.get("ronpo_target_column", "target_fullexp_k0p02"))
    if column.startswith("target_fullexp_"):
        result["ronpo_target_column"] = "target_fullexp_k0p005"
    elif column.startswith("target_os_"):
        result["ronpo_target_column"] = "target_os_k0p005"
    elif column.startswith("target_cvar_"):
        result["ronpo_target_column"] = "target_cvar_a0p3_k0p005"
    if result.get("ronpo_target_schedule_columns"):
        prefix = "target_fullexp" if "fullexp" in column else "target_os"
        result["ronpo_target_schedule_columns"] = [
            f"{prefix}_k0p05", f"{prefix}_k0p02", f"{prefix}_k0p01",
            f"{prefix}_k0p007", f"{prefix}_k0p005"]
        result["ronpo_target_schedule_boundaries"] = [0, 180, 360, 540, 720]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--evaluator-lock", type=Path, required=True)
    parser.add_argument("--round1-grid", type=Path, required=True)
    parser.add_argument("--output-grid", type=Path, required=True)
    parser.add_argument("--output-lock", type=Path, required=True)
    args = parser.parse_args()
    selection = json.loads(args.selection_lock.read_text(encoding="utf-8"))
    if selection.get("status") != "locked_after_validation_rm_scoring_before_any_variant_panel_judgment":
        raise RuntimeError("Round 1 RM-only selection is not locked")
    if selection.get("panel_judgments_used_for_selection") is not False:
        raise RuntimeError("panel leakage into Round 1 selection")
    best = selection.get("overall_validation_incumbent")
    if not best:
        raise RuntimeError("Round 1 has no eligible incumbent")
    source = {key: best["candidate_config"][key] for key in TUNABLE
              if key in best["candidate_config"] and best["candidate_config"][key] is not None}
    anchor = dict(source)
    anchor.update({"reference_anchor_weight": 0.5, "preference_sft_weight": 0.05})
    low_lr = dict(source); low_lr["learning_rate"] = 2.5e-8
    sharp = retarget_k005(source)
    os_anneal = dict(source)
    os_anneal.update({
        "ronpo_target_column": "target_os_k0p05",
        "ronpo_target_schedule_columns": ["target_os_k0p05", "target_os_k0p02",
                                           "target_os_k0p01", "target_os_k0p007",
                                           "target_os_k0p005"],
        "ronpo_target_schedule_boundaries": [0, 180, 360, 540, 720],
        "reference_anchor_weight": 0.3, "preference_sft_weight": 0.03,
        "learning_rate": 5e-8, "ronpo_alpha": 0.15,
    })
    candidates = []
    for identifier, recipe, theory in [
        ("r2_best_anchor050", anchor, "Strengthen the trust region around base (cause a/c)."),
        ("r2_best_lr025", low_lr, "Halve the policy step while retaining the Round-1 estimator (cause a/d)."),
        ("r2_best_k0005", sharp, "Sharpen the adversary toward the hard worst floor (cause b/d)."),
        ("r2_os_anneal_anchor030", os_anneal,
         "Combine objective-stratified coverage with soft-to-hard annealing and a stronger anchor (causes a/b/e)."),
    ]:
        candidates.append({"id": identifier, **recipe, "theory_note": theory,
                           "source_round1_candidate": best["candidate_id"],
                           "source_round1_checkpoint": best["model_id"]})
    round1 = json.loads(args.round1_grid.read_text(encoding="utf-8"))
    grid = {
        "status": "frozen_before_round_launch_and_ranking",
        "round": 2, "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "adaptive_input": "Round 1 locked validation RM metric only; no panel judgment",
        "source_selection_lock_sha256": sha256(args.selection_lock),
        "source_validation_incumbent": {key: best[key] for key in
            ["candidate_id", "model_id", "step", "selection_metric", "selection_metric_ci95"]},
        "common": round1["common"], "candidates": candidates,
        "spent_sealed_split_touched": False,
    }
    atomic_json(args.output_grid, grid)
    lock = {
        "status": "locked_before_round2_training_and_ranking",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "grid_sha256": sha256(args.output_grid),
        "evaluator_lock_sha256": sha256(args.evaluator_lock),
        "source_selection_lock_sha256": sha256(args.selection_lock),
        "panel_judgments_used_for_adaptation": False,
        "spent_sealed_split_touched": False,
    }
    atomic_json(args.output_lock, lock)
    print(json.dumps({"grid": grid, "lock": lock}, indent=2))


if __name__ == "__main__":
    main()
