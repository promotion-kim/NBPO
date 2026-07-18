#!/usr/bin/env python3
"""Choose the P4 kappa grid from real-pool sigma entropy, reward-blind."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from datasets import load_from_disk


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def entropy_for_row(example: dict, kappa: float) -> float:
    names = list(example["objective_names"])
    scores = np.asarray([example["normalized_objective_scores"][name] for name in names], dtype=float)
    scale = float(example.get("homogeneous_oracle_preference_scale", 8.0))
    pairwise = sigmoid(scale * (scores[:, :, None] - scores[:, None, :]))
    cost = pairwise.mean(axis=1).reshape(-1)
    logits = -cost / kappa
    logits -= logits.max()
    sigma = np.exp(logits)
    sigma /= sigma.sum()
    return float(-(sigma * np.log(np.maximum(sigma, 1e-300))).sum() / math.log(len(sigma)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--entropy-targets", default="0.05,0.15,0.35,0.55,0.85")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidates = [float(value) for value in args.candidates.split(",")]
    targets = [float(value) for value in args.entropy_targets.split(",")]
    if len(set(candidates)) != len(candidates) or any(value <= 0 for value in candidates):
        raise ValueError("candidate kappas must be unique and positive")

    loaded = load_from_disk(str(args.input_dir))
    splits = list(loaded.keys()) if hasattr(loaded, "keys") else [None]
    rows = []
    fingerprints = {}
    for split in splits:
        ds = loaded[split] if split is not None else loaded
        fingerprints[split or "dataset"] = getattr(ds, "_fingerprint", None)
        rows.extend(ds)
    if not rows:
        raise RuntimeError("empty precomputed dataset")
    entropy_map = {}
    for kappa in candidates:
        values = np.asarray([entropy_for_row(row, kappa) for row in rows], dtype=float)
        entropy_map[f"{kappa:g}"] = {
            "kappa": kappa,
            "mean_normalized_sigma_entropy": float(values.mean()),
            "population_std": float(values.std()),
            "rows": int(values.size),
        }
    selected, used = [], set()
    for target in targets:
        options = sorted(
            (entry for entry in entropy_map.values() if entry["kappa"] not in used),
            key=lambda entry: (abs(entry["mean_normalized_sigma_entropy"] - target), entry["kappa"]),
        )
        if not options:
            raise RuntimeError("not enough distinct kappa candidates")
        choice = options[0]
        used.add(choice["kappa"])
        selected.append({
            "entropy_target": target,
            "selected_kappa": choice["kappa"],
            "measured_entropy": choice["mean_normalized_sigma_entropy"],
            "absolute_distance": abs(choice["mean_normalized_sigma_entropy"] - target),
        })
    confirmatory = next(item for item in selected if item["entropy_target"] == 0.55)
    payload = {
        "status": "locked_before_training",
        "selection_basis": "real precomputed-pool normalized sigma entropy only; no policy, reward-panel, validation, or fresh outcome consulted",
        "input_dir": str(args.input_dir),
        "input_fingerprints": fingerprints,
        "candidate_kappas": candidates,
        "entropy_targets": targets,
        "entropy_map": entropy_map,
        "selected": selected,
        "confirmatory_os_kappa": confirmatory["selected_kappa"],
        "confirmatory_entropy_target": 0.55,
        "tie_break": "lower kappa after entropy distance; no duplicate kappa",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
