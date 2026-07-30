#!/usr/bin/env python3
"""Build frozen RONPO variant targets from the existing matched precompute.

No response is decoded and no log probability is recomputed.  The script adds
full-expectation, objective-stratified, and objective-CVaR target columns while
preserving every training row and its frozen reference/history logps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from datasets import load_from_disk


def sigmoid(value: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-value))


def tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prompt_seed(example: dict, suffix: str) -> int:
    text = f"{example.get('prompt_id') or example.get('prompt')}|{suffix}"
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def cvar_objective_weights(values: np.ndarray, alpha: float) -> np.ndarray:
    """Worst-tail CVaR weights for a uniform discrete objective prior.

    Smaller values are worse.  The density cap p_k / alpha gives the standard
    dual representation of lower-tail CVaR on the objective distribution.
    """
    count = len(values)
    cap = 1.0 / (count * alpha)
    remaining = 1.0
    weights = np.zeros(count, dtype=float)
    for index in np.argsort(values):
        mass = min(cap, remaining)
        weights[int(index)] = mass
        remaining -= mass
        if remaining <= 1e-12:
            break
    if remaining > 1e-9:
        raise RuntimeError("CVaR objective weights did not sum to one")
    return weights / weights.sum()


def targets(example: dict, kappas: tuple[float, ...], cvar_alpha: float) -> dict:
    names = example["objective_names"]
    scores = np.asarray([example["normalized_objective_scores"][name] for name in names], dtype=float)
    objective_count, response_count = scores.shape
    scale = float(example.get("homogeneous_oracle_preference_scale", 8.0))
    chosen = int(example["chosen_index"])
    rejected = int(example["rejected_index"])
    difference = scores[:, :, None] - scores[:, None, :]
    preference = sigmoid(scale * difference)
    cost = preference.mean(axis=1)
    zhat = preference[:, chosen, :] - preference[:, rejected, :]
    output: dict[str, float] = {}
    for kappa in kappas:
        suffix = tag(kappa)
        logits = -cost.reshape(-1) / kappa
        logits -= logits.max()
        sigma = np.exp(logits).reshape(objective_count, response_count)
        sigma /= sigma.sum()
        omega = sigma.sum(axis=1)
        conditional = sigma / omega[:, None]
        output[f"target_fullexp_k{suffix}"] = float((sigma * zhat).sum())

        rng = np.random.default_rng(prompt_seed(example, f"os-k{kappa:g}"))
        os_target = 0.0
        for objective in range(objective_count):
            opponent = int(rng.choice(response_count, p=conditional[objective]))
            os_target += float(omega[objective] * zhat[objective, opponent])
        output[f"target_os_k{suffix}"] = float(os_target)

        conditional_cost = (conditional * cost).sum(axis=1)
        cvar_weights = cvar_objective_weights(conditional_cost, cvar_alpha)
        output[f"target_cvar_a{tag(cvar_alpha)}_k{suffix}"] = float(
            sum(cvar_weights[k] * np.dot(conditional[k], zhat[k]) for k in range(objective_count))
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--kappas", default="0.05,0.02,0.01,0.007,0.005")
    parser.add_argument("--cvar-alpha", type=float, default=0.3)
    parser.add_argument("--num-proc", type=int, default=12)
    args = parser.parse_args()
    kappas = tuple(float(value) for value in args.kappas.split(","))
    if any(value <= 0 for value in kappas) or not 0 < args.cvar_alpha <= 1:
        raise ValueError("invalid kappa or CVaR alpha")
    source = load_from_disk(str(args.input_dir))
    reference = source["train"]
    max_error = 0.0
    validation_modes: set[str] = set()
    for index in range(min(200, len(reference))):
        example = reference[index]
        names = example["objective_names"]
        scores = np.asarray([example["normalized_objective_scores"][name] for name in names], dtype=float)
        scale = float(example.get("homogeneous_oracle_preference_scale", 8.0))
        chosen, rejected = int(example["chosen_index"]), int(example["rejected_index"])
        objective = int(example["ronpo_objective_index"])
        opponent = int(example["ronpo_adversary_response_index"])
        if objective >= 0 and opponent >= 0:
            # Atom-level rows store z_hat(k, a) directly.
            reconstructed = float(
                sigmoid(scale * (scores[objective, chosen] - scores[objective, opponent]))
                - sigmoid(scale * (scores[objective, rejected] - scores[objective, opponent]))
            )
            validation_modes.add("atom")
        else:
            # The flagship full-expectation pool is Rao-Blackwellized: its
            # rows use objective/adversary index -1 and store the expectation
            # under the precomputed sigma, rather than one atom's gap.
            sigma_record = example.get("ronpo_sigma")
            if not sigma_record:
                raise RuntimeError("expected-relative row is missing ronpo_sigma")
            sigma = np.asarray([sigma_record[name] for name in names], dtype=float)
            difference = scores[:, :, None] - scores[:, None, :]
            preference = sigmoid(scale * difference)
            zhat = preference[:, chosen, :] - preference[:, rejected, :]
            reconstructed = float((sigma * zhat).sum())
            validation_modes.add("expected")
        max_error = max(max_error, abs(reconstructed - float(example["ronpo_objective_gap"])))
    if max_error >= 1e-6:
        raise RuntimeError(f"target reconstruction failed: max error {max_error}")

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    built = source.map(
        lambda example: targets(example, kappas, args.cvar_alpha),
        num_proc=args.num_proc,
        desc="RONPO variant targets",
    )
    built.save_to_disk(str(args.output_dir))
    train = built["train"]
    target_columns = sorted(column for column in train.column_names if column.startswith("target_"))
    distributions = {}
    for column in target_columns:
        values = np.asarray(train[column], dtype=float)
        distributions[column] = {
            "mean": float(values.mean()), "std": float(values.std()),
            "mean_abs": float(np.abs(values).mean()), "min": float(values.min()), "max": float(values.max()),
        }
    manifest = {
        "status": "completed", "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": str(args.input_dir), "destination": str(args.output_dir),
        "kappas": list(kappas), "cvar_alpha": args.cvar_alpha,
        "counts": {split: len(dataset) for split, dataset in built.items()},
        "reconstruction_max_abs_error": max_error,
        "reconstruction_validation_modes": sorted(validation_modes),
        "target_distributions": distributions,
        "training_rows_and_precomputed_logps_preserved": True,
        "spent_sealed_split_touched": False,
    }
    manifest_path = args.output_dir.parent / "target_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**manifest, "manifest_sha256": sha256(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
