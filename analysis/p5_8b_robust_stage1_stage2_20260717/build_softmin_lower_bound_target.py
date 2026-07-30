#!/usr/bin/env python3
"""Add a certified smooth worst-objective lower-bound RONPO target.

For every sampled response pair, let ``g_k`` be its expected RONPO advantage
against the (uniform) response opponent for objective ``k``.  This builder
adds

    r_tau(g) = -tau log sum_k exp(-g_k / tau).

For every finite objective vector, ``min(g)-tau*log(K) <= r_tau(g) <=
min(g)``.  Thus maximising the exact target maximises a differentiable lower
bound on the pair's worst-objective advantage.  The statement is algebraic;
it does not claim a global neural-network optimisation guarantee.

The script never changes pairs or precomputed log probabilities.  It writes a
new DatasetDict and an audit so the new arm remains directly traceable to the
frozen Stage-1 pool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from datasets import load_from_disk


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def _tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _row_target(example: dict, tau: float, column: str) -> dict:
    names = list(example["objective_names"])
    scores = np.asarray(
        [example["normalized_objective_scores"][name] for name in names], dtype=np.float64
    )
    scale = float(example.get("homogeneous_oracle_preference_scale", 8.0))
    chosen = int(example["chosen_index"])
    rejected = int(example["rejected_index"])
    pref = _sigmoid(scale * (scores[:, :, None] - scores[:, None, :]))
    zhat = pref[:, chosen, :] - pref[:, rejected, :]
    g = zhat.mean(axis=1)
    shifted = -g / tau
    maximum = float(shifted.max())
    lower_bound = -tau * (maximum + math.log(float(np.exp(shifted - maximum).sum())))
    if not math.isfinite(lower_bound):
        raise RuntimeError("non-finite softmin lower-bound target")
    return {
        column: float(lower_bound),
        f"{column}_per_objective_advantages": [float(value) for value in g],
    }


def _sha_tree(path: Path) -> str:
    digest = hashlib.sha256()
    for file in sorted(path.rglob("*")):
        if not file.is_file():
            continue
        digest.update(str(file.relative_to(path)).encode("utf-8"))
        digest.update(file.read_bytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--num-proc", type=int, default=12)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    if not (args.tau > 0.0 and math.isfinite(args.tau)):
        raise ValueError("tau must be a positive finite number")
    column = f"target_softmin_lb_tau{_tag(args.tau)}"
    source_sha = _sha_tree(args.input_dir)
    dataset = load_from_disk(args.input_dir)

    def add(example: dict) -> dict:
        return _row_target(example, args.tau, column)

    updated = dataset.map(add, num_proc=args.num_proc, desc="smooth worst-objective lower-bound target")
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    updated.save_to_disk(args.output_dir)
    train = updated["train"] if hasattr(updated, "keys") and "train" in updated else updated
    values = np.asarray(train[column], dtype=np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError("saved target contains non-finite values")
    gap = args.tau * math.log(len(train[0]["objective_names"]))
    audit = {
        "status": "complete",
        "input_dir": str(args.input_dir),
        "input_tree_sha256": source_sha,
        "output_dir": str(args.output_dir),
        "target_column": column,
        "tau": args.tau,
        "objective_count": len(train[0]["objective_names"]),
        "theorem": {
            "formula": "r_tau(g)=-tau*log(sum_k exp(-g_k/tau))",
            "bound": "min(g)-tau*log(K) <= r_tau(g) <= min(g)",
            "max_additive_gap": gap,
            "scope": "algebraic target bound; no claim of global non-convex optimisation optimality",
        },
        "train_rows": len(train),
        "target_stats": {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "mean_abs": float(np.abs(values).mean()),
            "p05": float(np.quantile(values, 0.05)),
            "p50": float(np.quantile(values, 0.50)),
            "p95": float(np.quantile(values, 0.95)),
        },
        "spent_sealed_split_touched": False,
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
