#!/usr/bin/env python3
"""Game-value evaluator for NBPO (the finite-temperature counterpart of eval_bpo_surplus).

On held-out prompts with a separately generated reference comparator pool
(a preference-tensor artifact from ``build_preference_tensor.py``), computes:

- ``V_{k,beta}(pi)`` (Eq. (8)) for the evaluated policy's response pool,
- ``d_k = V_{k,beta}(mu)`` (Eq. (10)) from the reference-as-learner tensor,
- ``s_k = V_{k,beta}(pi) - d_k``, min and average surplus,
- Nash welfare ``sum_k log s_k`` ONLY when every ``s_k > 0``: when any surplus
  is nonpositive the output carries ``nash_welfare: null`` and
  ``nash_welfare_defined: false`` -- surpluses are NEVER clamped before the log,
- opponent entropy and effective sample size at the evaluated policy.

The legacy ``scripts/bpo/eval_bpo_surplus.py`` is a fixed-reference
(beta = infinity) diagnostic that clamps nonpositive surpluses; it is kept,
separately labeled, and is NOT this evaluator.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from mnpo_scripts.nbpo_core import (
    compute_disagreement_point,
    compute_margins,
    compute_regularized_game_value,
    compute_regularized_opponent,
    opponent_entropy,
    opponent_ess,
    uniform_policy,
)
from scripts.nbpo.nbpo_common import sha256_file, write_json


def evaluate_game_value(A_policy: torch.Tensor, A_ref: torch.Tensor, beta: torch.Tensor,
                        reference_construction: str = "shared_pool") -> dict:
    """Pure evaluation given the two centered tensors; policy uniform over its pool."""
    K, X, I, J = A_policy.shape
    mu = uniform_policy(X, J)
    pi = uniform_policy(X, I)
    r = compute_margins(A_policy, pi)
    V = compute_regularized_game_value(r, mu, beta, form="softmin")
    d = compute_disagreement_point(A_ref, mu, beta, reference_construction)
    s = V - d
    nu = compute_regularized_opponent(r, mu, beta)
    all_positive = bool((s > 0).all())
    return {
        "V": [float(v) for v in V],
        "d": [float(v) for v in d],
        "surplus": [float(v) for v in s],
        "min_surplus": float(s.min()),
        "avg_surplus": float(s.mean()),
        "nash_welfare_defined": all_positive,
        # Nash welfare only exists on the individually-rational set (Eq. (11));
        # a nonpositive surplus makes it undefined, not "very negative".
        "nash_welfare": float(torch.log(s).sum()) if all_positive else None,
        "opponent_entropy": [float(v) for v in opponent_entropy(nu)],
        "opponent_ess": [float(v) for v in opponent_ess(nu)],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tensor-dir", type=Path, required=True,
                    help="held-out preference-tensor artifact (policy = the evaluated model)")
    ap.add_argument("--beta", required=True,
                    help="opponent temperatures: one value or comma list per objective")
    ap.add_argument("--label", default="policy")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    meta = json.loads((args.tensor_dir / "meta.json").read_text())
    A_policy = torch.from_numpy(np.load(args.tensor_dir / "tensor_policy.npz")["A"])
    A_ref = torch.from_numpy(np.load(args.tensor_dir / "tensor_ref.npz")["A"])
    K = A_policy.shape[0]
    beta_vals = [float(b) for b in args.beta.split(",") if b.strip()]
    beta = torch.tensor(beta_vals * K if len(beta_vals) == 1 else beta_vals, dtype=torch.float64)
    if beta.shape != (K,):
        raise ValueError(f"--beta must give 1 or {K} values, got {len(beta_vals)}")

    construction = meta.get("reference_construction")
    if construction is None:
        raise ValueError(
            f"{args.tensor_dir}/meta.json declares no reference_construction "
            "('shared_pool' or 'independent_samples')")
    result = evaluate_game_value(A_policy, A_ref, beta, construction)
    out = {
        "label": args.label,
        "objectives": meta["objectives"],
        "n_prompts": len(meta["prompt_ids"]),
        "beta": [float(b) for b in beta],
        "reference_construction": construction,
        **result,
        "comparator_pool_hash": sha256_file(args.tensor_dir / "tensor_ref.npz"),
        "tensor_policy_hash": sha256_file(args.tensor_dir / "tensor_policy.npz"),
        "judge_models": meta.get("judge_models"),
        "rubric_versions": meta.get("rubric_versions"),
        "evaluator": "game_value (finite-temperature; NOT the fixed-reference eval_bpo_surplus)",
    }
    print(json.dumps(out, indent=2))
    if args.out:
        write_json(args.out, out)


if __name__ == "__main__":
    main()
