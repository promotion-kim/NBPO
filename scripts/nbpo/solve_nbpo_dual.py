#!/usr/bin/env python3
"""Deterministic finite-pool NBPO dual solve (Algorithm 1's inner machinery).

Reads a versioned preference-tensor artifact (``build_preference_tensor.py``)
and runs projected dual gradient descent on the raw multipliers
(Eq. (27) ``eq:dual-update``) with the fixed-point weighted-policy solve of
Section 5.2 (Eq. (21), centered at the proximal center), or one of the matched
finite-game controls (utilitarian / absolute max-min / surplus max-min) on the
same tensors and budget.

Scope: every one of the ``M`` dual iterations (4e3--3e5 in the manuscript) is a
cheap tensor computation on the FROZEN finite response pool. The neural policy
is fit afterwards, once, from the targets built by ``build_nbpo_pairs.py`` --
no 8B model is retrained inside this loop.

Outputs (all raw, none normalized or clamped):
``solution.json`` -- raw lambda, V, d, surplus, the inverse-surplus residual
``||s - 1/lambda||_inf`` AND the projected (box-aware) KKT residual with the
active-bound coordinates, fixed-point and one-extra-map residuals, opponent
entropy/ESS of the final policy, the full config (beta, eta, gamma schedule,
M, R, lambda box), input artifact hashes, hashes of the opponent files, and the
iteration history; ``nu_update.npz`` -- the opponent that generated the final
policy (what Eq. (26) pair construction samples from); ``nu_final_policy.npz``
-- ``nu*`` recomputed at the final policy (diagnostics); ``pi_star.npz`` -- the
finite-pool policy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mnpo_scripts.nbpo_core import (
    uniform_policy,
    validate_centered_preference_tensor,
    validate_reference_tensor,
)
from mnpo_scripts.nbpo_solver import AGGREGATIONS, solve_nbpo_dual
from scripts.nbpo.nbpo_common import sha256_file, write_json


def load_tensor_artifact(tensor_dir: Path):
    meta = json.loads((tensor_dir / "meta.json").read_text())
    A_policy = torch.from_numpy(np.load(tensor_dir / "tensor_policy.npz")["A"])
    A_ref = torch.from_numpy(np.load(tensor_dir / "tensor_ref.npz")["A"])
    hashes = {name: sha256_file(tensor_dir / name)
              for name in ("tensor_policy.npz", "tensor_ref.npz", "meta.json")}
    return meta, A_policy, A_ref, hashes


def parse_gamma(text: str, M: int):
    parts = [float(p) for p in text.split(",") if p.strip()]
    return parts[0] if len(parts) == 1 else parts


def write_solution_artifact(out_dir: Path, res, tensor_meta: dict, hashes: dict,
                            tensor_dir: Path, stage: int, lambda_warm_started: bool) -> dict:
    """Persist a DualSolveResult as the versioned solver artifact (shared with run_nbpo_stage)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # Two opponents, written separately and hashed separately (never confuse them):
    #   nu_update.npz       -- generated the final policy; Eq. (26) targets sample z_k here
    #   nu_final_policy.npz -- nu* recomputed AT the final policy; diagnostics only
    np.savez_compressed(out_dir / "nu_update.npz", nu=res.nu_update.numpy())
    np.savez_compressed(out_dir / "nu_final_policy.npz", nu=res.nu_final_policy.numpy())
    np.savez_compressed(out_dir / "pi_star.npz", pi=res.pi.numpy())
    solution = {
        "aggregation": res.aggregation,
        "stage": int(stage),
        "objectives": tensor_meta.get("objectives"),
        "lambda_raw": [float(v) for v in res.lam],
        "V": [float(v) for v in res.V],
        "d": [float(v) for v in res.d],
        "surplus": [float(v) for v in res.surplus],
        "min_surplus": float(res.surplus.min()),
        "kkt_residual": res.kkt_residual,
        "inverse_surplus_residual": res.kkt_residual,
        "projected_kkt_residual": res.projected_kkt_residual,
        "gamma_ref": res.gamma_ref,
        "lambda_at_lower_bound": res.lambda_at_lower_bound,
        "lambda_at_upper_bound": res.lambda_at_upper_bound,
        "kkt_note": ("lambda_k = 1/s_k is an empirical equality only for coordinates "
                     "strictly inside the box; when a bound is active read "
                     "projected_kkt_residual, not inverse_surplus_residual"),
        "control_residual": res.control_residual,
        "fixed_point_residual": res.fixed_point_residual,
        "extra_map_residual": res.extra_map_residual,
        "opponent_entropy": [float(v) for v in res.opponent_entropy],
        "opponent_ess": [float(v) for v in res.opponent_ess],
        "opponent_diagnostics_from": "nu_final_policy",
        "artifact_hashes": {
            "nu_update.npz": sha256_file(out_dir / "nu_update.npz"),
            "nu_final_policy.npz": sha256_file(out_dir / "nu_final_policy.npz"),
            "pi_star.npz": sha256_file(out_dir / "pi_star.npz"),
        },
        "config": res.config,
        "lambda_warm_started": bool(lambda_warm_started),
        "input_hashes": hashes,
        "tensor_dir": str(tensor_dir),
        "history": res.history,
    }
    write_json(out_dir / "solution.json", solution)
    return solution


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tensor-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--beta", required=True,
                    help="opponent temperatures: one value, or comma list per objective "
                         "(this is opponent_betas -- NOT the metrics-only trainer `beta`)")
    ap.add_argument("--eta", type=float, default=1.0, help="proximal coefficient eta_t")
    ap.add_argument("--gamma", required=True,
                    help="dual step size gamma_m: scalar or comma list of length M")
    ap.add_argument("--dual-iterations", "-M", type=int, required=True, dest="M")
    ap.add_argument("--fixed-point-iterations", "-R", type=int, default=1, dest="R",
                    help="R=1 is the manuscript's disclosed practical approximation; the "
                         "fixed-point residual is reported either way")
    ap.add_argument("--lambda-min", type=float, default=1e-3)
    ap.add_argument("--lambda-max", type=float, default=1e3)
    ap.add_argument("--warm-start-lambda", type=Path, default=None,
                    help="solution.json of the previous outer stage (lambda warm start); "
                         "omit at t=0 (lambda initialized to ones)")
    ap.add_argument("--aggregation", choices=list(AGGREGATIONS), default="nash")
    ap.add_argument("--damping", type=float, default=0.0)
    ap.add_argument("--adversary-step", type=float, default=1.0)
    ap.add_argument("--log-every", type=int, default=0)
    ap.add_argument("--stage", type=int, default=0)
    ap.add_argument("--tensor-kind", choices=["centered_preference", "game_utility"],
                    default="centered_preference",
                    help="centered_preference (raw P-1/2, range-checked to [-1/2,1/2]) or "
                         "game_utility (positively rescaled c_k A_k; finite only)")
    args = ap.parse_args()

    meta, A_policy, A_ref, hashes = load_tensor_artifact(args.tensor_dir)
    if args.tensor_kind == "centered_preference":
        validate_centered_preference_tensor(A_policy, "A_policy")
        validate_centered_preference_tensor(A_ref, "A_ref")
    validate_reference_tensor(A_ref, "A_ref")
    K = A_policy.shape[0]
    beta_vals = [float(b) for b in args.beta.split(",") if b.strip()]
    beta = torch.tensor(beta_vals * K if len(beta_vals) == 1 else beta_vals, dtype=torch.float64)
    if beta.shape != (K,):
        raise ValueError(f"--beta must give 1 or {K} values, got {len(beta_vals)}")
    lambda_init = None
    if args.warm_start_lambda is not None:
        prev = json.loads(args.warm_start_lambda.read_text())
        lambda_init = torch.tensor(prev["lambda_raw"], dtype=torch.float64)

    mu = uniform_policy(A_policy.shape[1], A_policy.shape[3])
    res = solve_nbpo_dual(
        A_policy, A_ref, mu, beta,
        eta=args.eta, gamma=parse_gamma(args.gamma, args.M), M=args.M, R=args.R,
        lambda_box=(args.lambda_min, args.lambda_max),
        lambda_init=lambda_init, aggregation=args.aggregation,
        damping=args.damping, adversary_step=args.adversary_step,
        log_every=args.log_every,
    )

    solution = write_solution_artifact(
        args.out_dir, res, meta, hashes, args.tensor_dir, args.stage,
        lambda_warm_started=lambda_init is not None,
    )
    print(json.dumps({k: solution[k] for k in
                      ("aggregation", "lambda_raw", "surplus", "min_surplus",
                       "inverse_surplus_residual", "projected_kkt_residual",
                       "lambda_at_lower_bound", "lambda_at_upper_bound",
                       "control_residual", "fixed_point_residual", "extra_map_residual")},
                     indent=1))


if __name__ == "__main__":
    main()
