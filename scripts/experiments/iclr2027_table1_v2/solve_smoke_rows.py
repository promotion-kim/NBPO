#!/usr/bin/env python3
"""Solve every matched row of the one-seed smoke on one frozen pool.

All rows share the response pool, the frozen oracle and the proximal centre, so
what differs between them is only the objective representation and the
aggregation rule -- which is the entire point of a matched comparison.

Weight norms follow the matched protocol: each representation is solved at ITS
OWN Nash multiplier norm, and the non-Nash aggregations of that representation
are solved at the same norm. A control solved on the simplex would take a step
hundreds of times smaller and would lose on step size alone.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from mnpo_scripts.nbpo_core import uniform_policy
from mnpo_scripts.nbpo_generic import (
    KSUndefinedError, matched_weight_l1, solve_finite_pool,
)
from mnpo_scripts.nbpo_representations import (
    AdaptiveGameRepresentation, BTRewardRepresentation, FixedReferenceRepresentation,
)

ROWS = ("nbpo", "fixed_reference_nash", "bt_rm_nash", "game_utilitarian", "game_ks",
        "bt_rm_utilitarian")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tensor-gpm", type=Path, required=True)
    ap.add_argument("--tensor-bt", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--eta", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.25)
    ap.add_argument("--dual-tol", type=float, default=1e-6)
    ap.add_argument("--max-dual-calls", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--rows", nargs="+", default=list(ROWS))
    args = ap.parse_args()

    A_pol = torch.from_numpy(np.load(args.tensor_gpm / "tensor_policy.npz")["A"])
    A_ref = torch.from_numpy(np.load(args.tensor_gpm / "tensor_ref.npz")["A"])
    K, X, I, _ = A_pol.shape
    mu = uniform_policy(X, I)
    beta = torch.full((K,), args.beta, dtype=torch.float64)

    game = AdaptiveGameRepresentation(A_pol, A_ref, mu, beta)
    fixed = FixedReferenceRepresentation(A_pol, A_ref, mu)
    bt_tab = np.load(args.tensor_bt / "bt_reward_table.npz")
    bt = BTRewardRepresentation(torch.from_numpy(bt_tab["r_learner"]),
                                torch.from_numpy(bt_tab["r_reference"]), mu)

    common = dict(eta=args.eta, R=1, M=args.max_dual_calls, inner_solver="exact",
                  dual_solver="root", dual_tol=args.dual_tol,
                  inner_workers=args.workers)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = {}

    nash, solved = {}, {}
    for name, rep in (("nbpo", game), ("fixed_reference_nash", fixed),
                      ("bt_rm_nash", bt)):
        if name not in args.rows:
            continue
        t0 = time.time()
        sol = solve_finite_pool(rep, "nash", **common)
        nash[name] = solved[name] = sol
        out[name] = record(name, sol, time.time() - t0, mu)
        say(name, out[name])

    # The matched norm is per REPRESENTATION, not global: the game rows take the
    # game's own Nash norm and the BT rows take BT's. Borrowing the game's norm
    # for a BT row would compare a scalar representation at a step size its own
    # dual never chose, which is the confound this protocol exists to remove.
    l1_game = matched_weight_l1(nash["nbpo"]) if "nbpo" in nash else 1.0
    l1_bt = matched_weight_l1(nash["bt_rm_nash"]) if "bt_rm_nash" in nash else 1.0
    for name, rep, agg, l1 in (("game_utilitarian", game, "utilitarian", l1_game),
                               ("game_ks", game, "kalai_smorodinsky", l1_game),
                               ("bt_rm_utilitarian", bt, "utilitarian", l1_bt)):
        if name not in args.rows:
            continue
        t0 = time.time()
        try:
            sol = solve_finite_pool(
                rep, agg, eta=args.eta, R=1, weight_l1=l1, inner_solver="exact",
                inner_workers=args.workers,
                ks_kwargs=(dict(stage1_iters=400, ideal_iters=60)
                           if agg == "kalai_smorodinsky" else None))
            solved[name] = sol
            out[name] = record(name, sol, time.time() - t0, mu)
            out[name]["matched_weight_l1"] = l1
            out[name]["matched_norm_source"] = (
                "bt_rm_nash" if rep is bt else "nbpo")
            say(name, out[name])
        except KSUndefinedError as exc:
            out[name] = {"row": name, "status": "undefined", "reason": str(exc)[:400]}
            print(f"  {name}: UNDEFINED -- {str(exc)[:120]}", flush=True)

    for name, sol in list(solved.items()):
        np.savez_compressed(args.out_dir / f"target_{name}.npz",
                            target=(torch.log(sol.pi) - torch.log(sol.pi_t)).numpy(),
                            pi=sol.pi.numpy(), q=sol.q_update.numpy(),
                            weights=sol.weights.numpy())
    (args.out_dir / "smoke_solutions.json").write_text(json.dumps(
        {"config": {k: str(v) for k, v in vars(args).items()}, "rows": out}, indent=2))
    print(f"\nwrote {args.out_dir}/smoke_solutions.json")


def finite(x):
    """``None`` for an inapplicable or non-finite diagnostic, never a bare NaN.

    ``projected_kkt_residual`` measures the Nash inverse-surplus system and has
    no meaning for a utilitarian or Kalai--Smorodinsky dual. Serialising that as
    ``NaN`` produces invalid strict JSON and, worse, reads downstream as a solver
    that failed rather than a diagnostic that does not apply.
    """
    if x is None:
        return None
    v = float(x)
    return v if math.isfinite(v) else None


def record(name, sol, secs, mu):
    tgt = (torch.log(sol.pi) - torch.log(sol.pi_t)).numpy()
    return {
        "row": name, "status": "ok",
        "representation": sol.representation, "aggregation": sol.aggregation,
        "seconds": secs,
        "weights": [float(v) for v in sol.weights],
        "weight_l1": float(sol.weights.sum()),
        "surplus": [float(v) for v in sol.surplus],
        "min_surplus": float(sol.surplus.min()),
        "avg_surplus": float(sol.surplus.mean()),
        "disagreement": [float(v) for v in sol.d],
        "dual_converged": sol.dual_converged,
        "outer_iterations_used": sol.outer_iterations_used,
        "projected_kkt_residual": finite(sol.projected_kkt_residual),
        "inverse_surplus_residual": finite(sol.kkt_residual),
        "fixed_point_residual": finite(sol.fixed_point_residual),
        "extra_map_residual": finite(sol.extra_map_residual),
        "target_identity_residual": finite(sol.target_log_ratio_check()),
        "proximal_kl": finite(sol.proximal_kl),
        "target_rms": float(np.sqrt(np.mean(tgt ** 2))),
        "target_p10": float(np.percentile(tgt, 10)),
        "target_p90": float(np.percentile(tgt, 90)),
        "target_finite": bool(np.isfinite(tgt).all()),
        "projected_kkt_applies": sol.aggregation == "nash",
    }


def say(name, r):
    print(f"  {name:22s} min_s {r['min_surplus']:+.5f}  |w|1 {r['weight_l1']:8.2f}  "
          f"pkkt {r['projected_kkt_residual'] if r['projected_kkt_residual'] is not None else float('nan'):.1e}  "
          f"inner {r['extra_map_residual']:.1e}  identity {r['target_identity_residual']:.1e}  "
          f"tgtRMS {r['target_rms']:.3f}  ({r['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
