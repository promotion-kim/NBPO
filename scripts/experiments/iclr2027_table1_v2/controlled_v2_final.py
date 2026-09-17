#!/usr/bin/env python3
"""The final feasibility-preserving controlled-nontransitivity benchmark.

Ten methods on one generated instance per (seed, alpha), every one of them
scored under the **same true adaptive-game values** so that nothing in the
headline table is a method-native quantity in method-native units.

Construction (`nontransitivity_v2.match_rho_star`)
--------------------------------------------------
`A_k(alpha) = T + base_k + alpha * C_k`, adding a cyclic component to a fixed
transitive one instead of interpolating between them, so that

* `C_k` is a circulant tournament on an **odd** number of responses and
  therefore has exactly zero row sums, hence zero column sums, hence contributes
  nothing to `d_k = V_k(mu)` -- the disagreement point provably does not move
  with alpha, and the builder raises if it drifts by more than 1e-12;
* amplitudes are chosen so `|A| <= 1/2` holds outright, so **nothing is
  clipped** and the reference/witness geometry is never deformed;
* skew symmetry is exact and the diagonal is exactly zero, both asserted;
* the weight on the shared transitive direction is bisected per alpha to hold
  `rho*` at the declared target as closely as the amplitude cap allows.

Convergence, not iteration count
--------------------------------
A cell is only usable if it actually solved. The criteria are checked per
method and a cell that misses any of them is marked non-converged and is
excluded from the summary rather than being quoted:

    projected KKT residual  < 1e-6      (dual stationarity, box-aware)
    inverse-surplus residual< 1e-6      (only where no box bound is active)
    inner fixed-point resid < 1e-4
    individual rationality              (for rules whose solution must be IR)

Reaching an iteration cap is itself a failure, not a result.
"""
from __future__ import annotations

import argparse
import json
import math
import resource
import statistics
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
from scripts.experiments.iclr2027_table1_v2 import nontransitivity_feasibility as nf
from scripts.experiments.iclr2027_table1_v2.controlled_nontransitivity import bt_fit
from scripts.experiments.iclr2027_table1_v2.nontransitivity_v2 import (
    build_v2, build_v2_projected, match_rho_star,
)

ALPHAS = (0.00, 0.25, 0.50, 0.75, 1.00)

# The order the paper reports them in: what is attainable, then what the exact
# rules achieve, then what a practical solver achieves, then the matched
# controls, then the floor.
METHODS = [
    "reference",
    "exact_global_nash",
    "exact_proximal_nash",
    "nbpo_direct",
    "nbpo_rstep_legacy",
    "fixed_reference_nash",
    "bt_rm_nash",
    "game_utilitarian",
    "game_ks",
    "bt_rm_utilitarian",
]

LABEL = {
    "reference": r"Reference $\piref$",
    "exact_global_nash": r"Exact global Nash",
    "exact_proximal_nash": r"Exact proximal Nash",
    "nbpo_direct": r"\NBPO{} (direct inner solve)",
    "nbpo_rstep_legacy": r"\NBPO{} ($R$-step map, legacy)",
    "fixed_reference_nash": r"Fixed-reference Nash",
    "bt_rm_nash": r"BT-RM--Nash",
    "game_utilitarian": r"Game-utilitarian",
    "game_ks": r"Game-KS",
    "bt_rm_utilitarian": r"BT-RM--utilitarian",
}

# Rules whose solution is supposed to be individually rational.
IR_REQUIRED = {"game_ks"}

# The legacy R-step map is included precisely BECAUSE it fails to converge; its
# non-convergence is the ablation's result, not a defect to be excluded. It is
# flagged rather than dropped so the table can show it failing.
EXPECTED_NON_CONVERGENT = {"nbpo_rstep_legacy"}

TOL = {"projected_kkt": 1e-6, "inverse_surplus": 1e-6, "inner_fixed_point": 1e-4,
       "ir": -1e-9}


# Which columns are mathematically DEFINED for which method. A blank cell is not
# a failed convergence -- for most of these rows the quantity does not exist:
#
#   * `reference` is not solved at all, so no solver diagnostic applies;
#   * the two `exact_*` rows are solved by direct concave maximization, so they
#     have no Eq. (21) map residual and no dual multipliers;
#   * `utilitarian` and `kalai_smorodinsky` carry no Nash dual, so the
#     inverse-surplus and projected-KKT residuals do not exist for them.
#
# `nash_welfare` is different again: it is APPLICABLE everywhere and simply
# UNDEFINED where some surplus is nonpositive, which is a result about the
# policy rather than a gap in the record. The three states are written as a
# number, `NA`, and `undefined`, and never as a bare blank.
SOLVER_DIAGNOSTIC_COLUMNS = (
    "fixed_point_residual", "extra_map_residual", "target_identity_residual",
    "raw_multiplier_l1", "proximal_kl")
DUAL_ONLY_COLUMNS = ("inverse_surplus_residual", "projected_kkt_residual",
                     "outer_iterations_used")
NO_SOLVER_METHODS = {"reference", "exact_global_nash", "exact_proximal_nash"}
NO_DUAL_METHODS = NO_SOLVER_METHODS | {"game_utilitarian", "game_ks",
                                       "bt_rm_utilitarian"}


def applicability(method: str, column: str) -> bool:
    """Is ``column`` a quantity that EXISTS for ``method``?"""
    if column in SOLVER_DIAGNOSTIC_COLUMNS:
        return method not in NO_SOLVER_METHODS
    if column in DUAL_ONLY_COLUMNS:
        return method not in NO_DUAL_METHODS
    return True


def cell(row: dict, column: str):
    """One CSV cell, distinguishing NA from undefined from a value."""
    if not applicability(row["method"], column):
        return "NA"
    v = row.get(column)
    if v is None or v == "":
        return "undefined" if column == "nash_welfare" else "NA"
    return v


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def bt_scores(A: np.ndarray) -> np.ndarray:
    K, X, I, _ = A.shape
    r = np.zeros((K, X, I))
    for k in range(K):
        for x in range(X):
            _, sc = bt_fit(0.5 + A[k, x])
            r[k, x] = sc
    return r


def build_reps(A: np.ndarray, beta: np.ndarray):
    K, X, I, _ = A.shape
    At = torch.from_numpy(A)
    mu_t = uniform_policy(X, I)
    r = torch.from_numpy(bt_scores(A))
    return {
        "adaptive_game": AdaptiveGameRepresentation(
            At, At, mu_t, torch.full((K,), float(beta[0]), dtype=torch.float64)),
        "fixed_reference": FixedReferenceRepresentation(At, At, mu_t),
        "bt_reward": BTRewardRepresentation(r, r, mu_t),
    }, mu_t


def convergence(name: str, sol, surplus: np.ndarray) -> dict:
    """Every criterion, evaluated explicitly, with the reasons it failed."""
    reasons = []
    pk = getattr(sol, "projected_kkt_residual", None)
    inv = getattr(sol, "kkt_residual", None)
    # The inner criterion is `extra_map_residual` -- ||map(pi) - pi||, i.e. how
    # far one MORE application of the Eq. (21) map moves the returned policy.
    # `fixed_point_residual` is the last-iteration CHANGE, which for a constant-map
    # representation at R = 1 is the move away from the proximal centre and is
    # nonzero by definition even though the solve is exact. Using it as the
    # criterion marked every BT-RM-utilitarian cell non-converged at 0.797 when
    # its extra-map residual is 0.
    fp = getattr(sol, "extra_map_residual", None)
    lower = getattr(sol, "lambda_at_lower_bound", []) or []
    upper = getattr(sol, "lambda_at_upper_bound", []) or []
    box_active = bool(lower or upper)
    if pk is not None and pk >= TOL["projected_kkt"]:
        reasons.append(f"projected_kkt {pk:.2e} >= {TOL['projected_kkt']:.0e}")
    if inv is not None and not box_active and inv >= TOL["inverse_surplus"]:
        reasons.append(f"inverse_surplus {inv:.2e} >= {TOL['inverse_surplus']:.0e} "
                       "with no active box bound")
    if fp is not None and fp >= TOL["inner_fixed_point"]:
        reasons.append(f"inner_extra_map {fp:.2e} >= {TOL['inner_fixed_point']:.0e}")
    if getattr(sol, "dual_converged", None) is False:
        reasons.append("dual solver hit its evaluation cap")
    if name in IR_REQUIRED and float(surplus.min()) < TOL["ir"]:
        reasons.append(f"individual rationality violated: min surplus "
                       f"{float(surplus.min()):.3e}")
    return {"converged": not reasons,
            "expected_non_convergent": name in EXPECTED_NON_CONVERGENT,
            "non_converged_reasons": reasons,
            "box_bound_active": box_active,
            "lambda_at_lower_bound": lower, "lambda_at_upper_bound": upper}


def evaluate(A, mu, beta, d, pi, pi_ref, refs):
    s = nf.game_values(A, pi, mu, beta) - d
    out = {
        "surplus_per_objective": s.tolist(),
        "min_surplus": float(s.min()),
        "avg_surplus": float(s.mean()),
        "nash_welfare": float(np.log(s).sum()) if (s > 0).all() else None,
        "nash_welfare_defined": bool((s > 0).all()),
        "exploitability": nf.exploitability(A, pi),
        "kl_from_reference": nf.mean_kl(pi, pi_ref),
        "target_log_ratio_l2": float(np.linalg.norm(
            np.log(np.clip(pi, nf.FLOOR, None)) - np.log(np.clip(pi_ref, nf.FLOOR, None)))),
    }
    for nm, ref in refs.items():
        out[f"tv_to_{nm}"] = nf.total_variation(pi, ref)
    return out, s


def solve_all(A, mu, beta, d, eta, rstep_M, rstep_R, dual_tol, max_dual_calls):
    """One instance, ten methods, each timed and each convergence-checked."""
    reps, mu_t = build_reps(A, beta)
    X, I = A.shape[1], A.shape[2]
    out = {}

    def timed(fn):
        t0, m0 = time.time(), peak_rss_mb()
        val = fn()
        return val, {"wall_clock_seconds": time.time() - t0,
                     "peak_rss_mb": max(peak_rss_mb(), m0)}

    # --- exact references -------------------------------------------------
    feas = nf.solve_max_min(A, mu, beta, d)
    if feas["rho_star"] <= 1e-8:
        raise RuntimeError(f"instance is outside Assumption 1: rho*={feas['rho_star']:.3e}")
    (pi_a, res_a, _), t_a = timed(lambda: nf.solve_nash(A, mu, beta, d, pi0=feas["pi"]))
    (pi_b, res_b, _), t_b = timed(lambda: nf.solve_nash(A, mu, beta, d, pi0=feas["pi"],
                                                        eta=eta, pi_ref=mu))
    out["exact_global_nash"] = {"pi": pi_a, "timing": t_a, "sol": None,
                                "extra": {"slsqp_status": int(res_a.status)}}
    out["exact_proximal_nash"] = {"pi": pi_b, "timing": t_b, "sol": None,
                                  "extra": {"slsqp_status": int(res_b.status)}}
    out["reference"] = {"pi": mu, "timing": {"wall_clock_seconds": 0.0,
                                             "peak_rss_mb": peak_rss_mb()},
                        "sol": None, "extra": {}}

    # --- Nash rows, dual solved as a root problem, inner solved directly ---
    nash = {}
    for name, key in (("nbpo_direct", "adaptive_game"),
                      ("fixed_reference_nash", "fixed_reference"),
                      ("bt_rm_nash", "bt_reward")):
        sol, t = timed(lambda key=key: solve_finite_pool(
            reps[key], "nash", eta=eta, M=max_dual_calls, R=1,
            inner_solver="exact", dual_solver="root", dual_tol=dual_tol))
        nash[name] = sol
        out[name] = {"pi": sol.pi.numpy(), "timing": t, "sol": sol, "extra": {}}

    # --- the legacy R-step map, kept as the failure ablation ---------------
    sol, t = timed(lambda: solve_finite_pool(
        reps["adaptive_game"], "nash", eta=eta, M=rstep_M, R=rstep_R,
        gamma=0.5, inner_solver="fixed_point", dual_solver="subgradient"))
    out["nbpo_rstep_legacy"] = {"pi": sol.pi.numpy(), "timing": t, "sol": sol,
                                "extra": {}}

    # --- matched-norm aggregation controls --------------------------------
    l1_game = matched_weight_l1(nash["nbpo_direct"])
    l1_bt = matched_weight_l1(nash["bt_rm_nash"])
    for name, key, agg, l1 in (
            ("game_utilitarian", "adaptive_game", "utilitarian", l1_game),
            ("bt_rm_utilitarian", "bt_reward", "utilitarian", l1_bt),
            ("game_ks", "adaptive_game", "kalai_smorodinsky", l1_game)):
        try:
            sol, t = timed(lambda key=key, agg=agg, l1=l1: solve_finite_pool(
                reps[key], agg, eta=eta, R=1, weight_l1=l1, inner_solver="exact",
                ks_kwargs=(dict(stage1_iters=400, ideal_iters=60)
                           if agg == "kalai_smorodinsky" else None)))
            out[name] = {"pi": sol.pi.numpy(), "timing": t, "sol": sol,
                         "extra": {"matched_weight_l1": l1}}
        except KSUndefinedError as exc:
            out[name] = {"pi": None, "timing": {"wall_clock_seconds": 0.0,
                                                "peak_rss_mb": peak_rss_mb()},
                         "sol": None, "extra": {"undefined": str(exc)[:300],
                                                "matched_weight_l1": l1}}
    return out, feas


def run_instance(inst, seed, alpha, args):
    A, mu, beta, d = inst["A"], inst["mu"], inst["beta"], inst["d"]
    K, X, I, _ = A.shape
    assert np.abs(A + np.swapaxes(A, -1, -2)).max() < 1e-12, "not skew-symmetric"
    idx = np.arange(I)
    assert np.abs(A[..., idx, idx]).max() == 0.0, "nonzero diagonal"
    assert np.abs(A).max() <= 0.5 + 1e-12, "payoff out of range without clipping"

    devs = [bt_fit(0.5 + A[k, x])[0] for k in range(K) for x in range(X)]
    solved, feas = solve_all(A, mu, beta, d, args.eta, args.rstep_dual_iterations,
                             args.rstep_fixed_point_iterations, args.dual_tol,
                             args.max_dual_calls)
    refs = {"exact_global": solved["exact_global_nash"]["pi"],
            "exact_proximal": solved["exact_proximal_nash"]["pi"]}

    rows = []
    for name in METHODS:
        blk = solved[name]
        base = {"seed": seed, "alpha": alpha, "method": name,
                "bt_deviance_per_edge": float(np.mean(devs)),
                "rho_star": feas["rho_star"],
                "rho_star_certified_upper_bound": feas["certified_upper_bound"],
                "rho_star_gap": feas["rho_star_gap"],
                "disagreement": d.tolist(),
                **blk["timing"], **blk["extra"]}
        if blk["pi"] is None:
            rows.append({**base, "status": "undefined", "converged": False,
                         "non_converged_reasons": ["rule undefined on this instance"]})
            continue
        ev, s = evaluate(A, mu, beta, d, blk["pi"], mu, refs)
        sol = blk["sol"]
        conv = (convergence(name, sol, s) if sol is not None
                else {"converged": True, "non_converged_reasons": [],
                      "box_bound_active": False})
        rows.append({
            **base, "status": "ok", **ev, **conv,
            "normalized_min_surplus": float(s.min()) / feas["rho_star"],
            "raw_multiplier_l1": (float(sol.weights.sum()) if sol is not None else None),
            "fixed_point_residual": (sol.fixed_point_residual if sol is not None else None),
            "extra_map_residual": (sol.extra_map_residual if sol is not None else None),
            "inverse_surplus_residual": (sol.kkt_residual if sol is not None else None),
            "projected_kkt_residual": (sol.projected_kkt_residual if sol is not None else None),
            "outer_iterations_used": (sol.outer_iterations_used if sol is not None else None),
            "proximal_kl": (sol.proximal_kl if sol is not None else None),
            "target_identity_residual": (sol.target_log_ratio_check()
                                         if sol is not None else None),
        })
    return rows


def _report(got, seed, alpha, elapsed, args):
    nc = [r["method"] for r in got
          if not r.get("converged") and not r.get("expected_non_convergent")]
    best = {r["method"]: r.get("normalized_min_surplus") for r in got}
    f = lambda k: ("  n/a" if best.get(k) is None else f"{best[k]:+.3f}")
    print(f"  [{args.family[:4]} b={args.beta}] seed {seed} alpha {alpha:.2f} "
          f"rho*={got[0]['rho_star']:+.6f} BTdev={got[0]['bt_deviance_per_edge']:.4f} "
          f"nbpo={f('nbpo_direct')} ks={f('game_ks')} gutil={f('game_utilitarian')} "
          f"fixref={f('fixed_reference_nash')} btrm={f('bt_rm_nash')} "
          f"btutil={f('bt_rm_utilitarian')} "
          f"non-conv={nc or 'none'} ({elapsed:.0f}s)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--objectives", type=int, default=4)
    ap.add_argument("--prompts", type=int, default=40)
    ap.add_argument("--responses", type=int, default=5)
    ap.add_argument("--beta", type=float, default=0.25)
    ap.add_argument("--eta", type=float, default=1.0)
    ap.add_argument("--target-rho-star", type=float, default=0.030)
    ap.add_argument("--base-amp", type=float, default=0.05)
    ap.add_argument("--cycle-amp", type=float, default=0.28)
    ap.add_argument("--margin", type=float, default=0.16,
                    help="shared-direction weight for the projected family "
                         "(the circulant family bisects it instead)")
    ap.add_argument("--dual-tol", type=float, default=1e-6)
    ap.add_argument("--max-dual-calls", type=int, default=800)
    ap.add_argument("--rstep-dual-iterations", type=int, default=1500)
    ap.add_argument("--rstep-fixed-point-iterations", type=int, default=3)
    ap.add_argument("--margin-cache", type=Path, default=None)
    ap.add_argument("--family", choices=("circulant", "projected"),
                    default="circulant",
                    help="which cycle family. `projected` is the independent "
                         "robustness construction: a random skew perturbation "
                         "projected to annihilate both mu and a witness policy, so "
                         "its alpha-invariance comes from a projection rather than "
                         "from a number-theoretic property of circulant "
                         "tournaments. It needs no margin bisection because rho* is "
                         "already near-constant, so the measured value is reported "
                         "as-is and the normalized metric absorbs the drift.")
    ap.add_argument("--label", default=None,
                    help="suffix for the output files, so families and beta values "
                         "do not overwrite each other")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.margin_cache or (args.out_dir / "v2_margins.json")
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    # The cache key carries the construction, not just the seed. Margins bisected
    # at one pool geometry or amplitude are wrong at another, and a seed-only key
    # would reuse them silently -- which would move rho* off its target without
    # any visible error.
    geom = (f"K{args.objectives}_X{args.prompts}_I{args.responses}"
            f"_b{args.beta}_base{args.base_amp}_cyc{args.cycle_amp}"
            f"_tgt{args.target_rho_star}")

    rows = []
    for seed in args.seeds:
        if args.family == "projected":
            fam, _ = build_v2_projected(
                seed, K=args.objectives, X=args.prompts, I=args.responses,
                alphas=ALPHAS, m=args.margin, base_amp=args.base_amp,
                cycle_amp=args.cycle_amp, beta=args.beta)
            print(f"[seed {seed}] projected-skew family, no margin bisection",
                  flush=True)
            for alpha in ALPHAS:
                t0 = time.time()
                got = run_instance(fam[alpha], seed, alpha, args)
                rows.extend(got)
                _report(got, seed, alpha, time.time() - t0, args)
            continue
        key = f"{geom}|seed{seed}"
        if key in cache:
            margins = {float(a): m for a, m in cache[key].items()}
            fam, _ = {}, None
            fam = {}
            for a in ALPHAS:
                f, _ = build_v2(seed, K=args.objectives, X=args.prompts,
                                I=args.responses, alphas=(a,), m=margins[a],
                                base_amp=args.base_amp, cycle_amp=args.cycle_amp,
                                beta=args.beta)
                fam[a] = f[a]
                fam[a]["margin"] = margins[a]
            print(f"[seed {seed}] margins from cache", flush=True)
        else:
            fam, cfg = match_rho_star(
                seed, args.target_rho_star, ALPHAS, K=args.objectives,
                X=args.prompts, I=args.responses, base_amp=args.base_amp,
                cycle_amp=args.cycle_amp, beta=args.beta)
            cache[key] = {str(a): fam[a]["margin"] for a in ALPHAS}
            cache_path.write_text(json.dumps(cache, indent=2))
            print(f"[seed {seed}] margins bisected and cached", flush=True)
        for alpha in ALPHAS:
            t0 = time.time()
            got = run_instance(fam[alpha], seed, alpha, args)
            rows.extend(got)
            _report(got, seed, alpha, time.time() - t0, args)

    tag = args.label or (args.family if args.family != "circulant" else "")
    (args.out_dir / (f"controlled_v2_raw{'_' + tag if tag else ''}.json")).write_text(
        json.dumps({"config": {k: str(v) for k, v in vars(args).items()},
                    "alphas": list(ALPHAS), "rows": rows}, indent=2, default=str))
    write_outputs(rows, args.out_dir, tag=tag)


def write_outputs(rows, out_dir: Path, tag: str = ""):
    import csv
    sfx = f"_{tag}" if tag else ""
    per_seed = out_dir / f"controlled_v2_per_seed{sfx}.csv"
    cols = ["seed", "alpha", "method", "status", "converged", "bt_deviance_per_edge",
            "rho_star", "rho_star_gap", "min_surplus", "avg_surplus",
            "normalized_min_surplus", "nash_welfare", "nash_welfare_defined",
            "exploitability", "kl_from_reference", "raw_multiplier_l1",
            "target_log_ratio_l2", "fixed_point_residual", "extra_map_residual",
            "inverse_surplus_residual", "projected_kkt_residual",
            "target_identity_residual", "outer_iterations_used",
            "expected_non_convergent",
            "tv_to_exact_proximal", "tv_to_exact_global", "wall_clock_seconds",
            "peak_rss_mb", "box_bound_active", "non_converged_reasons"]
    with per_seed.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            out = {c: cell(r, c) for c in cols}
            out["non_converged_reasons"] = "; ".join(r.get("non_converged_reasons") or [])
            w.writerow(out)
    (per_seed.parent / "controlled_v2_applicability.json").write_text(json.dumps({
        "note": ("A blank was previously ambiguous between 'this quantity does not "
                 "exist for this method' and 'the solve failed'. Cells now read NA, "
                 "undefined, or a number."),
        "NA_means": "the quantity is mathematically inapplicable to this method",
        "undefined_means": ("the quantity applies but is undefined on this instance; "
                            "only nash_welfare, where some surplus is nonpositive"),
        "methods_without_a_solver": sorted(NO_SOLVER_METHODS),
        "methods_without_a_nash_dual": sorted(NO_DUAL_METHODS),
        "solver_diagnostic_columns": list(SOLVER_DIAGNOSTIC_COLUMNS),
        "dual_only_columns": list(DUAL_ONLY_COLUMNS),
    }, indent=2))

    agg = {}
    for r in rows:
        agg.setdefault((r["alpha"], r["method"]), []).append(r)
    m = lambda v: (sum(v) / len(v)) if v else None
    sd = lambda v: statistics.stdev(v) if len(v) > 1 else 0.0
    lines = ["alpha,method,n_seeds,n_converged,n_used,expected_non_convergent,bt_deviance,rho_star,"
             "min_surplus_mean,min_surplus_std,normalized_min_surplus_mean,"
             "normalized_min_surplus_std,avg_surplus_mean,nash_welfare_mean,"
             "exploitability_mean,kl_from_reference_mean,raw_multiplier_l1_mean,"
             "tv_to_exact_proximal_mean,tv_to_exact_global_mean,"
             "max_projected_kkt,max_inner_fixed_point"]
    for (alpha, meth), rs in sorted(agg.items(), key=lambda kv: (kv[0][0],
                                                                METHODS.index(kv[0][1]))):
        strict = [r for r in rs if r.get("status") == "ok" and r.get("converged")]
        expected_nc = any(r.get("expected_non_convergent") for r in rs)
        # The legacy R-step arm is USED despite never converging, because its
        # non-convergence is the ablation's result. Reporting that as
        # "n_converged" would tell a reader it converged, so the two counts are
        # separate columns and the expectation is flagged.
        ok = [r for r in rs if r.get("status") == "ok"
              and (r.get("converged") or r.get("expected_non_convergent"))]
        g = lambda k, src=None: [x[k] for x in (src or ok)
                                 if x.get(k) is not None]
        f = lambda v: "" if v is None else v
        lines.append(",".join(map(str, [
            alpha, meth, len(rs), len(strict), len(ok), expected_nc,
            f(m(g("bt_deviance_per_edge", rs))),
            f(m(g("rho_star", rs))), f(m(g("min_surplus"))), f(sd(g("min_surplus"))),
            f(m(g("normalized_min_surplus"))), f(sd(g("normalized_min_surplus"))),
            f(m(g("avg_surplus"))), f(m(g("nash_welfare"))),
            f(m(g("exploitability"))), f(m(g("kl_from_reference"))),
            f(m(g("raw_multiplier_l1"))), f(m(g("tv_to_exact_proximal"))),
            f(m(g("tv_to_exact_global"))),
            f(max(g("projected_kkt_residual"), default=None)),
            f(max(g("fixed_point_residual"), default=None))])))
    (out_dir / f"controlled_v2_summary{sfx}.csv").write_text("\n".join(lines) + "\n")

    rt = ["alpha,method,n,wall_clock_mean_s,wall_clock_max_s,peak_rss_mb_max,"
          "outer_iterations_mean"]
    for (alpha, meth), rs in sorted(agg.items(), key=lambda kv: (kv[0][0],
                                                                METHODS.index(kv[0][1]))):
        w = [r["wall_clock_seconds"] for r in rs if r.get("wall_clock_seconds") is not None]
        p = [r["peak_rss_mb"] for r in rs if r.get("peak_rss_mb") is not None]
        it = [r["outer_iterations_used"] for r in rs if r.get("outer_iterations_used")]
        rt.append(",".join(map(str, [alpha, meth, len(rs),
                                     (sum(w)/len(w)) if w else "",
                                     max(w) if w else "", max(p) if p else "",
                                     (sum(it)/len(it)) if it else ""])))
    (out_dir / f"controlled_v2_runtime{sfx}.csv").write_text("\n".join(rt) + "\n")
    print(f"\nwrote {out_dir}/controlled_v2_{{per_seed,summary,runtime}}{sfx}.csv")


def regenerate() -> None:
    """Rebuild the CSVs from `controlled_v2_raw.json` without re-solving anything.

    A bug in how results are *reported* must never cost hours of solver time to
    correct, so the raw per-cell record is the source of truth and the tables are
    derived from it.
    """
    ap = argparse.ArgumentParser(description=regenerate.__doc__)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    raw = json.loads((a.out_dir / "controlled_v2_raw.json").read_text())
    write_outputs(raw["rows"], a.out_dir)


if __name__ == "__main__":
    import sys
    if "--regenerate" in sys.argv:
        sys.argv.remove("--regenerate")
        regenerate()
    else:
        main()
