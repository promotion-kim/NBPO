#!/usr/bin/env python3
"""Audit the controlled-nontransitivity stress test before reading it as a result.

The v1 sweep reports that NBPO loses to a fixed-reference control, and to the
uniform reference itself, as intransitivity rises. That is only a statement about
NBPO if the instances are inside Assumption 1 -- i.e. if some policy improves
*every* objective over the reference. This tool establishes that first, exactly,
and only then compares solvers.

Per instance it reports
  * ``rho* = max_pi min_k s_k(pi)`` from a direct float64 primal solver with a
    concavity certificate, not from the alternating fixed point under audit;
  * the disagreement point, the objective-wise maximum attainable surplus, and
    the individually-rational ideal point that decides whether Game-KS is defined
    at all;
  * whether the reference is already the max-min optimum and whether it is an
    equilibrium of the underlying game;
and then four policies scored on one common yardstick:
  A  exact global Nash          max_pi sum_k log s_k
  B  exact proximal Nash        max_pi sum_k log s_k - D(pi||pi_ref)/eta
  C  practical alternating      the deployed dual/fixed-point solver
  D  matched-step diagnostic    C's weight direction at a common ||w||_1

An instance with ``rho* <= 0`` is marked ``outside_assumption_1`` and is excluded
from every method comparison rather than being reported as a method failure.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import torch

from mnpo_scripts.nbpo_core import uniform_policy
from mnpo_scripts.nbpo_generic import (
    KSUndefinedError, matched_weight_l1, solve_finite_pool, solve_proximal,
)
from mnpo_scripts.nbpo_representations import (
    AdaptiveGameRepresentation, BTRewardRepresentation, FixedReferenceRepresentation,
)
from scripts.experiments.iclr2027_table1_v2 import nontransitivity_feasibility as nf
from scripts.experiments.iclr2027_table1_v2.controlled_nontransitivity import (
    base_tensor, bt_fit, circulation, mix,
)
from scripts.experiments.iclr2027_table1_v2.nontransitivity_v2 import (
    build_v2, match_rho_star,
)

ALPHAS = (0.00, 0.25, 0.50, 0.75, 1.00)
RHO_POSITIVE = 1e-8


# --------------------------------------------------------------------------
# instances
# --------------------------------------------------------------------------

def v1_instance(seed, alpha, K, X, I, beta):
    """The existing benchmark, rebuilt with the identical seeds and construction."""
    rng = np.random.default_rng(20260908 + seed)
    base, _ = base_tensor(rng, K, X, I)
    C = circulation(rng, K, X, I)
    A = mix(base, C, alpha)
    mu = np.full((X, I), 1.0 / I)
    b = np.full(K, float(beta))
    return {"A": A, "mu": mu, "beta": b, "d": nf.game_values(A, mu, mu, b)}


# --------------------------------------------------------------------------
# common evaluation -- every policy scored on the same true adaptive game
# --------------------------------------------------------------------------

def evaluate(A, mu, beta, d, pi, pi_ref, eta, refs=None):
    s = nf.game_values(A, pi, mu, beta) - d
    out = {
        "surplus": s.tolist(),
        "min_surplus": float(s.min()),
        "avg_surplus": float(s.mean()),
        "nash_welfare": float(np.log(s).sum()) if (s > 0).all() else None,
        "nash_welfare_defined": bool((s > 0).all()),
        "kl_from_reference": nf.mean_kl(pi, pi_ref),
        "exploitability": nf.exploitability(A, pi),
        # lambda_k = 1/s_k is the Nash stationarity condition; reported raw
        "implied_raw_lambda": (1.0 / s).tolist() if (s > 0).all() else None,
        "implied_raw_lambda_l1": float((1.0 / s).sum()) if (s > 0).all() else None,
        "target_log_ratio_l2": float(np.linalg.norm(
            np.log(np.clip(pi, nf.FLOOR, None)) - np.log(np.clip(pi_ref, nf.FLOOR, None)))),
        "target_log_ratio_mean_abs": float(np.mean(np.abs(
            np.log(np.clip(pi, nf.FLOOR, None)) - np.log(np.clip(pi_ref, nf.FLOOR, None))))),
    }
    for name, ref in (refs or {}).items():
        out[f"tv_to_{name}"] = nf.total_variation(pi, ref)
    return out


def practical_solutions(A, beta, eta, M, R, gamma=0.5):
    """C and D: the deployed alternating solver, and its matched-step diagnostic."""
    K, X, I, _ = A.shape
    At = torch.from_numpy(A)
    mu_t = uniform_policy(X, I)
    game = AdaptiveGameRepresentation(At, At, mu_t,
                                      torch.full((K,), float(beta[0]), dtype=torch.float64))
    fixed = FixedReferenceRepresentation(At, At, mu_t)
    # BT-RM: the best SCALAR summary of the same payoff, fitted per prompt. Its
    # Eq. (21) map is constant, so like fixed_reference it lands on its fixed
    # point in one application and cannot diverge -- which is exactly why it has
    # to be in the table next to a solver that can.
    r = np.zeros((K, X, I))
    for k in range(K):
        for x in range(X):
            _, sc = bt_fit(0.5 + A[k, x])
            r[k, x] = sc
    bt = BTRewardRepresentation(torch.from_numpy(r), torch.from_numpy(r), mu_t)

    nash_game = solve_finite_pool(game, "nash", eta=eta, M=M, R=R, gamma=gamma)
    nash_fixed = solve_finite_pool(fixed, "nash", eta=eta, M=M, R=R, gamma=gamma)
    nash_bt = solve_finite_pool(bt, "nash", eta=eta, M=M, R=R, gamma=gamma)
    L = matched_weight_l1(nash_fixed)
    w = nash_game.weights * (L / float(nash_game.weights.sum()))
    matched = solve_proximal(game, mu_t, w, eta, R)
    return {
        "C_practical": {
            "pi": nash_game.pi.numpy(),
            "weight_l1": float(nash_game.weights.sum()),
            "weights": [float(v) for v in nash_game.weights],
            "fixed_point_residual": nash_game.fixed_point_residual,
            "extra_map_residual": nash_game.extra_map_residual,
            "kkt_residual": nash_game.kkt_residual,
            "projected_kkt_residual": nash_game.projected_kkt_residual,
        },
        "D_matched_step": {
            "pi": matched.pi.numpy(),
            "weight_l1": float(w.sum()),
            "weights": [float(v) for v in w],
            "fixed_point_residual": matched.fixed_point_residual,
            "extra_map_residual": matched.extra_map_residual,
            "common_weight_l1": L,
        },
        "fixed_reference_native": {
            "pi": nash_fixed.pi.numpy(),
            "weight_l1": float(nash_fixed.weights.sum()),
            "fixed_point_residual": nash_fixed.fixed_point_residual,
        },
        "bt_rm_nash": {
            "pi": nash_bt.pi.numpy(),
            "weight_l1": float(nash_bt.weights.sum()),
            "fixed_point_residual": nash_bt.fixed_point_residual,
            "extra_map_residual": nash_bt.extra_map_residual,
        },
    }


def ks_definedness(A, mu, beta, d):
    """Is Game-KS defined here? Two independent readings, reported side by side."""
    ideal = nf.objectivewise_max_surplus(A, mu, beta, d, ir_constrained=True)
    u = np.array([e["max_surplus"] for e in ideal])
    unconstrained = nf.objectivewise_max_surplus(A, mu, beta, d, ir_constrained=False)
    block = {
        "ir_constrained_ideal": u.tolist(),
        "ir_constrained_ideal_min": float(u.min()),
        "objectivewise_max_surplus": [e["max_surplus"] for e in unconstrained],
        # A positive ideal produced only by float noise is the failure mode the
        # solver guards against, so the threshold is stated rather than implied.
        "ideal_is_numerically_meaningful": bool(u.min() > 1e-6),
    }
    K, X, I, _ = A.shape
    At = torch.from_numpy(A)
    rep = AdaptiveGameRepresentation(At, At, uniform_policy(X, I),
                                     torch.full((K,), float(beta[0]), dtype=torch.float64))
    try:
        sol = solve_finite_pool(rep, "kalai_smorodinsky", eta=1.0, R=3, weight_l1=1.0,
                                ks_kwargs=dict(stage1_iters=200, ideal_iters=30))
        block["solver_says_defined"] = True
        block["solver_rho_star"] = sol.ks.get("rho_star")
    except KSUndefinedError as exc:
        block["solver_says_defined"] = False
        block["solver_refusal"] = str(exc)[:300]
    return block


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def audit_instance(inst, seed, alpha, eta, M, R, skip_ks=False, c2_outer=120):
    A, mu, beta, d = inst["A"], inst["mu"], inst["beta"], inst["d"]
    K, X, I, _ = A.shape
    pi_ref = mu
    devs = [bt_fit(0.5 + A[k, x])[0] for k in range(K) for x in range(X)]

    feas = nf.solve_max_min(A, mu, beta, d)
    ref_block = nf.reference_is_equilibrium(A, mu, beta, d, pi_ref)
    row = {
        "seed": seed, "alpha": alpha,
        "bt_deviance_per_edge": float(np.mean(devs)),
        "disagreement": d.tolist(),
        "rho_star": feas["rho_star"],
        "rho_star_certified_upper_bound": feas["certified_upper_bound"],
        "rho_star_gap": feas["rho_star_gap"],
        "rho_star_surplus_at_argmax": feas["surplus"],
        "certificate_weights": feas["certificate_weights"],
        "primal_residual": feas["primal_feasibility_residual"],
        "complementary_slackness_residual": feas["complementary_slackness_residual"],
        "slsqp_status": feas["slsqp_status"],
        "feasible": bool(feas["rho_star"] > RHO_POSITIVE),
        "reference": ref_block,
    }
    if not skip_ks:
        row["kalai_smorodinsky"] = ks_definedness(A, mu, beta, d)

    prac = practical_solutions(A, beta, eta, M, R)
    policies = {"C_practical": prac["C_practical"]["pi"],
                "D_matched_step": prac["D_matched_step"]["pi"],
                "fixed_reference_native": prac["fixed_reference_native"]["pi"],
                "bt_rm_nash": prac["bt_rm_nash"]["pi"],
                "reference": pi_ref}

    # C2: the SAME outer dual loop with the inner proximal map replaced by a
    # direct concave maximization. This is the repaired practical solver, and it
    # is what Section 4's "fixed-point residual < 1e-4 and close to the exact
    # proximal policy" gate is actually asking for.
    pi_c2, c2 = nf.solve_nash_dual_exact_inner(A, mu, beta, d, pi_ref, eta,
                                               M=c2_outer, gamma=0.5)
    policies["C2_exact_inner_solve"] = pi_c2
    prac["C2_exact_inner_solve"] = {
        "weight_l1": c2["weight_l1"], "weights": c2["weights"],
        "fixed_point_residual": c2["stationarity_residual"],
        "extra_map_residual": c2["extra_map_residual"],
        "target_log_ratio_identity_residual": c2["target_log_ratio_identity_residual"],
        "kkt_residual": c2["kkt_residual"],
        "outer_iterations": c2["outer_iterations"]}

    if row["feasible"]:
        pi_a, res_a, s_a = nf.solve_nash(A, mu, beta, d, pi0=feas["pi"])
        pi_b, res_b, s_b = nf.solve_nash(A, mu, beta, d, pi0=feas["pi"],
                                         eta=eta, pi_ref=pi_ref)
        policies["A_exact_global_nash"] = pi_a
        policies["B_exact_proximal_nash"] = pi_b
        row["exact_status"] = {"global": int(res_a.status), "proximal": int(res_b.status)}
    else:
        row["outside_assumption_1"] = True
        row["exact_status"] = {"global": None, "proximal": None,
                               "reason": "rho* <= 0: the Nash program has no interior"}

    refs = {k: policies[k] for k in ("A_exact_global_nash", "B_exact_proximal_nash")
            if k in policies}
    row["policies"] = {}
    for name, pi in policies.items():
        ev = evaluate(A, mu, beta, d, pi, pi_ref, eta,
                      refs={f"exact_{k[0]}": v for k, v in refs.items()})
        for key in ("weight_l1", "weights", "fixed_point_residual",
                    "extra_map_residual", "kkt_residual", "projected_kkt_residual",
                    "common_weight_l1", "target_log_ratio_identity_residual"):
            if name in prac and key in prac[name]:
                ev[key] = prac[name][key]
        row["policies"][name] = ev
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", choices=("v1", "v2", "v2_matched"), default="v1")
    ap.add_argument("--v2-target-rho-star", type=float, default=0.030)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--objectives", type=int, default=4)
    ap.add_argument("--prompts", type=int, default=40)
    ap.add_argument("--responses", type=int, default=4)
    ap.add_argument("--beta", type=float, default=0.25)
    ap.add_argument("--eta", type=float, default=1.0)
    ap.add_argument("--dual-iterations", type=int, default=1500)
    ap.add_argument("--fixed-point-iterations", type=int, default=3)
    ap.add_argument("--v2-margin", type=float, default=0.16)
    ap.add_argument("--v2-base-amp", type=float, default=0.06)
    ap.add_argument("--v2-cycle-amp", type=float, default=0.28)
    ap.add_argument("--v2-responses", type=int, default=5)
    ap.add_argument("--skip-ks", action="store_true")
    ap.add_argument("--c2-outer-iterations", type=int, default=120)
    args = ap.parse_args()

    rows, v2_cfg = [], None
    for seed in range(args.seeds):
        if args.benchmark == "v2":
            fam, v2_cfg = build_v2(seed, K=args.objectives, X=args.prompts,
                                   I=args.v2_responses, alphas=ALPHAS,
                                   m=args.v2_margin, base_amp=args.v2_base_amp,
                                   cycle_amp=args.v2_cycle_amp, beta=args.beta)
        elif args.benchmark == "v2_matched":
            fam, v2_cfg = match_rho_star(
                seed, args.v2_target_rho_star, ALPHAS, K=args.objectives,
                X=args.prompts, I=args.v2_responses, base_amp=args.v2_base_amp,
                cycle_amp=args.v2_cycle_amp, beta=args.beta)
        for alpha in ALPHAS:
            inst = (fam[alpha] if args.benchmark.startswith("v2")
                    else v1_instance(seed, alpha, args.objectives, args.prompts,
                                     args.responses, args.beta))
            row_extra = {k: inst[k] for k in ("margin", "margin_at_cap")
                         if k in inst}
            rows.append(audit_instance(inst, seed, alpha, args.eta,
                                       args.dual_iterations,
                                       args.fixed_point_iterations,
                                       skip_ks=args.skip_ks,
                                       c2_outer=args.c2_outer_iterations))
            rows[-1].update(row_extra)
            r = rows[-1]
            pol = r["policies"]
            print(f"  [{args.benchmark}] seed {seed} alpha {alpha:.2f}  "
                  f"rho*={r['rho_star']:+.6f} (gap {r['rho_star_gap']:.1e})  "
                  f"BTdev={r['bt_deviance_per_edge']:.4f}  feasible={r['feasible']}  "
                  f"C={pol['C_practical']['min_surplus']:+.5f}  "
                  f"C2={pol['C2_exact_inner_solve']['min_surplus']:+.5f}  "
                  f"resid={pol['C2_exact_inner_solve'].get('fixed_point_residual', float('nan')):.1e}",
                  flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.benchmark
    (args.out_dir / f"nontransitivity_audit_{tag}.json").write_text(
        json.dumps({"config": vars(args) | {"out_dir": str(args.out_dir)},
                    "v2_config": v2_cfg, "rows": rows}, indent=2, default=str))
    write_tables(rows, args.out_dir, tag)


def write_tables(rows, out_dir, tag):
    by_alpha = {}
    for r in rows:
        by_alpha.setdefault(r["alpha"], []).append(r)

    m = lambda v: (sum(v) / len(v)) if v else float("nan")
    sd = lambda v: statistics.stdev(v) if len(v) > 1 else 0.0

    lines = ["alpha,n_seeds,bt_deviance,rho_star_mean,rho_star_std,rho_star_min,"
             "frac_rho_star_positive,max_rho_star_gap,ref_is_maxmin_optimal_frac,"
             "ref_exploitability,ks_defined_frac,ir_ideal_min_mean"]
    for alpha, rs in sorted(by_alpha.items()):
        rho = [r["rho_star"] for r in rs]
        ks = [r.get("kalai_smorodinsky", {}).get("ideal_is_numerically_meaningful")
              for r in rs]
        ks = [x for x in ks if x is not None]
        ideal = [r.get("kalai_smorodinsky", {}).get("ir_constrained_ideal_min")
                 for r in rs]
        ideal = [x for x in ideal if x is not None]
        lines.append(",".join(map(str, [
            alpha, len(rs), m([r["bt_deviance_per_edge"] for r in rs]),
            m(rho), sd(rho), min(rho),
            sum(1 for x in rho if x > RHO_POSITIVE) / len(rho),
            max(r["rho_star_gap"] for r in rs),
            m([1.0 if r["reference"]["reference_is_max_min_optimal"] else 0.0 for r in rs]),
            m([r["reference"]["exploitability"] for r in rs]),
            (m([1.0 if x else 0.0 for x in ks]) if ks else ""),
            (m(ideal) if ideal else "")])))
    (out_dir / f"feasibility_{tag}.csv").write_text("\n".join(lines) + "\n")

    methods = ["A_exact_global_nash", "B_exact_proximal_nash", "C_practical",
               "C2_exact_inner_solve", "D_matched_step", "fixed_reference_native",
               "bt_rm_nash", "reference"]
    head = ("alpha,method,n,min_surplus_mean,min_surplus_std,avg_surplus_mean,"
            "nash_welfare_mean,kl_from_reference_mean,exploitability_mean,"
            "weight_l1_mean,target_log_ratio_l2_mean,fixed_point_residual_max,"
            "tv_to_exact_A_mean,tv_to_exact_B_mean,rho_star_mean,"
            "min_surplus_over_rho_star_mean")
    out = [head]
    for alpha, rs in sorted(by_alpha.items()):
        rho = m([r["rho_star"] for r in rs])
        for meth in methods:
            got = [r["policies"][meth] for r in rs if meth in r["policies"]]
            if not got:
                continue
            g = lambda k: [x[k] for x in got if x.get(k) is not None]
            out.append(",".join(map(str, [
                alpha, meth, len(got), m(g("min_surplus")), sd(g("min_surplus")),
                m(g("avg_surplus")), (m(g("nash_welfare")) if g("nash_welfare") else ""),
                m(g("kl_from_reference")), m(g("exploitability")),
                (m(g("weight_l1")) if g("weight_l1") else ""),
                m(g("target_log_ratio_l2")),
                (max(g("fixed_point_residual")) if g("fixed_point_residual") else ""),
                (m(g("tv_to_exact_A")) if g("tv_to_exact_A") else ""),
                (m(g("tv_to_exact_B")) if g("tv_to_exact_B") else ""),
                rho,
                (m([x / rho for x in g("min_surplus")]) if rho > RHO_POSITIVE else "")])))
    (out_dir / f"solver_comparison_{tag}.csv").write_text("\n".join(out) + "\n")
    print(f"\nwrote {out_dir}/feasibility_{tag}.csv and solver_comparison_{tag}.csv")


if __name__ == "__main__":
    main()
