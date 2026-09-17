"""The controlled 2x2 for tab:factorial_game -- cyclicity x objective conflict.

Construction, reusing the validated v2 primitives:

    A_k = T(m) + base_k(base_amp) + alpha * C(cycle_amp)

T is the single direction every objective agrees on, base_k is the
per-objective component that makes objectives disagree, and C is a circulant
tournament on an odd number of responses whose row sums are exactly zero, so
the disagreement point d does not move with alpha.

Two corrections to the first design, both forced by measurement rather than
taste:

1. C is drawn ONCE PER PROMPT and shared by every objective. The upstream
   `circulation` helper draws an independent cyclic permutation per objective,
   and with that the smoke run measured cross-objective disagreement
   D = 0.4625 in BOTH high-cycle cells -- the same value at base_amp 0.02 and
   0.14 -- because independent cycles disagree across objectives by
   themselves. Cyclicity was driving the conflict axis, so the design was not a
   2x2. A shared cycle adds cyclicity to every objective identically and
   leaves D to the base component. The per-objective variant is kept behind
   --cycle-per-objective and its measured contamination is recorded.

2. The matched joint-feasibility margin is chosen by a pre-scan instead of
   being declared blind. The smoke run asked for rho* = 0.030 and the
   low-cycle/high-conflict cell returned 0.0611, because even the smallest
   admissible shared-direction weight leaves that cell above the target: the
   cell was unmatched while reporting a matched design. The scan measures the
   attainable rho* interval per cell and takes a target inside the
   intersection; if the intersection is empty the run says so and reports the
   achieved margin per cell instead of pretending they match. The scan uses
   only the construction, never a method ranking.

Every policy is then scored under the SAME adaptive-game evaluator, so no row
is in method-native units, and instances that miss any convergence criterion
are counted and excluded rather than quoted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from mnpo_scripts.nbpo_core import uniform_policy
from mnpo_scripts.nbpo_generic import matched_weight_l1, solve_finite_pool
from mnpo_scripts.nbpo_representations import (
    AdaptiveGameRepresentation, FixedReferenceRepresentation,
)
from scripts.experiments.iclr2027_table1_v2 import nontransitivity_feasibility as nf
from scripts.experiments.iclr2027_table1_v2.controlled_nontransitivity import (
    base_tensor, circulation, skew,
)
from scripts.experiments.iclr2027_table1_v2.nontransitivity_v2 import shared_direction

CONTRACT = {
    "contract_id": "sub_20260914_factorial_v2",
    "objectives_K": 2,
    "prompts_X": 40,
    "responses_I": 5,
    "beta": 0.25,
    "eta": 1.0,
    "cycle_amp": 0.28,
    "base_amp_low": 0.02,
    "base_amp_high": 0.14,
    "alpha_low": 0.0,
    "alpha_high": 1.0,
    "shared_cycle": True,
    "instance_seeds": list(range(20)),
    "cells": {"ll": ("low", "low"), "lh": ("low", "high"),
              "hl": ("high", "low"), "hh": ("high", "high")},
    "factor_checks": {
        "C": ("mean over (prompt, objective) of the fraction of the C(I,3)=10 distinct "
              "response triples carrying a directed 3-cycle under sign(A_k[i,j]); the "
              "synthetic payoffs are continuous, so the tie margin is exactly 0"),
        "D": ("mean over (prompt, unordered response pair) of the indicator that the two "
              "objectives order the pair oppositely, sign(A_0) != sign(A_1)")},
    "outcome_definitions": {
        "gamma_star": "achieved max_p min_k [V_k(p) - d_k] from nf.solve_max_min",
        "normalized_worst_surplus": "min_k [V_k(pi) - d_k] / gamma_star",
        "Delta_M": "normalized worst surplus of NBPO minus that of fixed-reference Nash",
        "target_TV": "total variation between the NBPO and game-utilitarian target policies"},
    "convergence_tolerances": {"projected_kkt": 1e-6, "inverse_surplus": 1e-6,
                               "inner_fixed_point": 1e-4},
    "uncertainty": "mean and SD across instance seeds; seed variation only, no policy seeds",
}
TOL = CONTRACT["convergence_tolerances"]
CELLS = {}


def set_levels(base_low, base_high, alpha_low, alpha_high):
    """The 2x2 factor levels. Amplitudes are a construction choice, and the
    scan below is what decides whether a shared feasibility margin exists at
    them; no method outcome enters the choice."""
    CELLS.clear()
    CELLS.update({
        "ll": {"alpha": alpha_low, "base_amp": base_low},
        "lh": {"alpha": alpha_low, "base_amp": base_high},
        "hl": {"alpha": alpha_high, "base_amp": base_low},
        "hh": {"alpha": alpha_high, "base_amp": base_high}})


set_levels(CONTRACT["base_amp_low"], CONTRACT["base_amp_high"],
           CONTRACT["alpha_low"], CONTRACT["alpha_high"])


def digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True,
                                     default=str).encode("utf-8")).hexdigest()


# ------------------------------------------------------------- construction
def build(seed, alpha, base_amp, m, cycle_amp, K, X, I, beta, shared_cycle=True):
    """One instance. Skewness, diagonal, payoff cap and d-invariance asserted."""
    if I % 2 == 0:
        raise ValueError("I must be odd so the circulant tournament has zero row sums")
    rng = np.random.default_rng(20260914 + seed)
    T, _ = shared_direction(rng, K, X, I, m)
    base_raw, _ = base_tensor(rng, K, X, I)
    base = base_raw * (base_amp / 0.45)
    C = circulation(rng, K, X, I) * (cycle_amp / 0.45)
    if shared_cycle:
        # one cyclic structure per prompt, identical for every objective, so the
        # cycle factor cannot move the cross-objective disagreement factor
        C = np.broadcast_to(C[0:1], C.shape).copy()
    rs = float(np.abs(C.sum(axis=-1)).max())
    if rs > 1e-12:
        raise AssertionError("circulation row sums are %.3e, not zero" % rs)

    mu = np.full((X, I), 1.0 / I)
    b = np.full(K, float(beta))
    A0 = skew(T + base)
    A = skew(T + base + alpha * C)
    if np.abs(A).max() > 0.5 + 1e-12:
        raise AssertionError("|A| reaches %.4f without clipping allowed" % np.abs(A).max())
    d0 = nf.game_values(A0, mu, mu, b)
    d = nf.game_values(A, mu, mu, b)
    drift = float(np.abs(d - d0).max())
    if drift > 1e-12:
        raise AssertionError("disagreement point moved by %.3e with the cycle" % drift)
    idx = np.arange(I)
    assert np.abs(A + np.swapaxes(A, -1, -2)).max() < 1e-12, "not skew-symmetric"
    assert np.abs(A[..., idx, idx]).max() == 0.0, "nonzero diagonal"
    return {"A": A, "mu": mu, "beta": b, "d": d, "cycle_row_sum": rs, "d_drift": drift}


def rho_of(seed, alpha, base_amp, m, args):
    inst = build(seed, alpha, base_amp, m, args.cycle_amp, args.objectives,
                 args.prompts, args.responses, args.beta,
                 shared_cycle=not args.cycle_per_objective)
    r = nf.solve_max_min(inst["A"], inst["mu"], inst["beta"], inst["d"])["rho_star"]
    return float(r), inst


def match_margin(seed, alpha, base_amp, target, args):
    """Bisect the shared-direction weight m so rho* hits the declared target."""
    m_cap = 0.5 - base_amp - args.cycle_amp * (1.0 if alpha > 0 else 0.0)
    lo, hi = 0.0, max(m_cap, 0.0)
    if hi <= 0.0:
        raise ValueError("no admissible m range")
    for _ in range(args.bisection_iterations):
        mid = 0.5 * (lo + hi)
        r, _ = rho_of(seed, alpha, base_amp, mid, args)
        if r < target:
            lo = mid
        else:
            hi = mid
    m = 0.5 * (lo + hi)
    r, inst = rho_of(seed, alpha, base_amp, m, args)
    inst["margin_m"] = m
    inst["rho_star"] = r
    inst["margin_at_cap"] = bool(m >= m_cap - 1e-6)
    inst["margin_at_floor"] = bool(m <= 1e-6)
    return inst


# ------------------------------------------------------------ factor checks
def cycle_fraction(A: np.ndarray) -> float:
    K, X, I, _ = A.shape
    triples = list(combinations(range(I), 3))
    hits = 0
    for k in range(K):
        for x in range(X):
            s = np.sign(A[k, x])
            for i, j, l in triples:
                if (s[i, j] > 0 and s[j, l] > 0 and s[l, i] > 0) or \
                   (s[j, i] > 0 and s[l, j] > 0 and s[i, l] > 0):
                    hits += 1
    return hits / max(K * X * len(triples), 1)


def disagreement_fraction(A: np.ndarray) -> float:
    K, X, I, _ = A.shape
    assert K == 2, "the disagreement check is defined for the two-objective contract"
    iu = np.triu_indices(I, k=1)
    s0 = np.sign(A[0][:, iu[0], iu[1]])
    s1 = np.sign(A[1][:, iu[0], iu[1]])
    return float(np.mean(s0 * s1 < 0))


# ------------------------------------------------------------------ solving
def non_convergence(sol) -> list:
    reasons = []
    pk = getattr(sol, "projected_kkt_residual", None)
    inv = getattr(sol, "kkt_residual", None)
    fp = getattr(sol, "extra_map_residual", None)
    box = bool(getattr(sol, "lambda_at_lower_bound", []) or
               getattr(sol, "lambda_at_upper_bound", []))
    if pk is not None and pk >= TOL["projected_kkt"]:
        reasons.append("projected_kkt %.2e" % pk)
    if inv is not None and not box and inv >= TOL["inverse_surplus"]:
        reasons.append("inverse_surplus %.2e with no active box bound" % inv)
    if fp is not None and fp >= TOL["inner_fixed_point"]:
        reasons.append("inner_extra_map %.2e" % fp)
    if getattr(sol, "dual_converged", None) is False:
        reasons.append("dual solver hit its evaluation cap")
    return reasons


def run_one(task):
    cell, seed, target, args = task
    spec = CELLS[cell]
    t0 = time.time()
    try:
        inst = match_margin(seed, spec["alpha"], spec["base_amp"], target, args)
    except Exception as exc:                                      # noqa: BLE001
        return {"cell": cell, "seed": seed, "usable": False,
                "reason": "construction failed: %s" % str(exc)[:200]}
    A, mu, beta, d = inst["A"], inst["mu"], inst["beta"], inst["d"]
    gamma = inst["rho_star"]
    if gamma <= 1e-8:
        return {"cell": cell, "seed": seed, "usable": False,
                "reason": "outside Assumption 1: rho*=%.3e" % gamma}

    At = torch.from_numpy(A)
    mu_t = uniform_policy(A.shape[1], A.shape[2])
    bt = torch.full((A.shape[0],), float(beta[0]), dtype=torch.float64)
    adaptive = AdaptiveGameRepresentation(At, At, mu_t, bt)
    fixed = FixedReferenceRepresentation(At, At, mu_t)

    sols = {}
    sols["nbpo"] = solve_finite_pool(adaptive, "nash", eta=args.eta, M=800, R=1,
                                     inner_solver="exact", dual_solver="root",
                                     dual_tol=1e-6)
    sols["fixed_reference"] = solve_finite_pool(fixed, "nash", eta=args.eta, M=800, R=1,
                                                inner_solver="exact", dual_solver="root",
                                                dual_tol=1e-6)
    l1 = matched_weight_l1(sols["nbpo"])
    sols["game_utilitarian"] = solve_finite_pool(adaptive, "utilitarian", eta=args.eta,
                                                 R=1, weight_l1=l1, inner_solver="exact")
    failures = {n: r for n, r in ((n, non_convergence(s)) for n, s in sols.items()) if r}

    pis = {n: s.pi.numpy() for n, s in sols.items()}
    worst = {}
    for name, pi in pis.items():
        s = nf.game_values(A, pi, mu, beta) - d
        worst[name] = {"surplus": s.tolist(), "min_surplus": float(s.min()),
                       "normalized_worst": float(s.min()) / gamma}
    return {
        "cell": cell, "seed": seed, "usable": not failures, "non_converged": failures,
        "alpha": spec["alpha"], "base_amp": spec["base_amp"],
        "margin_m": inst["margin_m"], "margin_at_cap": inst["margin_at_cap"],
        "margin_at_floor": inst["margin_at_floor"],
        "cycle_row_sum": inst["cycle_row_sum"], "d_drift": inst["d_drift"],
        "C": cycle_fraction(A), "D": disagreement_fraction(A), "gamma_star": gamma,
        "worst": worst,
        "Delta_M": worst["nbpo"]["normalized_worst"]
                   - worst["fixed_reference"]["normalized_worst"],
        "Delta_M_util": worst["nbpo"]["normalized_worst"]
                        - worst["game_utilitarian"]["normalized_worst"],
        "target_TV": float(nf.total_variation(pis["nbpo"], pis["game_utilitarian"])),
        "tv_nbpo_fixed": float(nf.total_variation(pis["nbpo"], pis["fixed_reference"])),
        "matched_weight_l1": float(l1), "seconds": time.time() - t0}


def summarize(values):
    a = np.asarray([v for v in values if v is not None], dtype=float)
    if a.size == 0:
        return {"n": 0, "mean": None, "sd": None, "ci95": None}
    n = a.size
    sd = float(a.std(ddof=1)) if n > 1 else 0.0
    half = 1.96 * sd / np.sqrt(n) if n > 1 else 0.0
    return {"n": int(n), "mean": float(a.mean()), "sd": sd,
            "ci95": [float(a.mean() - half), float(a.mean() + half)]}


def scan(args):
    """Attainable rho* interval per cell, from the construction alone."""
    seeds = args.scan_seeds
    table = {}
    for cell, spec in CELLS.items():
        m_cap = 0.5 - spec["base_amp"] - args.cycle_amp * (1.0 if spec["alpha"] > 0 else 0.0)
        floors, ceilings = [], []
        for s in seeds:
            floors.append(rho_of(s, spec["alpha"], spec["base_amp"], 0.0, args)[0])
            ceilings.append(rho_of(s, spec["alpha"], spec["base_amp"], m_cap, args)[0])
        table[cell] = {"m_cap": m_cap,
                       "rho_at_m0": summarize(floors), "rho_at_cap": summarize(ceilings),
                       "floor_max": max(floors), "ceiling_min": min(ceilings)}
    lower = max(v["floor_max"] for v in table.values())
    upper = min(v["ceiling_min"] for v in table.values())
    feasible = lower <= upper
    target = 0.5 * (lower + upper) if feasible else None
    return {"per_cell": table, "intersection": [lower, upper], "feasible": feasible,
            "target_rho_star": target,
            "note": ("the matched target must be attainable in every cell at some m in "
                     "[0, m_cap]; without an intersection the cells cannot share a margin "
                     "and the achieved gamma* is reported per cell instead")}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="/work/sub_20260914/factorial")
    ap.add_argument("--workers", type=int, default=min(20, len(os.sched_getaffinity(0))))
    ap.add_argument("--seeds", type=int, nargs="+", default=CONTRACT["instance_seeds"])
    ap.add_argument("--scan-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--objectives", type=int, default=CONTRACT["objectives_K"])
    ap.add_argument("--prompts", type=int, default=CONTRACT["prompts_X"])
    ap.add_argument("--responses", type=int, default=CONTRACT["responses_I"])
    ap.add_argument("--beta", type=float, default=CONTRACT["beta"])
    ap.add_argument("--eta", type=float, default=CONTRACT["eta"])
    ap.add_argument("--cycle-amp", type=float, default=CONTRACT["cycle_amp"])
    ap.add_argument("--bisection-iterations", type=int, default=18)
    ap.add_argument("--cycle-per-objective", action="store_true",
                    help="the contaminated variant kept for the record")
    ap.add_argument("--target-rho-star", type=float, default=None,
                    help="skip the scan and force a target")
    ap.add_argument("--base-amp-low", type=float, default=CONTRACT["base_amp_low"])
    ap.add_argument("--base-amp-high", type=float, default=CONTRACT["base_amp_high"])
    ap.add_argument("--grid-scan", action="store_true",
                    help="report the attainable margin intersection over an amplitude grid")
    ap.add_argument("--grid-base-high", type=float, nargs="+",
                    default=[0.05, 0.07, 0.09, 0.12])
    ap.add_argument("--grid-cycle-amp", type=float, nargs="+",
                    default=[0.16, 0.22, 0.28])
    args = ap.parse_args()
    set_levels(args.base_amp_low, args.base_amp_high, CONTRACT["alpha_low"],
               CONTRACT["alpha_high"])
    CONTRACT["base_amp_low"] = args.base_amp_low
    CONTRACT["base_amp_high"] = args.base_amp_high
    CONTRACT["cycle_amp"] = args.cycle_amp

    if args.grid_scan:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        grid = []
        for base_high in args.grid_base_high:
            for cyc in args.grid_cycle_amp:
                args.cycle_amp = cyc
                set_levels(args.base_amp_low, base_high, CONTRACT["alpha_low"],
                           CONTRACT["alpha_high"])
                try:
                    rep = scan(args)
                except Exception as exc:                          # noqa: BLE001
                    grid.append({"base_amp_high": base_high, "cycle_amp": cyc,
                                 "error": str(exc)[:160]})
                    continue
                lo, hi = rep["intersection"]
                row = {"base_amp_high": base_high, "cycle_amp": cyc,
                       "intersection": [lo, hi], "feasible": rep["feasible"],
                       "width": hi - lo, "target": rep["target_rho_star"],
                       "per_cell_cap": {c: v["m_cap"] for c, v in rep["per_cell"].items()}}
                grid.append(row)
                print(json.dumps({"grid": row}, default=float), flush=True)
        (out / "design_grid.json").write_text(json.dumps(grid, indent=1, default=float) + "\n")
        ok = [r for r in grid if r.get("feasible")]
        print(json.dumps({"phase": "grid_done", "feasible_configs": len(ok),
                          "widest": max(ok, key=lambda r: r["width"]) if ok else None},
                         default=float), flush=True)
        return 0 if ok else 1

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    if args.target_rho_star is not None:
        scan_report = {"skipped": "target forced", "target_rho_star": args.target_rho_star}
        target = args.target_rho_star
    else:
        scan_report = scan(args)
        target = scan_report["target_rho_star"]
        print(json.dumps({"phase": "scan", **{k: scan_report[k]
                                              for k in ("intersection", "feasible",
                                                        "target_rho_star")}}), flush=True)
        if not scan_report["feasible"]:
            (out / "scan.json").write_text(json.dumps(scan_report, indent=1) + "\n")
            print(json.dumps({"phase": "abort", "reason": "no shared margin exists"}))
            return 1
    (out / "scan.json").write_text(json.dumps(scan_report, indent=1) + "\n")

    tasks = [(c, s, target, args) for c in ("ll", "lh", "hl", "hh") for s in args.seeds]
    print(json.dumps({"phase": "start", "tasks": len(tasks), "workers": args.workers,
                      "target_rho_star": target,
                      "contract_sha256": digest(CONTRACT)}), flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(run_one, tasks):
            rows.append(row)
            print(json.dumps({"done": "%s/%s" % (row["cell"], row["seed"]),
                              "usable": row.get("usable"),
                              "C": round(row.get("C", -1), 4), "D": round(row.get("D", -1), 4),
                              "gamma": round(row.get("gamma_star", -1), 5),
                              "dM": None if row.get("Delta_M") is None
                                    else round(row["Delta_M"], 5)}), flush=True)

    cells = {}
    for cell in ("ll", "lh", "hl", "hh"):
        cr = [r for r in rows if r["cell"] == cell]
        ok = [r for r in cr if r.get("usable")]
        cells[cell] = {
            "spec": {**CELLS[cell], "cycle": CONTRACT["cells"][cell][0],
                     "conflict": CONTRACT["cells"][cell][1]},
            "instances_attempted": len(cr), "instances_usable": len(ok),
            "excluded": [{"seed": r["seed"], "reason": r.get("reason"),
                          "non_converged": r.get("non_converged")}
                         for r in cr if not r.get("usable")],
            "C": summarize([r.get("C") for r in ok]),
            "D": summarize([r.get("D") for r in ok]),
            "gamma_star": summarize([r.get("gamma_star") for r in ok]),
            "Delta_M": summarize([r.get("Delta_M") for r in ok]),
            "Delta_M_util": summarize([r.get("Delta_M_util") for r in ok]),
            "target_TV": summarize([r.get("target_TV") for r in ok]),
            "margin_m": summarize([r.get("margin_m") for r in ok]),
            "margin_at_cap": sum(1 for r in ok if r.get("margin_at_cap")),
            "margin_at_floor": sum(1 for r in ok if r.get("margin_at_floor")),
            "nbpo_normalized_worst": summarize(
                [r["worst"]["nbpo"]["normalized_worst"] for r in ok]),
            "fixed_normalized_worst": summarize(
                [r["worst"]["fixed_reference"]["normalized_worst"] for r in ok]),
            "util_normalized_worst": summarize(
                [r["worst"]["game_utilitarian"]["normalized_worst"] for r in ok])}

    (out / "rows.json").write_text(json.dumps(rows, indent=1, default=float) + "\n")
    report = {"contract": CONTRACT, "contract_sha256": digest(CONTRACT),
              "shared_cycle": not args.cycle_per_objective,
              "target_rho_star": target, "scan": scan_report,
              "seconds": time.time() - started, "cells": cells,
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out / "summary.json").write_text(json.dumps(report, indent=1, default=float) + "\n")
    print(json.dumps({"phase": "done", "seconds": report["seconds"],
                      "cells": {c: {"usable": v["instances_usable"],
                                    "C": v["C"]["mean"], "D": v["D"]["mean"],
                                    "gamma": v["gamma_star"]["mean"],
                                    "dM": v["Delta_M"]["mean"],
                                    "TV": v["target_TV"]["mean"]}
                                for c, v in cells.items()}}, default=float), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
