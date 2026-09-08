#!/usr/bin/env python3
"""Measured scaling of the direct finite-pool concave inner solve.

"Usable at 7000 prompts" is a claim about a measurement, so the 7000-prompt case
is run rather than extrapolated from smaller ones.

What is varied is only infrastructure: the number of prompts, the pool geometry,
and the number of worker processes. The mathematical program and every tolerance
are held fixed -- the per-prompt subproblem, the `dual_tol`, and the declared
inner stationarity requirement are identical at every size. A faster number
obtained by loosening either would not be a scaling result.

Tensors are SafeRLHF-shaped: two objectives (helpfulness, harmlessness) whose
disagreement rate on unordered pairs is calibrated to the 24.3% measured on the
released human annotations, centered preferences in [-1/2, 1/2], exactly
skew-symmetric with a zero diagonal.
"""
from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import torch

from mnpo_scripts.nbpo_core import uniform_policy
from mnpo_scripts.nbpo_generic import solve_finite_pool
from mnpo_scripts.nbpo_representations import AdaptiveGameRepresentation

TARGET_CONFLICT = 0.2433          # measured on the SafeRLHF held-out split


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def skew(M):
    M = 0.5 * (M - np.swapaxes(M, -1, -2))
    idx = np.arange(M.shape[-1])
    M[..., idx, idx] = 0.0
    return M


def saferlhf_shaped_tensor(rng, X, I, amp=0.35, rho=None):
    """Two objectives that disagree on about a quarter of pairs, as SafeRLHF does.

    Built from two latent utilities with a tuned correlation rather than from
    independent noise, so the conflict rate -- the quantity the bargaining
    problem actually depends on -- matches the real panel instead of being
    whatever independence happens to give.
    """
    if rho is None:
        # P(sign disagreement) for a bivariate normal difference is
        # arccos(rho)/pi, so invert it at the measured conflict rate.
        rho = float(np.cos(np.pi * TARGET_CONFLICT))
    z0 = rng.normal(size=(X, I))
    z1 = rho * z0 + np.sqrt(max(0.0, 1.0 - rho ** 2)) * rng.normal(size=(X, I))
    A = np.empty((2, X, I, I))
    for k, u in enumerate((z0, z1)):
        d = u[:, :, None] - u[:, None, :]
        A[k] = amp * np.tanh(0.75 * d)
    A = skew(A)
    if np.abs(A).max() > 0.5 + 1e-12:
        raise AssertionError("payoff out of range")
    return A


def measured_conflict(A):
    s = np.sign(A)
    iu = np.triu_indices(A.shape[-1], k=1)
    a, b = s[0][:, iu[0], iu[1]], s[1][:, iu[0], iu[1]]
    live = (a != 0) & (b != 0)
    return float((a[live] != b[live]).mean())


def run_case(X, I, workers, eta, dual_tol, max_dual_calls, seed, out_dir,
             write_artifact=True):
    rng = np.random.default_rng(20260908 + seed)
    A = saferlhf_shaped_tensor(rng, X, I)
    conflict = measured_conflict(A)
    At = torch.from_numpy(A)
    mu = uniform_policy(X, I)
    rep = AdaptiveGameRepresentation(At, At, mu,
                                     torch.full((2,), 0.25, dtype=torch.float64))
    rss0 = peak_rss_mb()
    t0 = time.time()
    sol = solve_finite_pool(rep, "nash", eta=eta, R=1, M=max_dual_calls,
                            inner_solver="exact", dual_solver="root",
                            dual_tol=dual_tol, inner_workers=workers)
    solve_s = time.time() - t0
    calls = sol.outer_iterations_used or 1

    art_s, art_bytes = None, None
    if write_artifact:
        from scripts.nbpo.solve_nbpo_dual import write_generic_solution_artifact
        d = out_dir / f"artifact_X{X}_I{I}"
        t1 = time.time()
        write_generic_solution_artifact(
            d, sol, {"reference_construction": "shared_pool"}, {}, d, 0,
            lambda_warm_started=False,
            extra={"scaling_probe": True, "prompts": X, "responses": I})
        art_s = time.time() - t1
        art_bytes = sum(f.stat().st_size for f in d.glob("*"))
        for f in d.glob("*"):
            f.unlink()
        d.rmdir()

    return {
        "prompts": X, "responses": I, "objectives": 2, "workers": workers,
        "pool_geometry": f"{I}+{I}",
        "measured_pairwise_conflict_rate": conflict,
        "total_solve_seconds": solve_s,
        "dual_evaluations": calls,
        "seconds_per_dual_evaluation": solve_s / max(1, calls),
        "dual_converged": bool(sol.dual_converged),
        "projected_kkt_residual": sol.projected_kkt_residual,
        "inverse_surplus_residual": sol.kkt_residual,
        "inner_extra_map_residual": sol.extra_map_residual,
        "inner_fixed_point_residual": sol.fixed_point_residual,
        "target_identity_residual": sol.target_log_ratio_check(),
        "weight_l1": float(sol.weights.sum()),
        "min_surplus": float(sol.surplus.min()),
        "peak_rss_mb": max(peak_rss_mb(), rss0),
        "tensor_mb": A.nbytes / 1e6,
        "artifact_write_seconds": art_s,
        "artifact_bytes": art_bytes,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path,
                    default=Path("results/iclr2027_table1_v2/solver_scaling"))
    ap.add_argument("--prompts", type=int, nargs="+", default=[200, 1000, 7000])
    ap.add_argument("--responses", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 32, 96])
    ap.add_argument("--eta", type=float, default=1.0)
    ap.add_argument("--dual-tol", type=float, default=1e-6)
    ap.add_argument("--max-dual-calls", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--serial-limit", type=int, default=1000,
                    help="skip the 1-worker case above this many prompts; it is "
                         "measured at smaller sizes and the point of the sweep is "
                         "the parallel path")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for X in args.prompts:
        for I in args.responses:
            for w in args.workers:
                if w == 1 and X > args.serial_limit:
                    continue
                t = time.time()
                r = run_case(X, I, w, args.eta, args.dual_tol, args.max_dual_calls,
                             args.seed, args.out_dir)
                rows.append(r)
                print(f"  X={X:>5} I={I:>2} workers={w:>3}  "
                      f"solve={r['total_solve_seconds']:7.1f}s  "
                      f"calls={r['dual_evaluations']:>3}  "
                      f"s/call={r['seconds_per_dual_evaluation']:6.3f}  "
                      f"conv={r['dual_converged']}  "
                      f"pkkt={r['projected_kkt_residual']:.1e}  "
                      f"inner={r['inner_extra_map_residual']:.1e}  "
                      f"rss={r['peak_rss_mb']:7.0f}MB  "
                      f"artifact={r['artifact_write_seconds']:.2f}s "
                      f"({time.time()-t:.0f}s wall)", flush=True)
                (args.out_dir / "direct_solver_scaling.json").write_text(
                    json.dumps({"config": {k: str(v) for k, v in vars(args).items()},
                                "rows": rows}, indent=2))
    write_outputs(rows, args.out_dir)


def write_outputs(rows, out_dir: Path):
    import csv
    cols = ["prompts", "responses", "pool_geometry", "objectives", "workers",
            "measured_pairwise_conflict_rate", "total_solve_seconds",
            "dual_evaluations", "seconds_per_dual_evaluation", "dual_converged",
            "projected_kkt_residual", "inverse_surplus_residual",
            "inner_extra_map_residual", "inner_fixed_point_residual",
            "target_identity_residual", "weight_l1", "min_surplus",
            "peak_rss_mb", "tensor_mb", "artifact_write_seconds", "artifact_bytes"]
    with (out_dir / "direct_solver_scaling.csv").open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            wr.writerow(r)

    best = {}
    for r in rows:
        k = (r["prompts"], r["responses"])
        if k not in best or r["total_solve_seconds"] < best[k]["total_solve_seconds"]:
            best[k] = r
    L = [r"\begin{tabular}{r r r r r r r r}", r"\toprule",
         r"prompts & pool & workers & solve (s) & dual evals & s/eval & "
         r"proj.\ KKT & peak RSS (MB) \\", r"\midrule"]
    for (X, I), r in sorted(best.items()):
        L.append(" & ".join([
            f"{X}", f"{I}+{I}", f"{r['workers']}",
            f"{r['total_solve_seconds']:.1f}", f"{r['dual_evaluations']}",
            f"{r['seconds_per_dual_evaluation']:.3f}",
            f"{r['projected_kkt_residual']:.0e}",
            f"{r['peak_rss_mb']:.0f}"]) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (out_dir / "direct_solver_scaling_generated.tex").write_text("\n".join(L) + "\n")
    print(f"\nwrote {out_dir}/direct_solver_scaling.{{csv,json}} and _generated.tex")


if __name__ == "__main__":
    main()
