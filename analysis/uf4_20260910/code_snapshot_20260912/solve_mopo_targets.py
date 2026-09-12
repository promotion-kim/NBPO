"""MOPO targets: the importance ratio rho(y) of Agnihotri et al., Eq. (3).

The trainer already implements MOPO's policy step -- `loss_type == "mopo"` is
importance-weighted behaviour cloning, `-rho(y) log pi(y)` -- so all that is
missing is rho. This script produces it from the same frozen teacher tensor and
candidate pool every other arm uses, so the feedback budget is identical by
construction.

Declarations, all taken from the paper (see
analysis/uf4_20260910/mopo_declarations_20260912.md):

* Primary objective is fixed positionally, as the paper fixes it: it writes p_K
  for "the K-th (the primary) objective" and takes r_1 as primary in every
  experiment. Applied to our pre-registered objective order that is instruction
  following, and the other three are constrained. It is NOT chosen by which
  choice would flatter the baseline.
* Lower bounds follow the paper's adaptive self-referential schedule
  b = beta * G(rho^(t-t0)) at beta = 0.9995. With the main row's t0 = infinity --
  one of the four lag settings the paper's own ablation reports as stable --
  there is no refresh, so b is its t = 0 value: beta times what the reference
  policy itself achieves, G(rho = 1).
* tau = 0.08 is the paper's Table 3 value, verbatim.
* epsilon is DERIVED, not copied, and this is the one place the paper's numbers
  cannot both be taken as given. The paper's epsilon = 0.15 is a KL radius on a
  full response distribution, where it is a small perturbation. Our adaptation
  compares eight pooled candidates, and a KL ball of radius 0.15 on eight atoms
  lets the adversary move about a third of the mass: measured on our own dev
  tensor, the robustification then eats 0.061 to 0.083 of each secondary value
  at the reference policy itself, while the paper's beta = 0.9995 allows a slack
  of 2.5e-4. The constraint is therefore violated before optimisation starts,
  the dual multiplier grows without bound (lambda reached 73.8 and was still
  rising linearly after 300 iterations), and rho collapses to one-hot -- which
  turns MOPO's policy step into best-of-8 cloning and is not MOPO.
  So epsilon is fixed by requiring the robustification to consume exactly the
  slack the paper's own beta allows at the reference policy:
  max_k [ G_k(1) - LowerBound_k(1) ] = (1 - beta) * min_k G_k(1).
  Both numbers stay the paper's; the one whose calibration depends on the size
  of the support is solved for instead of transplanted.

Two things this script does that the paper does and a naive reading would not:

* rho is normalised per prompt so that E_{y~pi_ref}[rho(y)] = 1. This is the
  paper's own Policy Extraction step, pi*(y) = pi_ref(y) rho*(y) / sum_y'
  pi_ref(y') rho*(y'); without it rho is not an importance ratio.
* the secondary values average over the seven OTHER candidates, not over all
  eight. A self-comparison is not a preference, and the campaign already
  excludes self elsewhere (`W_bank_min_excludes_self`). The teacher's diagonal
  is a real output rather than a placeholder, which is exactly why it has to be
  excluded deliberately instead of by accident.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/work/uf4_20260910/code")
import solve_uf4_targets as base

ROOT = Path("/work/uf4_20260910")
OBJECTIVES = base.OBJECTIVES
POOL = base.POOL


def candidate_values(A):
    """Wbar[k, x, i] = mean over j != i of the teacher's win probability of i over j."""
    K, X, I, J = A.shape
    if (I, J) != (POOL, POOL):
        raise ValueError("expected an %dx%d comparison block, got %dx%d" % (POOL, POOL, I, J))
    W = A + 0.5
    off = ~np.eye(I, dtype=bool)
    return np.stack([np.stack([W[k, x][off].reshape(I, I - 1).mean(axis=1)
                               for x in range(X)]) for k in range(K)])


def rho_of_lambda(pbar, qbar, lam, tau):
    """Eq. (3), normalised per prompt so rho is an importance ratio under pi_ref."""
    logits = (pbar + np.tensordot(lam, qbar, axes=(0, 0))) / tau - 1.0
    logits = logits - logits.max(axis=1, keepdims=True)          # per-prompt stabiliser
    r = np.exp(logits)
    return r / r.mean(axis=1, keepdims=True)


def lower_bound(rho, qbar_k, eps, chi):
    """max over the KL(eps) ball of the underestimated value, the paper's Problem (5).

    Closed form: -chi ln E[exp(-rho q / chi)] - chi eps, maximised over chi > 0.
    Same Donsker-Varadhan form as this campaign's bank diagnostic.
    """
    z = -(rho * qbar_k) / chi
    m = z.max()
    return float(-chi * (np.log(np.mean(np.exp(z - m))) + m) - chi * eps)


def best_lower_bound(rho, qbar_k, eps, grid, iters=200):
    if eps <= 0.0:
        # eps = 0 leaves the adversary no room, so the lower bound IS the value.
        # This is the paper Problem (3) constraint on the estimated values
        # directly, before the chi robustification is layered on.
        return float(np.mean(rho * qbar_k))
    """Maximise over chi > 0 by ternary search on log chi, which is exact to
    machine precision because the objective is concave in chi.

    A log-spaced grid was tried first and left a 1e-4 relative gap against a
    direct constrained solve -- small, but it is the quantity the dual reads, so
    the resolution of a grid is not the right thing for it to depend on. `grid`
    is retained only to bracket the search."""
    lo, hi = float(np.min(grid)), float(np.max(grid))
    llo, lhi = np.log(lo), np.log(hi)
    for _ in range(iters):
        a = llo + (lhi - llo) / 3.0
        b = lhi - (lhi - llo) / 3.0
        if lower_bound(rho, qbar_k, eps, np.exp(a)) < lower_bound(rho, qbar_k, eps, np.exp(b)):
            llo = a
        else:
            lhi = b
    return lower_bound(rho, qbar_k, eps, np.exp(0.5 * (llo + lhi)))


def dual_objective(lam, pbar, qbar, b, tau, eps, grid):
    """J(lambda) = F(rho_lambda) - lambda^T (b - LowerBound(G(rho_lambda)))."""
    rho = rho_of_lambda(pbar, qbar, lam, tau)
    F = float(np.mean(rho * pbar) - tau * np.mean(rho * np.log(np.maximum(rho, 1e-300))))
    lb = np.array([best_lower_bound(rho, qbar[k], eps, grid) for k in range(qbar.shape[0])])
    return F - float(lam @ (b - lb)), rho, lb, F


def calibrate_epsilon(qbar, beta, grid, lo=1e-12, hi=1.0, iters=200):
    """Choose epsilon so the robustification consumes exactly the slack beta allows.

    Solves max_k [G_k(1) - LowerBound_k(1)] = (1 - beta) * min_k G_k(1) by
    bisection; the gap is monotone in epsilon. See the module docstring for why
    the paper's 0.15 cannot be transplanted onto an eight-atom pool.
    """
    ones = np.ones(qbar.shape[1:])
    G = qbar.mean(axis=(1, 2))
    want = (1.0 - beta) * float(G.min())

    def gap(eps):
        lb = np.array([best_lower_bound(ones, qbar[k], eps, grid)
                       for k in range(qbar.shape[0])])
        return float(np.max(G - lb))

    if gap(lo) > want:
        raise SystemExit("even epsilon=%g over-consumes the slack; the pool is too small "
                         "for this robustification" % lo)
    for _ in range(iters):
        mid = np.sqrt(lo * hi)
        if gap(mid) > want:
            hi = mid
        else:
            lo = mid
    eps = float(np.sqrt(lo * hi))
    return eps, {"target_gap": want, "achieved_gap": gap(eps), "G_reference": G.tolist()}


def solve_dual(pbar, qbar, b, tau, eps, *, iters, step, grid, verbose=False):
    """Projected subgradient on lambda >= 0, the outer minimisation of Eq. (3).

    Step decays as 1/sqrt(t), the standard schedule for a subgradient method on a
    nonsmooth dual, and the averaged iterate is returned. A runaway multiplier is
    a failure, not an answer: the first version reported lambda = 73.8 with a
    one-hot rho as though it had converged, so divergence is detected and raised.
    """
    lam = np.zeros(qbar.shape[0])
    history, running = [], np.zeros_like(lam)
    for t in range(1, iters + 1):
        J, rho, lb, F = dual_objective(lam, pbar, qbar, b, tau, eps, grid)
        grad = -(b - lb)                       # dJ/dlambda at fixed rho (envelope theorem)
        lam = np.maximum(0.0, lam - (step / np.sqrt(t)) * grad)
        running += lam
        history.append({"iter": t, "J": J, "F": F, "lambda": lam.tolist(),
                        "slack": (lb - b).tolist()})
        if verbose and t % max(1, iters // 10) == 0:
            print(json.dumps(history[-1]), flush=True)
    half = np.array(history[iters // 2]["lambda"])
    last = np.array(history[-1]["lambda"])
    if np.max(last) > 10.0 * max(np.max(half), 1e-9) and np.max(last) > 1.0:
        raise SystemExit(
            "dual diverged: lambda grew from %s at iteration %d to %s at %d, so the "
            "constraints are never satisfied and rho degenerates. Check epsilon against "
            "the pool size before reading anything into the result."
            % (half.round(4).tolist(), iters // 2, last.round(4).tolist(), iters))
    lam = running / iters
    J, rho, lb, F = dual_objective(lam, pbar, qbar, b, tau, eps, grid)
    if float(np.min(lb - b)) < -1e-6:
        raise SystemExit("final iterate violates a constraint by %.3e; not writing targets"
                         % float(-np.min(lb - b)))
    return lam, rho, lb, J, F, history


def verify_lower_bound(rng, grid, epsilons=(0.15, 0.01, 1e-3, 1e-5), draws=20):
    """Check the closed form against a direct constrained minimisation.

    Problem (5) minimises the value over a KL ball, so the direct value is
    +sum w v at the tilted w found by bisecting the radius; an earlier version
    wrote -sum w v and failed with a relative residual of exactly 2, which is
    what a sign flip looks like. The Lagrangian value equals the constrained
    value at the KKT point, where KL(w||u) = eps makes the penalty vanish.

    The sweep is over fixed radii rather than the run's own epsilon: this tests
    the closed form, and the run's epsilon is derived from the data. It also
    calls best_lower_bound, not a grid, because checking a code path the solver
    does not take is not a check.
    """
    worst = 0.0
    for eps in epsilons:
        for _ in range(draws):
            n = int(rng.integers(4, 12))
            v = rng.normal(size=n)
            closed = best_lower_bound(np.ones(n), v, eps, grid)

            def kl_of(c):
                z = -v / c
                z = z - z.max()
                w = np.exp(z)
                w = w / w.sum()
                return float(np.sum(w * np.log(np.maximum(w * n, 1e-300)))), w

            lo, hi = 1e-6, 1e6
            for _ in range(300):
                mid = np.sqrt(lo * hi)
                kl, _ = kl_of(mid)
                if kl > eps:
                    lo = mid
                else:
                    hi = mid
            _, w = kl_of(np.sqrt(lo * hi))
            direct = float(np.sum(w * v))
            worst = max(worst, abs(closed - direct) / max(1.0, abs(direct)))
    return worst


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tensor-from", default=str(ROOT / "targets/nash_v1"),
                    help="reuse an existing target set's teacher tensor; every set on disk "
                         "carries the byte-identical pool settings hash")
    ap.add_argument("--out-name", default="mopo_v1")
    ap.add_argument("--primary", default="instruction_following", choices=list(OBJECTIVES))
    ap.add_argument("--tau", type=float, default=0.08)
    ap.add_argument("--epsilon", type=float, default=None,
                    help="the paper's value is 0.15; omit to calibrate against beta")
    ap.add_argument("--epsilon-mode", default="calibrated",
                    choices=("calibrated", "paper"))
    ap.add_argument("--beta", type=float, default=0.9995)
    ap.add_argument("--dual-iters", type=int, default=300)
    ap.add_argument("--dual-step", type=float, default=0.5)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # Bracket for the chi search, not a grid. It must cover small radii: the
    # optimal chi grows like 1/sqrt(eps), so a 1e2 ceiling silently truncated
    # the maximisation at eps = 1e-5 and showed up as a 3e-3 self-test residual.
    grid = np.array([1e-6, 1e6])
    residual = verify_lower_bound(np.random.default_rng(20260912), grid)
    print(json.dumps({"lower_bound_closed_form_vs_direct_rel_residual": residual}), flush=True)
    if residual > 1e-9:
        raise SystemExit("the lower-bound closed form disagrees with direct optimisation")

    src = Path(args.tensor_from)
    report = {"source_target_set": str(src),
              "declarations": {
                  "primary_objective": args.primary,
                  "primary_objective_rule": ("the paper fixes the primary objective "
                                             "positionally (p_K, r_1 in every experiment); "
                                             "applied to our pre-registered objective order"),
                  "tau": args.tau, "beta": args.beta,
                  "epsilon_rule": ("derived so the robustification consumes exactly the slack "
                                   "beta allows at the reference policy; the paper's 0.15 is a "
                                   "KL radius on a full response space and is infeasible on an "
                                   "eight-atom pool"),
                  "t0": "infinity (no lagged refresh); one of the four settings the paper ablates",
                  "b_rule": "beta * G(rho = 1): beta times the reference policy's own value",
                  "secondary_average": "over the seven other candidates; self-comparison excluded",
                  "rho_normalisation": "per prompt, E_{pi_ref}[rho] = 1, the paper's Policy "
                                       "Extraction step"},
              "splits": {}}

    for split in ("train", "dev"):
        tensor_dir = src / split / "tensor"
        meta = json.loads((tensor_dir / "meta.json").read_text())
        A = np.load(tensor_dir / "tensor_policy.npz")["A"]
        if list(meta["objectives"]) != list(OBJECTIVES):
            raise ValueError("objective order differs from the campaign's: %s" % meta["objectives"])
        kp = list(OBJECTIVES).index(args.primary)
        Wbar = candidate_values(A)
        pbar = Wbar[kp]
        qbar = np.stack([Wbar[k] for k in range(len(OBJECTIVES)) if k != kp])
        if args.epsilon_mode == "paper":
            eps = 0.15 if args.epsilon is None else args.epsilon
            eps_report = {"mode": "paper", "epsilon": eps}
        else:
            eps, eps_report = calibrate_epsilon(qbar, args.beta, grid)
            eps_report = {"mode": "calibrated", "epsilon": eps, **eps_report}
        # Record what the paper's own value would have cost here, so the
        # departure is evidenced in the artifact and not only in prose.
        ones = np.ones(qbar.shape[1:])
        eps_report["paper_epsilon_gap_per_objective"] = [
            float(qbar[k].mean() - best_lower_bound(ones, qbar[k], 0.15, grid))
            for k in range(qbar.shape[0])]
        print(json.dumps({"split": split, "epsilon": eps_report}, indent=1), flush=True)
        # b at t = 0: rho = 1 is the reference policy itself.
        g_ref = qbar.mean(axis=(1, 2))
        b = args.beta * g_ref
        lam, rho, lb, J, F, history = solve_dual(
            pbar, qbar, b, args.tau, eps,
            iters=args.dual_iters, step=args.dual_step, grid=grid, verbose=False)
        entry = {
            "n_prompts": int(A.shape[1]), "pool": int(A.shape[2]),
            "primary_index": kp,
            "secondary_objectives": [o for i, o in enumerate(OBJECTIVES) if i != kp],
            "G_reference": g_ref.tolist(), "b": b.tolist(),
            "lambda": lam.tolist(), "lower_bound_at_solution": lb.tolist(),
            "constraint_slack": (lb - b).tolist(),
            "dual_value": J, "primary_value_F": F,
            "rho": {"mean": float(rho.mean()), "min": float(rho.min()),
                    "max": float(rho.max()), "p99": float(np.quantile(rho, 0.99)),
                    "per_prompt_mean_max_dev": float(np.abs(rho.mean(axis=1) - 1).max())},
            "pool_settings_sha256": meta["pool_settings_sha256"],
            "dual_iterations": args.dual_iters,
            "epsilon": eps_report,
        }
        report["splits"][split] = entry
        print(json.dumps({"split": split, **{k: v for k, v in entry.items()
                                             if k not in ("pool_settings_sha256",)}},
                         indent=1), flush=True)
        if not args.dry_run:
            out = ROOT / "targets" / args.out_name / split
            out.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out / "rho.npz", rho=rho, lam=lam, b=b,
                                pbar=pbar, qbar=qbar)
            (out / "dual_history.json").write_text(json.dumps(history[-20:], indent=1) + "\n")

    if args.dry_run:
        print(json.dumps({"dry_run": True, "nothing_written": True}))
        return
    outdir = ROOT / "targets" / args.out_name
    report["source_sha256"] = base.file_hash(Path(__file__))
    (outdir / "complete.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"written": str(outdir / "complete.json")}), flush=True)


if __name__ == "__main__":
    main()
