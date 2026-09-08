#!/usr/bin/env python3
"""Frozen-pool surrogate acceptance, with the pool mapping fixed and unit-checked.

The finite-pool problem optimizes a distribution over the response pool, and the
policy induces one through the RELATIVE update

    p_theta(i)  proportional to  p_t(i) * exp( log pi_theta(i) - log pi_t(i) )

An earlier version used ``softmax(log pi_theta)`` instead. That is only the same
thing when ``p_t`` is the conditional LM mass over the pool, and here ``p_t`` is
uniform (verified: every row of ``pi_t.npz`` is 0.125). Using absolute
log-probabilities made the induced distribution track response length and content
rather than what training changed, and it failed the check below at theta =
theta_t. The per-response log-ratio is what the mapping needs, and both terms for
it are already in the scored precompute.

Two unit checks run before anything is reported, and the script refuses to
continue if either fails:

  h = 0                      must recover p_t exactly
  h = log(p*_i / p_t,i)      must recover p*, the solver's own optimizer output

What this measures is a **held-out frozen-pool surrogate**: same frozen pool,
same frozen preference models, no new generation, no independent judge, no
capability evaluation. It is not a measurement of generation quality.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def game_value(pi_x, A_kx, mu, beta):
    m = pi_x @ A_kx
    z = -m / beta
    zm = z.max()
    return -beta * (zm + np.log(np.sum(mu * np.exp(z - zm))))


def pool_from_h(h_x, pt_x):
    """p_theta ∝ p_t * exp(h); h is the per-response log-ratio to pi_t."""
    z = h_x - h_x.max()
    w = pt_x * np.exp(z)
    return w / w.sum()


def values(pi, A, mu, beta, xs):
    """Per-objective V_k, averaged over the given prompts."""
    K = A.shape[0]
    return np.array([[game_value(pi[c], A[k, x], mu, beta[k])
                      for c, x in enumerate(xs)] for k in range(K)])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scored", nargs="*", default=[], help="label=<scored dir>")
    ap.add_argument("--tensor-dir", type=Path, required=True)
    ap.add_argument("--solver-dir", type=Path, required=True)
    ap.add_argument("--heldout-split", type=Path, required=True)
    ap.add_argument("--resamples", type=int, default=1000)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from datasets import load_from_disk

    meta = json.loads((args.tensor_dir / "meta.json").read_text())
    A = np.load(args.tensor_dir / "tensor_policy.npz")["A"]
    A_ref = np.load(args.tensor_dir / "tensor_ref.npz")["A"]
    sol = json.loads((args.solver_dir / "solution.json").read_text())
    beta = np.asarray(sol["config"]["beta"], float)
    pt = np.load(args.solver_dir / "pi_t.npz"); pt = pt[list(pt.keys())[0]]
    ps = np.load(args.solver_dir / "pi_star.npz"); ps = ps[list(ps.keys())[0]]
    K, X, I, J = A.shape
    mu = np.full(J, 1.0 / J)
    mu_ref = np.full(A_ref.shape[2], 1.0 / A_ref.shape[2])
    objectives = meta["objectives"]
    x_of = {p: i for i, p in enumerate(meta["prompt_ids"])}
    i_of = {r: i for i, r in enumerate(meta["policy_learner_ids"])}
    assign = json.loads(args.heldout_split.read_text())["assignment"]

    # ---- unit checks on the mapping, before any result is produced ----
    zero = np.array([pool_from_h(np.zeros(I), pt[x]) for x in range(X)])
    err_pt = float(np.abs(zero - pt).max())
    h_star = np.log(ps) - np.log(pt)
    rec = np.array([pool_from_h(h_star[x], pt[x]) for x in range(X)])
    err_ps = float(np.abs(rec - ps).max())
    checks = {"h_zero_recovers_pi_t_max_abs_error": err_pt,
              "canonical_potential_recovers_pi_star_max_abs_error": err_ps,
              "pi_t_is_uniform_on_the_pool": bool(np.allclose(pt, 1.0 / I))}
    print(json.dumps(checks, indent=2))
    if err_pt > 1e-12 or err_ps > 1e-10:
        raise SystemExit("pool mapping unit check FAILED; refusing to report surplus")

    d = np.array([[game_value(mu_ref, A_ref[k, x], mu, beta[k]) for x in range(X)]
                  for k in range(K)])

    halves = {name: [x for x in range(X)
                     if assign.get(meta["prompt_ids"][x]) == name]
              for name in ("validation", "test")}
    halves["train"] = [x for x in range(X)
                       if meta["prompt_ids"][x] not in assign]

    report = {"objectives": objectives, "beta": beta.tolist(),
              "mapping": "p_theta ∝ p_t * exp(log pi_theta - log pi_t)",
              "unit_checks": checks,
              "measures": ("held-out frozen-pool surrogate acceptance: same frozen "
                           "pool, same frozen preference models, no new generation, "
                           "no independent judge, no capability evaluation"),
              "reference_policies": {}, "arms": {}}

    def summarize(pi_by_x, xs, label, rng=None):
        V = values(pi_by_x, A, mu, beta, xs)                 # (K, n)
        S = V - d[:, xs]
        per = {objectives[k]: {"mean_V": float(V[k].mean()),
                               "mean_d": float(d[k, xs].mean()),
                               "mean_surplus": float(S[k].mean()),
                               "fraction_prompts_positive": float((S[k] > 0).mean())}
               for k in range(K)}
        return {"n_prompts": len(xs), "per_objective": per,
                "min_over_objectives_of_mean_surplus": float(
                    min(v["mean_surplus"] for v in per.values())),
                "note_not_mean_of_min": "min_k E_x[s_kx], not E_x[min_k s_kx]",
                "accepts_all_objectives_positive": bool(
                    all(v["mean_surplus"] > 0 for v in per.values()))}, V, S

    for name, xs in halves.items():
        if not xs:
            continue
        r_t, V_t, _ = summarize(pt[xs], xs, "pi_t")
        r_s, V_s, _ = summarize(ps[xs], xs, "pi_star")
        report["reference_policies"].setdefault("pi_t", {})[name] = r_t
        report["reference_policies"].setdefault("pi_star", {})[name] = r_s

    for spec in args.scored:
        label, path = spec.split("=", 1)
        dd = load_from_disk(str(Path(path) / "precomputed"))["train"].to_dict()
        hmat = np.full((X, I), np.nan)
        for pid, cid, rid, rc, rr, hc, hr in zip(
                dd["prompt_id"], dd["chosen_response_id"], dd["rejected_response_id"],
                dd["reference_chosen_logps"], dd["reference_rejected_logps"],
                dd["history0_chosen_logps"], dd["history0_rejected_logps"]):
            x = x_of[pid]
            hmat[x, i_of[cid]] = rc - hc
            hmat[x, i_of[rid]] = rr - hr
        arm = {}
        for name, xs in halves.items():
            xs = [x for x in xs if np.isfinite(hmat[x]).all()]
            if not xs:
                continue
            pi = np.array([pool_from_h(hmat[x], pt[x]) for x in xs])
            res, V, S = summarize(pi, xs, label)
            V_t = values(pt[xs], A, mu, beta, xs)
            delta = V - V_t                                   # (K, n) paired per prompt
            rng = np.random.default_rng(0)
            boot = {objectives[k]: [] for k in range(K)}
            boot_min = []
            for _ in range(args.resamples):
                sel = rng.integers(0, len(xs), size=len(xs))   # prompts are the unit
                means = [float(delta[k][sel].mean()) for k in range(K)]
                for k in range(K):
                    boot[objectives[k]].append(means[k])
                boot_min.append(min(means))
            ci = lambda v: {"ci95_low": float(np.percentile(v, 2.5)),
                            "ci95_high": float(np.percentile(v, 97.5))}
            res["paired_delta_V_vs_pi_t"] = {
                objectives[k]: {"mean": float(delta[k].mean()),
                                **ci(boot[objectives[k]])} for k in range(K)}
            res["paired_delta_min_over_objectives"] = {
                "mean": float(min(delta[k].mean() for k in range(K))), **ci(boot_min)}
            res["delta_note"] = ("a positive paired delta means the policy raised that "
                                 "objective's game value above pi_t; it does NOT "
                                 "substitute for the absolute surplus > 0 condition")
            arm[name] = res
        report["arms"][label] = arm

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")

    def row(lbl, r):
        per = r["per_objective"]
        vals = "  ".join(f"{per[o]['mean_surplus']:+9.5f}" for o in objectives)
        dm = r.get("paired_delta_min_over_objectives")
        dtxt = (f"  dmin {dm['mean']:+8.5f} [{dm['ci95_low']:+.5f},{dm['ci95_high']:+.5f}]"
                if dm else "")
        print(f"{lbl:>22} {r['n_prompts']:>5}  {vals}  min {r['min_over_objectives_of_mean_surplus']:+9.5f}"
              f"  accepts {str(r['accepts_all_objectives_positive']):>5}{dtxt}")

    for half in ("train", "validation", "test"):
        print(f"\n=== {half} ===")
        print(f"{'policy':>22} {'n':>5}  " + "  ".join(f"{o[:9]:>9}" for o in objectives))
        for k in ("pi_t", "pi_star"):
            if half in report["reference_policies"].get(k, {}):
                row(k, report["reference_policies"][k][half])
        for lbl, arm in report["arms"].items():
            if half in arm:
                row(lbl, arm[half])


if __name__ == "__main__":
    main()
