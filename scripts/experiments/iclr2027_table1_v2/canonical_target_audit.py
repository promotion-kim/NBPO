#!/usr/bin/env python3
"""What IS the target, exactly? An audit of the finite-pool target chain.

Three quantities are meant to coincide, and the audit checks each link rather
than assuming it:

``m_ab``   the realized proximal log-ratio change, read off the solver's own
           optimizer: ``log(p*[a]/p_t[a]) - log(p*[b]/p_t[b])``
``mQ_ab``  the KKT form, ``eta * sum_k lambda_k sum_j nu*[k,j] (A[k,a,j]-A[k,b,j])``
``T_ab``   whatever the trainer was actually handed, per estimator

`m` and `mQ` are the same object by the finite-pool KKT identity, so their
residual is a check on the artifacts. `T` is an ESTIMATOR of that object, and
the point of the audit is to say -- per estimator, in the same units, on the same
pairs -- how far it sits from the thing it estimates and what randomness is left
in it.

The "canonical" target is `m`: it is computable in closed form from the payoff
tensor already on disk, so nothing here needs new generation or scoring.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def compare(name, T, m):
    d = T - m
    return {
        "estimator": name,
        "n": int(T.size),
        "mean_T": float(T.mean()), "rms_T": float(np.sqrt((T ** 2).mean())),
        "var_T": float(T.var()),
        "mean_difference": float(d.mean()),
        "rms_difference": float(np.sqrt((d ** 2).mean())),
        "max_abs_difference": float(np.abs(d).max()),
        "pearson_with_m": float(np.corrcoef(T, m)[0, 1]),
        "spearman_with_m": spearman(T, m),
        "var_of_residual_over_var_m": float(d.var() / m.var()),
        "fraction_of_var_T_explained_by_m": float(1 - d.var() / T.var()) if T.var() else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tensor-dir", type=Path, required=True)
    ap.add_argument("--solver-dir", type=Path, required=True)
    ap.add_argument("--pairs", nargs="*", default=[],
                    help="name=path.jsonl of a realized pair file to audit")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    sol = json.loads((args.solver_dir / "solution.json").read_text())
    lam = np.asarray(sol["lambda_raw"], float)
    eta = float(sol["eta"])
    meta = json.loads((args.tensor_dir / "meta.json").read_text())
    A = np.load(args.tensor_dir / "tensor_policy.npz")["A"]        # (K, X, I, J) centered
    K, X, I, J = A.shape

    art = {p.stem: p for p in args.solver_dir.glob("*.npz")}
    pi_star = np.load(args.solver_dir / "pi_star.npz")
    pi_t = np.load(args.solver_dir / "pi_t.npz")
    ps = pi_star[list(pi_star.keys())[0]]
    pt = pi_t[list(pi_t.keys())[0]]

    report = {
        "solver_artifacts_present": sorted(art),
        "eta": eta, "lambda_raw": lam.tolist(),
        "shapes": {"A": list(A.shape), "pi_star": list(ps.shape), "pi_t": list(pt.shape)},
        "opponent_source_note": (
            "nu_update is the opponent used to BUILD the update; nu_final_policy is "
            "the opponent at the final policy. Eq. (27) is stated at the maximizer, "
            "so the KKT form is checked against BOTH and the residuals are reported."),
    }

    # ---- canonical m, straight from the optimizer -------------------------
    logratio = np.log(ps) - np.log(pt)                              # (X, I)
    pairs = list(itertools.combinations(range(I), 2))
    a_idx = np.array([a for a, _ in pairs])
    b_idx = np.array([b for _, b in pairs])
    m = logratio[:, a_idx] - logratio[:, b_idx]                     # (X, P)

    # ---- KKT form, for each opponent artifact -----------------------------
    for nu_name in ("nu_update", "nu_final_policy"):
        f = args.solver_dir / f"{nu_name}.npz"
        if not f.exists():
            continue
        nu = np.load(f)["nu"]
        q = np.einsum("kxij,kxj->kxi", A + 0.5, nu)                 # (K, X, I)
        mQ = eta * np.einsum("k,kxp->xp", lam, q[:, :, a_idx] - q[:, :, b_idx])
        r = m - mQ
        report[f"kkt_identity_vs_{nu_name}"] = {
            "max_abs_residual": float(np.abs(r).max()),
            "rms_residual": float(np.sqrt((r ** 2).mean())),
            "rms_m": float(np.sqrt((m ** 2).mean())),
            "relative_rms": float(np.sqrt((r ** 2).mean()) / np.sqrt((m ** 2).mean())),
        }
        if nu_name == "nu_update":
            m_kkt = mQ

    # ---- how much randomness does each estimator leave? -------------------
    # E_j only (comparator integrated, Bernoulli integrated) == the canonical m
    P = A + 0.5
    nu = np.load(args.solver_dir / "nu_update.npz")["nu"]
    Ej = np.einsum("kxij,kxj->kxi", P, nu)
    Ej2 = np.einsum("kxij,kxj->kxi", P * P, nu)
    var_bernoulli = np.zeros_like(m)
    var_comparator = np.zeros_like(m)
    for c, (a, b) in enumerate(pairs):
        # Var over the Bernoulli flips at a FIXED drawn j, averaged over j
        ez2 = (Ej[:, :, a] - Ej2[:, :, a]) + (Ej[:, :, b] - Ej2[:, :, b])
        var_bernoulli[:, c] = (lam ** 2) @ ez2
        # Var over the comparator draw of (p_a - p_b)
        dm = P[:, :, a, :] - P[:, :, b, :]
        v = np.einsum("kxj,kxj->kx", dm * dm, nu) - (Ej[:, :, a] - Ej[:, :, b]) ** 2
        var_comparator[:, c] = (lam ** 2) @ v
    report["randomness_budget_per_pair"] = {
        "var_of_canonical_m": float(m.var()),
        "mean_bernoulli_variance": float(var_bernoulli.mean()),
        "mean_comparator_variance": float(var_comparator.mean()),
        "note": ("the sampled estimator carries both terms; the shipped "
                 "rao_blackwell mode integrates the BERNOULLI term only and still "
                 "draws one comparator per (row, objective), so it is a PARTIAL "
                 "Rao-Blackwellization, not the full finite-pool expectation"),
        "implied_max_r2_sampled": float(
            m.var() / (m.var() + var_bernoulli.mean() + var_comparator.mean())),
        "implied_max_r2_partial_rb": float(
            m.var() / (m.var() + var_comparator.mean())),
        "implied_max_r2_canonical": 1.0,
    }

    # ---- realized pair files ---------------------------------------------
    x_of = {p: i for i, p in enumerate(meta["prompt_ids"])}
    i_of = {r: i for i, r in enumerate(meta["policy_learner_ids"])}
    pair_col = {(a, b): c for c, (a, b) in enumerate(pairs)}
    report["realized_pair_files"] = {}
    for spec in args.pairs:
        name, path = spec.split("=", 1)
        T, M = [], []
        for line in open(path):
            r = json.loads(line)
            x = x_of[r["prompt_id"]]
            a, b = i_of[r["chosen_response_id"]], i_of[r["rejected_response_id"]]
            sgn = 1.0
            if (a, b) not in pair_col:
                a, b, sgn = b, a, -1.0
            T.append(r["nbpo_weighted_z"] * sgn)
            M.append(m[x, pair_col[(a, b)]])
        report["realized_pair_files"][name] = compare(name, np.asarray(T), np.asarray(M))
        report["realized_pair_files"][name]["orientation_note"] = (
            "pair columns are stored for a<b; a row whose chosen index exceeds its "
            "rejected index is compared against the sign-flipped canonical value")

    # ---- is the canonical target realizable by ANY policy? ----------------
    # every policy log-ratio is v_a - v_b, so solve min_v sum (v_a-v_b-m_ab)^2
    # with the gauge sum(v)=0; the residual bounds what a perfect policy could do.
    D = np.zeros((len(pairs), I))
    for c, (a, b) in enumerate(pairs):
        D[c, a], D[c, b] = 1.0, -1.0
    G = np.vstack([D, np.ones((1, I))])
    res = []
    for x in range(X):
        rhs = np.concatenate([m[x], [0.0]])
        v, *_ = np.linalg.lstsq(G, rhs, rcond=None)
        res.append(D @ v - m[x])
    res = np.asarray(res)
    report["realizability_of_canonical_m"] = {
        "max_abs_pairwise_residual": float(np.abs(res).max()),
        "rms_pairwise_residual": float(np.sqrt((res ** 2).mean())),
        "rms_m": float(np.sqrt((m ** 2).mean())),
        "note": ("any policy log-ratio difference is v_a - v_b, so a nonzero residual "
                 "here would mean the canonical target is not exactly realizable by "
                 "ANY policy on this pool. It should be at numerical-error level."),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2)[:4000])


if __name__ == "__main__":
    main()
