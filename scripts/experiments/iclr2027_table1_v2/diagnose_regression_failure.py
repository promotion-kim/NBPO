#!/usr/bin/env python3
"""Is the regression failing because it cannot fit, or because the target is noise?

The Section-7 gate scores ``h`` against the SAMPLED Eq. (24) target, which
carries opponent-draw and Bernoulli noise no policy can see. A policy could
therefore be learning the predictable part perfectly and still score
``normalized_mse_var`` near 1. This script separates the two by scoring ``h``
against the conditional mean

    m(x,i,i') = sum_k lambda_k ( A[k,x,i,:] - A[k,x,i',:] ) . nu[k,x]

which is exactly the part of the target that is a function of what the policy
sees. ``corr(h, m)`` is the question; ``corr(h, z)`` is what the gate measures.

If ``corr(h, m)`` is also ~0, the policy learned nothing and the gate is right.
If it is materially positive, the gate is noise-masked and that is a finding
about the METRIC, to be reported rather than silently swapped in.
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tensor-dir", type=Path, required=True)
    ap.add_argument("--solver-dir", type=Path, required=True)
    ap.add_argument("--h", nargs="+", type=Path, required=True)
    ap.add_argument("--heldout-split", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    sol = json.loads((args.solver_dir / "solution.json").read_text())
    lam = np.asarray(sol["lambda_raw"], float)
    meta = json.loads((args.tensor_dir / "meta.json").read_text())
    A = np.load(args.tensor_dir / "tensor_policy.npz")["A"]
    nu = np.load(args.solver_dir / "nu_update.npz")["nu"]
    P = A + 0.5
    Ej = np.einsum("kxij,kxj->kxi", P, nu)          # (K, X, I)

    x_of = {p: i for i, p in enumerate(meta["prompt_ids"])}
    i_of = {r: i for i, r in enumerate(meta["policy_learner_ids"])}
    assign = json.loads(args.heldout_split.read_text())["assignment"]

    report = {"definition": {
        "m": "conditional mean of the sampled target given (prompt, learner pair): "
             "sum_k lambda_k (E_j P(i beats j) - E_j P(i' beats j))",
        "z": "the sampled target the trainer actually regresses on",
        "why": "h is a function of (prompt, pair) only, so m is the largest part "
               "of z that any policy could reproduce"},
        "arms": {}}

    for hp in args.h:
        d = np.load(hp, allow_pickle=True)
        h, z = d["h"], d["z"]
        xs = np.array([x_of[p] for p in d["prompt_id"]])
        i1 = np.array([i_of[c] for c in d["chosen"]])
        i2 = np.array([i_of[r] for r in d["rejected"]])
        m = np.einsum("k,kn->n", lam, Ej[:, xs, i1] - Ej[:, xs, i2])
        half = np.array([assign.get(p, "?") for p in d["prompt_id"]])

        arm = {}
        for name in ("validation", "test"):
            s = half == name
            hh, mm, zz = h[s], m[s], z[s]
            arm[name] = {
                "n": int(s.sum()),
                "corr_h_vs_sampled_target": float(np.corrcoef(hh, zz)[0, 1]),
                "corr_h_vs_conditional_mean": float(np.corrcoef(hh, mm)[0, 1]),
                "spearman_h_vs_conditional_mean": spearman(hh, mm),
                "corr_sampled_target_vs_conditional_mean": float(np.corrcoef(zz, mm)[0, 1]),
                "sign_agreement_h_vs_conditional_mean": float(
                    np.mean(np.sign(hh[mm != 0]) == np.sign(mm[mm != 0]))),
                "normalized_mse_h_vs_conditional_mean": float(
                    np.mean((hh - mm) ** 2) / np.var(mm)),
                "h_rms": float(np.sqrt((hh ** 2).mean())),
                "m_rms": float(np.sqrt((mm ** 2).mean())),
                "z_rms": float(np.sqrt((zz ** 2).mean())),
            }
        report["arms"][hp.stem] = arm

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    for k, v in report["arms"].items():
        for name in ("validation", "test"):
            a = v[name]
            print(f"{k:12s} {name:11s} corr(h,z)={a['corr_h_vs_sampled_target']:+.4f}  "
                  f"corr(h,m)={a['corr_h_vs_conditional_mean']:+.4f}  "
                  f"spearman(h,m)={a['spearman_h_vs_conditional_mean']:+.4f}  "
                  f"sign(h,m)={a['sign_agreement_h_vs_conditional_mean']:.4f}  "
                  f"corr(z,m)={a['corr_sampled_target_vs_conditional_mean']:+.4f}")


if __name__ == "__main__":
    main()
