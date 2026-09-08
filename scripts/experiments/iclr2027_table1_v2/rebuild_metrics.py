#!/usr/bin/env python3
"""Recompute every arm's regression metrics from one definition, with identities.

Two exact decompositions of the mean squared error, both reported so neither can
be quietly assumed:

    MSE = E[T^2] + E[h^2] - 2 E[hT]
    MSE = Var(T) + Var(h) - 2 Cov(h,T) + (mean(h) - mean(T))^2

The form used in the earlier write-up, ``Var(T) + E[h^2] - 2 Cov``, is only
correct when ``mean(T) = 0``; likewise the "h = 0 scores exactly 1" reading of a
variance-normalized metric holds only at zero target mean. Both means are
therefore reported alongside, and the residual of each identity is printed so a
reader can see they close.

Two normalizations, because they answer different questions:

``nMSE_var``   MSE / Var(T)      -- beats the best CONSTANT predictor
``nMSE_zero``  MSE / E[T^2]      -- beats predicting ZERO, which is what an
                                    untrained policy gives since h = 0 at pi_t

The pre-registered gate is on ``nMSE_var``. Sign agreement is computed only where
both terms are nonzero, and the count of comparable rows is reported rather than
the near-zero rows being dropped silently to flatter the number.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def full_metrics(h, T):
    h = np.asarray(h, float); T = np.asarray(T, float)
    n = h.size
    mh, mT = float(h.mean()), float(T.mean())
    vh, vT = float(h.var()), float(T.var())
    ET2, Eh2, EhT = float((T ** 2).mean()), float((h ** 2).mean()), float((h * T).mean())
    cov = EhT - mh * mT
    mse = float(((h - T) ** 2).mean())
    nz = (T != 0) & (h != 0)
    pos, neg = int((T > 0).sum()), int((T < 0).sum())
    return {
        "n_pairs": int(n),
        "mean_h": mh, "mean_T": mT, "var_h": vh, "var_T": vT,
        "E_T2": ET2, "E_h2": Eh2, "E_hT": EhT, "cov_hT": cov,
        "mse": mse,
        "zero_predictor_mse": ET2,
        "mean_predictor_mse": vT,
        "nMSE_zero": mse / ET2 if ET2 else None,
        "nMSE_var": mse / vT if vT else None,
        "identity_moment_residual": abs(mse - (ET2 + Eh2 - 2 * EhT)),
        "identity_central_residual": abs(mse - (vT + vh - 2 * cov + (mh - mT) ** 2)),
        "legacy_formula_value": vT + Eh2 - 2 * cov,
        "legacy_formula_error": abs(mse - (vT + Eh2 - 2 * cov)),
        "sign_agreement": float(np.mean(np.sign(h[nz]) == np.sign(T[nz]))) if nz.any() else None,
        "n_sign_comparable": int(nz.sum()),
        "n_target_zero": int((T == 0).sum()),
        "n_target_positive": pos, "n_target_negative": neg,
        "majority_sign_baseline": max(pos, neg) / n if n else None,
        "pearson": float(np.corrcoef(h, T)[0, 1]) if h.std() and T.std() else None,
        "spearman": spearman(h, T) if h.std() and T.std() else None,
        "h_rms": float(np.sqrt(Eh2)), "T_rms": float(np.sqrt(ET2)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scored", nargs="+", required=True,
                    help="label=<scored dir>:<eta> entries. The eta MUST be the one "
                         "that arm trained at: an arm trained at eta=0.1 solved a "
                         "different proximal problem, and scoring it against the "
                         "eta=1 target grades it on someone else's problem.")
    ap.add_argument("--heldout-split", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from datasets import load_from_disk
    assign = json.loads(args.heldout_split.read_text())["assignment"]

    report = {"definitions": {
        "h": "(cand_chosen - cand_rejected) - (pi_t_chosen - pi_t_rejected), "
             "sequence-sum log-probabilities over response tokens, both terms from "
             "the same precompute code path",
        "T": "eta * nbpo_weighted_z, eta applied exactly once, with each arm's "
             "OWN training eta -- arms at different eta are different problems",
        "gate_metric": "nMSE_var on the test half, after selecting on validation",
    }, "arms": {}}

    for spec in args.scored:
        label, rest = spec.split("=", 1)
        path, eta = rest.rsplit(":", 1)
        eta = float(eta)
        d = load_from_disk(str(Path(path) / "precomputed"))["train"].to_dict()
        h = (np.asarray(d["reference_chosen_logps"], float)
             - np.asarray(d["reference_rejected_logps"], float)
             - np.asarray(d["history0_chosen_logps"], float)
             + np.asarray(d["history0_rejected_logps"], float))
        T = eta * np.asarray(d["nbpo_weighted_z"], float)
        half = np.asarray([assign.get(p, "?") for p in d["prompt_id"]])
        report["arms"][label] = {
            "eta": eta,
            **{name: full_metrics(h[half == name], T[half == name])
               for name in ("validation", "test")}}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")

    hdr = f"{'arm':>16} {'half':>10} {'nMSE_var':>9} {'nMSE_zero':>9} {'sign':>7} " \
          f"{'pearson':>8} {'mean_T':>8} {'legacy_err':>10} {'id_resid':>9}"
    print(hdr); print("-" * len(hdr))
    for label, halves in report["arms"].items():
        for name in ("validation", "test"):
            m = halves[name]
            print(f"{label:>16} {name:>10} {m['nMSE_var']:9.4f} {m['nMSE_zero']:9.4f} "
                  f"{m['sign_agreement']:7.4f} {m['pearson']:+8.4f} {m['mean_T']:+8.4f} "
                  f"{m['legacy_formula_error']:10.2e} {m['identity_moment_residual']:9.2e}")


if __name__ == "__main__":
    main()
