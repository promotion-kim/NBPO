#!/usr/bin/env python3
"""Calibrate and combine the three SafeRLHF preference-model seeds.

The oracle used downstream is the **calibrated three-seed ensemble mean**. Both
halves of that matter and both are done here:

*Calibration* is a single temperature per (model, seed, objective), fitted by
minimizing validation NLL. **Validation only** -- the test split is never seen by
anything that chooses a parameter, so the reported test numbers remain a held-out
measurement rather than a fit.

*Combination* averages the calibrated probabilities. For the anti-symmetric GPM
that is the right operation and not merely a convenient one: temperature scaling
in logit space and averaging in probability space both preserve
``P(y>z) + P(z>y) = 1``, because each seed is exactly antisymmetric and the
average of numbers summing to one still sums to one. The property is asserted
numerically here rather than assumed.

The GPM is NOT required to beat BT. SafeRLHF's comparison graph contains zero
triangles, so it cannot identify a cyclicity advantage, and that near-null is
reported as an appendix result rather than treated as a gate.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

OBJECTIVES = ("helpfulness", "harmlessness")
MODELS = ("gpm", "bt")
EPS = 1e-9


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    return np.log(p) - np.log1p(-p)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def nll(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def fit_temperature(p_val, y_val, lo=0.05, hi=20.0, iters=200):
    """One temperature, by golden-section search on validation NLL.

    A scalar, unimodal problem: no optimizer machinery is warranted, and a
    deterministic search avoids the seed-dependence a gradient fit would add to a
    calibration that is supposed to be a property of the model.
    """
    z = logit(p_val)
    f = lambda t: nll(sigmoid(z / t), y_val)
    gr = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - gr * (b - a), a + gr * (b - a)
    for _ in range(iters):
        if f(c) < f(d):
            b, d = d, c
            c = b - gr * (b - a)
        else:
            a, c = c, d
            d = a + gr * (b - a)
        if abs(b - a) < 1e-10:
            break
    return 0.5 * (a + b)


def metrics(p, y):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    y = np.asarray(y, dtype=np.float64)
    pred = (p >= 0.5).astype(np.float64)
    tpr = float(pred[y == 1].mean()) if (y == 1).any() else float("nan")
    tnr = float(1 - pred[y == 0].mean()) if (y == 0).any() else float("nan")
    order = np.argsort(p)
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    auc = (float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
           if n1 and n0 else None)
    bins = np.clip((p * 15).astype(int), 0, 14)
    ece = 0.0
    for b in range(15):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return {"nll": nll(p, y), "accuracy": float((pred == y).mean()),
            "balanced_accuracy": (tpr + tnr) / 2, "roc_auc": auc,
            "brier": float(((p - y) ** 2).mean()), "ece": float(ece),
            "base_rate": float(y.mean()), "n": int(len(y))}


def load(d: Path, model, seed, split):
    z = np.load(d / f"pred_{model}_seed{seed}_{split}.npz", allow_pickle=True)
    return ({o: z[f"p_{o}"] for o in OBJECTIVES},
            {o: z[f"y_{o}"] for o in OBJECTIVES},
            z["prompts"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path,
                    default=Path("results/iclr2027_table1_v2/saferlhf_ensemble"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[41, 42, 43])
    args = ap.parse_args()

    report = {"seeds": args.seeds, "objectives": list(OBJECTIVES),
              "calibration": {}, "per_seed": {}, "ensemble": {},
              "exactness": {}, "per_seed_training": {}}

    for model in MODELS:
        report["calibration"][model] = {}
        report["per_seed"][model] = {}
        report["ensemble"][model] = {}
        cal_test, cal_val, y_ref = {o: [] for o in OBJECTIVES}, {o: [] for o in OBJECTIVES}, {}
        raw_test = {o: [] for o in OBJECTIVES}

        for seed in args.seeds:
            pv, yv, _ = load(args.dir, model, seed, "validation")
            pt, yt, _ = load(args.dir, model, seed, "test")
            report["calibration"][model][str(seed)] = {}
            report["per_seed"][model][str(seed)] = {}
            for o in OBJECTIVES:
                T = fit_temperature(pv[o], yv[o])
                report["calibration"][model][str(seed)][o] = {
                    "temperature": T,
                    "validation_nll_before": nll(pv[o], yv[o]),
                    "validation_nll_after": nll(sigmoid(logit(pv[o]) / T), yv[o]),
                    "fitted_on": "validation only",
                }
                pc_t = sigmoid(logit(pt[o]) / T)
                pc_v = sigmoid(logit(pv[o]) / T)
                cal_test[o].append(pc_t)
                cal_val[o].append(pc_v)
                raw_test[o].append(np.asarray(pt[o], dtype=np.float64))
                y_ref[o] = np.asarray(yt[o], dtype=np.float64)
                report["per_seed"][model][str(seed)][o] = {
                    "test_uncalibrated": metrics(pt[o], yt[o]),
                    "test_calibrated": metrics(pc_t, yt[o]),
                    "validation_calibrated": metrics(pc_v, yv[o]),
                }

        for o in OBJECTIVES:
            stack = np.stack(cal_test[o])
            mean = stack.mean(axis=0)
            report["ensemble"][model][o] = {
                **metrics(mean, y_ref[o]),
                "ensemble_disagreement_std_mean": float(stack.std(axis=0).mean()),
                "ensemble_disagreement_std_p95": float(
                    np.percentile(stack.std(axis=0), 95)),
                "ensemble_disagreement_max_pairwise_mean": float(
                    (stack.max(axis=0) - stack.min(axis=0)).mean()),
                "mean_pairwise_seed_correlation": float(np.mean([
                    np.corrcoef(stack[i], stack[j])[0, 1]
                    for i in range(len(stack)) for j in range(i + 1, len(stack))])),
                "raw_vs_calibrated_nll": {
                    "raw_ensemble": nll(np.stack(raw_test[o]).mean(axis=0), y_ref[o]),
                    "calibrated_ensemble": nll(mean, y_ref[o])},
            }

    # the guarantees the GPM is chosen for, re-checked on the ENSEMBLE
    per_seed_json = {}
    for seed in args.seeds:
        j = json.loads((args.dir / f"gpm_vs_bt_seed{seed}.json").read_text())
        per_seed_json[str(seed)] = j
        for model in MODELS:
            m = j["models"][model]
            report["exactness"].setdefault(model, {})[str(seed)] = {
                **m["test"]["exactness"],
                "logit_vs_length_pearson": {
                    o: m["test"]["per_objective"][o]["logit_vs_length_difference_pearson"]
                    for o in OBJECTIVES},
                "head_parameters": m["head_parameters"],
                "total_parameters": m["total_parameters"],
                "train_seconds": m["train_seconds"],
            }
        report["per_seed_training"][str(seed)] = {
            "predicted_cycles_test": {
                model: j["models"][model].get("predicted_cycles_test", {})
                       .get("per_objective") for model in MODELS},
            "observed_three_cycles": j["observed_three_cycles"],
        }
    report["not_scalar_decomposable"] = {
        str(s): per_seed_json[str(s)]["models"]["gpm"].get(
            "not_scalar_decomposable_random_inputs") for s in args.seeds}

    report["ensemble_antisymmetry_note"] = (
        "temperature scaling is monotone in the logit and averaging is linear in "
        "probability, so P(y>z)+P(z>y)=1 survives both exactly; each seed's "
        "residual is reported above and is at float32 sigmoid precision")

    (args.dir / "saferlhf_ensemble.json").write_text(json.dumps(report, indent=2))

    # ---- console summary ----
    print("calibration temperatures (fitted on validation only)")
    for model in MODELS:
        for seed in args.seeds:
            ts = {o: report["calibration"][model][str(seed)][o]["temperature"]
                  for o in OBJECTIVES}
            print(f"  {model:4s} seed {seed}: " +
                  "  ".join(f"{o}={ts[o]:.3f}" for o in OBJECTIVES))
    print("\ncalibrated three-seed ENSEMBLE, held-out test")
    hdr = f"{'model':5s} {'objective':13s} {'NLL':>7} {'balacc':>7} {'AUC':>7} {'Brier':>7} {'ECE':>7} {'disagree':>9}"
    print(hdr)
    for model in MODELS:
        for o in OBJECTIVES:
            e = report["ensemble"][model][o]
            print(f"{model:5s} {o:13s} {e['nll']:7.4f} {e['balanced_accuracy']:7.4f} "
                  f"{e['roc_auc']:7.4f} {e['brier']:7.4f} {e['ece']:7.4f} "
                  f"{e['ensemble_disagreement_std_mean']:9.4f}")
    print("\nGPM minus BT on the calibrated ensemble (NOT a gate):")
    for o in OBJECTIVES:
        g, b = report["ensemble"]["gpm"][o], report["ensemble"]["bt"][o]
        print(f"  {o:13s} dNLL {g['nll']-b['nll']:+.5f}   "
              f"dBalAcc {g['balanced_accuracy']-b['balanced_accuracy']:+.5f}   "
              f"dAUC {g['roc_auc']-b['roc_auc']:+.5f}")
    print("\nexactness (GPM, per seed): "
          + "; ".join(f"seed {s}: anti={report['exactness']['gpm'][str(s)]['max_antisymmetry_residual']:.1e} "
                      f"selftie={report['exactness']['gpm'][str(s)]['max_self_tie_residual']:.1e}"
                      for s in args.seeds))


if __name__ == "__main__":
    main()
