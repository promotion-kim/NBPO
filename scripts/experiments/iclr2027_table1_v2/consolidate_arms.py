#!/usr/bin/env python3
"""One index of every trained arm: gate metrics, surplus, and what it consumed.

The paper table was reading the release manifest, which lists only the arms that
were uploaded, so arms trained afterwards were invisible to it. This gathers
every scored arm from its own gate file instead, joins the corrected surplus and
the realized accounting, and records for each one which target estimator, which
reference path and which code tree produced it -- the three things that make two
arms comparable or not.

Gate status is computed per criterion and reported per half. A criterion with no
measurement is `unmeasured`, never `pass`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

GATE = {"normalized_mse_var": ("<", 0.90), "sign_agreement": (">", 0.65),
        "pearson": (">", 0.0), "spearman": (">", 0.0)}


def status(m):
    out = {}
    for k, (op, thr) in GATE.items():
        v = m.get(k)
        out[k] = ("unmeasured" if v is None else
                  ("pass" if (v < thr if op == "<" else v > thr) else "fail"))
    out["all_pass"] = ("pass" if all(v == "pass" for k, v in out.items()
                                     if k in GATE) else "fail")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+", required=True,
                    help="label=<gate.json>=<estimator>=<reference path>=<code tree>")
    ap.add_argument("--surplus", type=Path, required=True)
    ap.add_argument("--accounting", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    sur = json.loads(args.surplus.read_text())
    acct = {r["arm"]: r for r in json.loads(args.accounting.read_text())} \
        if args.accounting and args.accounting.exists() else {}

    index = {"gate_thresholds": {k: f"{op} {thr}" for k, (op, thr) in GATE.items()},
             "selection_rule": "validation selects; the test half is reported once",
             "surplus_rule": ("Algorithm 1 accepts only if EVERY objective's held-out "
                              "surplus is positive"),
             "reference_policies": sur.get("reference_policies", {}),
             "arms": {}}

    for spec in args.arms:
        label, gate_path, est, refpath, tree = spec.split("=")
        g = json.loads(Path(gate_path).read_text())
        rec = {"estimator": est, "reference_path": refpath, "code_tree": tree,
               "eta": g["eta"], "gate_file": gate_path}
        for half in ("validation", "test"):
            m = g["splits"][half]["metrics"]
            s = sur.get("arms", {}).get(label, {}).get(half, {})
            rec[half] = {
                "metrics": {k: m.get(k) for k in
                            ("normalized_mse_var", "normalized_mse_zero",
                             "sign_agreement", "pearson", "spearman",
                             "prediction_rms", "target_rms", "n")},
                "gate": status(m),
                "surplus": {
                    "per_objective": s.get("per_objective"),
                    "min_over_objectives": s.get("min_over_objectives_of_mean_surplus"),
                    "accepts": s.get("accepts_all_objectives_positive"),
                    "paired_delta_min_vs_pi_t": s.get("paired_delta_min_over_objectives"),
                } if s else None,
            }
        if label in acct:
            rec["consumed"] = {k: acct[label][k] for k in
                               ("train_prompts", "train_pair_rows", "effective_batch",
                                "optimizer_updates", "exposure_epochs",
                                "approx_response_tokens_consumed", "gpu_hours")}
        index["arms"][label] = rec

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(index, indent=2) + "\n")

    hdr = (f"{'arm':>18} {'est':>9} {'ref':>7} {'val nMSE':>9} {'test nMSE':>10} "
           f"{'sign':>6} {'r':>7} {'rho':>7} {'val acc':>8} {'test acc':>9} {'gate':>5}")
    print(hdr); print("-" * len(hdr))
    for lbl, r in index["arms"].items():
        t, v = r["test"], r["validation"]
        va = v["surplus"]["accepts"] if v["surplus"] else None
        ta = t["surplus"]["accepts"] if t["surplus"] else None
        print(f"{lbl:>18} {r['estimator']:>9} {r['reference_path']:>7} "
              f"{v['metrics']['normalized_mse_var']:>9.4f} "
              f"{t['metrics']['normalized_mse_var']:>10.4f} "
              f"{t['metrics']['sign_agreement']:>6.4f} "
              f"{t['metrics']['pearson']:>+7.4f} {t['metrics']['spearman']:>+7.4f} "
              f"{str(va):>8} {str(ta):>9} {v['gate']['all_pass']:>5}")


if __name__ == "__main__":
    main()
