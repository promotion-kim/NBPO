#!/usr/bin/env python3
"""Section-7 gate: does the trained policy actually realize the Eq. (26) target?

The quantity the trainer regresses is

    h_t = [log pi(y|x) - log pi(z|x)] - [log pi_t(y|x) - log pi_t(z|x)]
    target = eta * nbpo_weighted_z

and the gate asks whether ``h_t`` tracks ``target`` on prompts the fit never
saw. It is a REGRESSION diagnostic, not a win rate: a policy can move win rates
around while having r^2 ~ 0 on its own target, which is exactly the failure that
made an earlier round of tables uninterpretable.

Both logps come from ``mnpo_scripts.precompute`` runs, so the tokenizer, chat
template, response mask, truncation rule and the sequence-SUM reduction are the
ones training used, rather than a second implementation that might differ.

Normalization is reported two ways and neither is hidden:

``normalized_mse_var``   MSE / Var(target)   -- 1 - r^2, beats the best CONSTANT
``normalized_mse_ms``    MSE / mean(target^2) -- beats predicting ZERO, which is
                         what an untrained policy gives, since h_t = 0 at pi = pi_t

The pre-registered gate is on ``normalized_mse_var``; the other is reported
because at t=0 the zero predictor is the honest baseline and the two differ
whenever the target has a nonzero mean.

The validation half selects; the test half is scored once. Which prompt is in
which half comes from the frozen split map, never from row order.
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


def metrics(h, t):
    h = np.asarray(h, float)
    t = np.asarray(t, float)
    mse = float(np.mean((h - t) ** 2))
    var = float(np.var(t))
    ms = float(np.mean(t ** 2))
    nz = (t != 0) & (h != 0)
    return {
        "n": int(len(t)),
        "target_rms": float(np.sqrt(ms)),
        "target_mean": float(np.mean(t)),
        "prediction_rms": float(np.sqrt(np.mean(h ** 2))),
        "mse": mse,
        "normalized_mse_var": mse / var if var > 0 else None,
        "normalized_mse_ms": mse / ms if ms > 0 else None,
        "sign_agreement": float(np.mean(np.sign(h[nz]) == np.sign(t[nz]))) if nz.any() else None,
        "n_sign_comparable": int(nz.sum()),
        "pearson": float(np.corrcoef(h, t)[0, 1]) if h.std() > 0 and t.std() > 0 else None,
        "spearman": spearman(h, t) if h.std() > 0 and t.std() > 0 else None,
    }


GATE = {
    "normalized_mse_var": ("<", 0.90),
    "sign_agreement": (">", 0.65),
    "pearson": (">", 0.0),
    "spearman": (">", 0.0),
}


def check(m):
    out = {}
    for k, (op, thr) in GATE.items():
        v = m.get(k)
        out[k] = {"value": v, "threshold": thr, "operator": op,
                  "passes": None if v is None else (v < thr if op == "<" else v > thr)}
    out["all_pass"] = all(c["passes"] for c in out.values() if isinstance(c, dict))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidate-precomputed", type=Path, required=True,
                    help="precompute output whose 'reference' columns are the CANDIDATE "
                         "policy's logps and whose history0 columns are pi_t's")
    ap.add_argument("--split", default="train",
                    help="split inside that dataset holding the held-out pairs")
    ap.add_argument("--heldout-split", type=Path, required=True)
    ap.add_argument("--eta", type=float, required=True)
    ap.add_argument("--method", default="nbpo")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--label", default="")
    ap.add_argument("--expected-solver-hash", default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from datasets import load_from_disk

    ds = load_from_disk(str(args.candidate_precomputed))[args.split]
    assign = json.loads(args.heldout_split.read_text())["assignment"]

    cols = set(ds.column_names)
    need = {"reference_chosen_logps", "reference_rejected_logps",
            "history0_chosen_logps", "history0_rejected_logps",
            "nbpo_weighted_z", "prompt_id"}
    if not need <= cols:
        raise SystemExit(f"missing columns: {sorted(need - cols)}")

    d = ds.to_dict()
    if args.expected_solver_hash:
        got = set(d.get("solver_hash", []))
        if got != {args.expected_solver_hash}:
            raise SystemExit(f"solver hash mismatch: dataset carries {got}")

    cand = np.asarray(d["reference_chosen_logps"], float) - np.asarray(d["reference_rejected_logps"], float)
    prev = np.asarray(d["history0_chosen_logps"], float) - np.asarray(d["history0_rejected_logps"], float)
    h = cand - prev
    target = args.eta * np.asarray(d["nbpo_weighted_z"], float)
    half = np.asarray([assign.get(p, "unassigned") for p in d["prompt_id"]])

    unassigned = int((half == "unassigned").sum())
    if unassigned:
        raise SystemExit(f"{unassigned} rows have a prompt id absent from the split map")

    report = {
        "method": args.method, "seed": args.seed, "eta": args.eta,
        "label": args.label or args.candidate_precomputed.name,
        "candidate_precomputed": str(args.candidate_precomputed),
        "heldout_split": str(args.heldout_split),
        "definition": {
            "h": "(cand_chosen - cand_rejected) - (pi_t_chosen - pi_t_rejected), "
                 "sequence-sum log-probabilities over response tokens",
            "target": "eta * nbpo_weighted_z; eta is applied exactly once, here, "
                      "matching the trainer -- the pair artifact stores it unscaled",
            "gate_metric": "normalized_mse_var = MSE / Var(target)",
        },
        "splits": {},
    }
    for name in ("validation", "test"):
        m = metrics(h[half == name], target[half == name])
        report["splits"][name] = {"metrics": m, "gate": check(m)}
    report["selection_protocol"] = (
        "validation selects eta, learning rate and steps; the test half is "
        "reported for the gate and never used for selection")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: {"metrics": v["metrics"], "all_pass": v["gate"]["all_pass"]}
                      for k, v in report["splits"].items()}, indent=2))


if __name__ == "__main__":
    main()
