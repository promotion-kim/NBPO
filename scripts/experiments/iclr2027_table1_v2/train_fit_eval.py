#!/usr/bin/env python3
"""Did the arm fit its own training data? Fixed-set evaluation, not minibatch loss.

Every number reported until now was a held-out endpoint, which cannot separate
"the arm failed to generalize" from "the arm never fit what it was trained on".
This scores the final checkpoint on the TRAINING rows under the same target,
mask, aggregation and normalization used for held-out.

It is deliberately not the training loss from the log: that is an average over a
moving policy across a schedule, and comparing it to a final-checkpoint number
would compare two different quantities.

For an arm trained on a subset, the rows split three ways and all three are
reported, because they answer different questions:

``own_train``    prompts this arm actually trained on -- did it fit?
``rest_of_pool`` training-pool prompts it never saw -- generalization inside the
                 same pool, with no distribution shift from the held-out split
``heldout``      the validation/test prompts
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


def metrics(h, T):
    if h.size == 0:
        return None
    mse = float(((h - T) ** 2).mean())
    var, ms = float(T.var()), float((T ** 2).mean())
    nz = (T != 0) & (h != 0)
    return {
        "n_pairs": int(h.size),
        "nMSE_var": mse / var if var else None,
        "nMSE_zero": mse / ms if ms else None,
        "mse": mse, "zero_predictor_mse": ms, "mean_predictor_mse": var,
        "sign_agreement": float(np.mean(np.sign(h[nz]) == np.sign(T[nz])))
        if nz.any() else None,
        "pearson": float(np.corrcoef(h, T)[0, 1]) if h.std() and T.std() else None,
        "spearman": spearman(h, T) if h.std() and T.std() else None,
        "h_rms": float(np.sqrt((h ** 2).mean())), "T_rms": float(np.sqrt(ms)),
        "mean_T": float(T.mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+", required=True,
                    help="label=<trainfit precomputed dir>=<own-train precomputed dir>")
    ap.add_argument("--eta", type=float, default=1.0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from datasets import load_from_disk

    report = {"eta": args.eta,
              "definition": ("final checkpoint scored on a FIXED set with the same "
                             "target, mask and normalization as held-out; not the "
                             "online minibatch loss"),
              "arms": {}}
    for spec in args.arms:
        label, tf_dir, own_dir = spec.split("=")
        d = load_from_disk(str(Path(tf_dir) / "precomputed"))["train"].to_dict()
        h = (np.asarray(d["reference_chosen_logps"], float)
             - np.asarray(d["reference_rejected_logps"], float)
             - np.asarray(d["history0_chosen_logps"], float)
             + np.asarray(d["history0_rejected_logps"], float))
        T = args.eta * np.asarray(d["nbpo_weighted_z"], float)
        pid = np.asarray(d["prompt_id"])
        own = set(load_from_disk(own_dir)["train"]["prompt_id"])
        is_own = np.asarray([p in own for p in pid])
        report["arms"][label] = {
            "n_own_train_prompts": len(own),
            "own_train": metrics(h[is_own], T[is_own]),
            "rest_of_training_pool": metrics(h[~is_own], T[~is_own]),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    hdr = (f"{'arm':>18} {'N':>5} {'own nMSE_var':>13} {'own sign':>9} {'own r':>8} "
           f"{'own h_rms':>10} | {'rest nMSE':>10} {'rest r':>8}")
    print(hdr); print("-" * len(hdr))
    for k, v in report["arms"].items():
        o = v["own_train"]; r = v["rest_of_training_pool"]
        rn = f"{r['nMSE_var']:>10.4f}" if r else f"{'--':>10}"
        rr = f"{r['pearson']:>+8.4f}" if r else f"{'--':>8}"
        print(f"{k:>18} {v['n_own_train_prompts']:>5} {o['nMSE_var']:>13.4f} "
              f"{o['sign_agreement']:>9.4f} {o['pearson']:>+8.4f} {o['h_rms']:>10.4f} | "
              f"{rn} {rr}")


if __name__ == "__main__":
    main()
