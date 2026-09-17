#!/usr/bin/env python3
"""Compare two arms on the SAME rows, and test the difference rather than each side.

An interval on arm A that excludes zero and an interval on arm B that includes it
says nothing about whether A and B differ. The paired quantity has to be
resampled directly, which is what this does: each bootstrap replicate draws a set
of PROMPTS, recomputes both arms' metrics on those same prompts, and records the
difference. Prompts are the resampling unit because the 28 pairs of a prompt
share its responses and its target row.

Both arms must have been scored on the same held-out rows against the same
target, which the row keys are checked for rather than assumed.
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
    nz = (T != 0) & (h != 0)
    out = {"nMSE_var": float(((h - T) ** 2).mean() / T.var()),
           "nMSE_zero": float(((h - T) ** 2).mean() / (T ** 2).mean()),
           "sign_agreement": float(np.mean(np.sign(h[nz]) == np.sign(T[nz])))
           if nz.any() else np.nan,
           "h_rms": float(np.sqrt((h ** 2).mean()))}
    out["pearson"] = float(np.corrcoef(h, T)[0, 1]) if h.std() and T.std() else np.nan
    out["spearman"] = spearman(h, T) if h.std() and T.std() else np.nan
    return out


def load(path, eta):
    from datasets import load_from_disk
    d = load_from_disk(str(Path(path) / "precomputed"))["train"].to_dict()
    h = (np.asarray(d["reference_chosen_logps"], float)
         - np.asarray(d["reference_rejected_logps"], float)
         - np.asarray(d["history0_chosen_logps"], float)
         + np.asarray(d["history0_rejected_logps"], float))
    T = eta * np.asarray(d["nbpo_weighted_z"], float)
    key = [f"{p}|{c}|{r}" for p, c, r in zip(d["prompt_id"], d["chosen_response_id"],
                                             d["rejected_response_id"])]
    return h, T, np.asarray(key), np.asarray(d["prompt_id"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", required=True, help="label=<scored dir>")
    ap.add_argument("--new", required=True, help="label=<scored dir>")
    ap.add_argument("--eta", type=float, default=1.0)
    ap.add_argument("--heldout-split", type=Path, required=True)
    ap.add_argument("--half", default="test", choices=["validation", "test", "both"])
    ap.add_argument("--resamples", type=int, default=2000)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    (ol, op), (nl, npath) = (args.old.split("=", 1), args.new.split("=", 1))
    h_o, T_o, k_o, p_o = load(op, args.eta)
    h_n, T_n, k_n, p_n = load(npath, args.eta)
    if not np.array_equal(k_o, k_n):
        raise SystemExit("the two arms were not scored on the same rows in the same order")
    if not np.allclose(T_o, T_n):
        raise SystemExit("the two arms were scored against different targets")
    assign = json.loads(args.heldout_split.read_text())["assignment"]
    half = np.asarray([assign.get(p, "?") for p in p_o])

    report = {"old": ol, "new": nl, "eta": args.eta, "n_rows": int(h_o.size),
              "rows_identical": True, "targets_identical": True,
              "resampling_unit": "prompt", "resamples": args.resamples, "halves": {}}
    halves = ("validation", "test") if args.half == "both" else (args.half,)
    rng = np.random.default_rng(0)

    for name in halves:
        s = half == name
        ho, hn, T, pid = h_o[s], h_n[s], T_o[s], p_o[s]
        mo, mn = metrics(ho, T), metrics(hn, T)
        idx = {}
        for i, p in enumerate(pid):
            idx.setdefault(p, []).append(i)
        keys = list(idx)
        draws = {k: [] for k in mo}
        for _ in range(args.resamples):
            pick = np.concatenate([idx[keys[j]] for j in
                                   rng.integers(0, len(keys), size=len(keys))])
            a, b, t = ho[pick], hn[pick], T[pick]
            if t.std() == 0:
                continue
            ma, mb = metrics(a, t), metrics(b, t)
            for k in draws:
                draws[k].append(mb[k] - ma[k])
        diff = {}
        for k, v in draws.items():
            v = np.sort(np.asarray([x for x in v if np.isfinite(x)]))
            if v.size == 0:
                continue
            diff[k] = {"new_minus_old": mn[k] - mo[k],
                       "ci95_low": float(v[int(0.025 * v.size)]),
                       "ci95_high": float(v[int(0.975 * v.size)]),
                       "fraction_of_resamples_favouring_new": float(
                           (v > 0).mean() if k in ("sign_agreement", "pearson",
                                                   "spearman") else (v < 0).mean())}
        report["halves"][name] = {"n_prompts": len(keys),
                                  "old": mo, "new": mn, "paired_difference": diff}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")

    for name, r in report["halves"].items():
        print(f"\n--- {name} ({r['n_prompts']} prompts, paired over prompts) ---")
        print(f"{'metric':>16} {'old':>10} {'new':>10} {'new-old':>10} "
              f"{'95% CI of the difference':>28}")
        for k in ("nMSE_var", "nMSE_zero", "sign_agreement", "pearson", "spearman",
                  "h_rms"):
            if k not in r["paired_difference"]:
                continue
            d = r["paired_difference"][k]
            print(f"{k:>16} {r['old'][k]:>10.4f} {r['new'][k]:>10.4f} "
                  f"{d['new_minus_old']:>+10.4f} "
                  f"{'[%+.4f, %+.4f]' % (d['ci95_low'], d['ci95_high']):>28}")


if __name__ == "__main__":
    main()
