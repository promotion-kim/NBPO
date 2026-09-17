"""Aggregate cross-play verdicts into the matrix, the minima, and plot coordinates.

Definitions are the contract's, and both vulnerability definitions are computed
so the manuscript cannot quietly pick whichever flatters a row:

  M_ab = N^-1 sum_x K_x^-1 sum_k P_k(y_a > y_b | x)      measured one orientation,
                                                          order-balanced, M_ba = 1 - M_ab
  B_a  = min_{b != a} M_ab                                within-bank vulnerability
  C_a  = min_b M_ab with M_aa fixed at .5                 common-comparator variant
  W_a  = min_b N^-1 sum_x min_k P_k,x(a, b)               local worst-item statistic

A cell counts only when both orders parsed for every item of that prompt, so the
matrix has one completeness mask rather than a per-cell denominator. The
bootstrap resamples whole prompts with the full bank together and recomputes
every minimum inside each replicate, as required.

Coordinates are emitted as PGFPlots matrix-plot text for the figure, and the
caller sets the ready flag only after they exist.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

SUB = Path("/work/sub_20260914")


def file_hash(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as s:
        for c in iter(lambda: s.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="uw_xplay")
    ap.add_argument("--items", type=int, default=4)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260917)
    ap.add_argument("--labels", nargs="+", default=None,
                    help="display labels in bank order")
    args = ap.parse_args()

    root = SUB / "crossplay" / args.tag
    settings = json.loads((root / "settings.json").read_text())
    bank = settings["bank"]
    index = {arm: i for i, arm in enumerate(bank)}

    # (a, b, prompt, item) -> {order: value_for_a}
    obs = defaultdict(dict)
    statuses = defaultdict(int)
    for line in (root / "verdicts.jsonl").open():
        if not line.strip():
            continue
        r = json.loads(line)
        statuses[r["status"]] += 1
        if r["status"] not in ("ok", "identical_text"):
            continue
        obs[(r["a"], r["b"], r["prompt_id"], r["item"])][r["order"]] = float(r["value_for_a"])

    prompts = sorted({k[2] for k in obs})
    pairs = sorted({(k[0], k[1]) for k in obs})
    # per (pair, prompt): the K item probabilities, only when every item has both orders
    cell = {}
    incomplete = 0
    for a, b in pairs:
        for pid in prompts:
            values = []
            for k in range(args.items):
                got = obs.get((a, b, pid, k))
                if not got or 0 not in got or 1 not in got:
                    values = None
                    break
                values.append(0.5 * (got[0] + got[1]))
            if values is None:
                incomplete += 1
                continue
            cell[(a, b, pid)] = np.array(values)
    usable = sorted({pid for (_, _, pid) in cell})
    complete = [pid for pid in usable
                if all((a, b, pid) in cell for a, b in pairs)]

    def matrices(sample):
        """M, and the per-prompt worst-item means, over a list of prompt ids."""
        n = len(bank)
        M = np.full((n, n), np.nan)
        W = np.full((n, n), np.nan)
        for a, b in pairs:
            arr = np.stack([cell[(a, b, pid)] for pid in sample])   # (prompts, items)
            m = float(arr.mean())
            M[index[a], index[b]] = m
            M[index[b], index[a]] = 1.0 - m
            w = float(arr.min(axis=1).mean())
            W[index[a], index[b]] = w
            W[index[b], index[a]] = float((1.0 - arr).min(axis=1).mean())
        return M, W

    def minima(M, W):
        n = len(bank)
        B, C, Wm = [], [], []
        for i in range(n):
            row = [M[i, j] for j in range(n) if j != i]
            B.append(min(row))
            C.append(min(row + [0.5]))
            Wm.append(min(W[i, j] for j in range(n) if j != i))
        return np.array(B), np.array(C), np.array(Wm)

    M, W = matrices(complete)
    B, C, Wm = minima(M, W)
    rng = np.random.default_rng(args.seed)
    draws = {"B": [], "C": [], "W": []}
    for _ in range(args.bootstrap):
        sample = [complete[i] for i in rng.integers(0, len(complete), len(complete))]
        Mb, Wb = matrices(sample)
        b, c, w = minima(Mb, Wb)
        draws["B"].append(b); draws["C"].append(c); draws["W"].append(w)

    def ci(name, point):
        arr = np.array(draws[name])
        lo, hi = np.percentile(arr, [2.5, 97.5], axis=0)
        return [{"arm": bank[i], "value": round(float(point[i]), 4),
                 "ci95": [round(float(lo[i]), 4), round(float(hi[i]), 4)]}
                for i in range(len(bank))]

    labels = args.labels or bank
    coords = []
    for i in range(len(bank)):
        for j in range(len(bank)):
            # a matrix plot needs the full grid; the diagonal is not measured
            # self-play, so it is nan -- an empty cell -- and never .5
            meta = "nan" if i == j else "%.4f" % M[i, j]
            coords.append("(%d,%d) [%s]" % (j, i, meta))
    plot = ("\\addplot[matrix plot*,mesh/cols=%d,point meta=explicit] "
            "coordinates {%s};" % (len(bank), " ".join(coords)))

    report = {
        "tag": args.tag, "bank": bank, "labels": labels,
        "prompts_seen": len(prompts), "prompts_usable": len(usable),
        "prompts_complete_across_all_pairs": len(complete),
        "incomplete_pair_prompt_cells": incomplete,
        "status_counts": dict(statuses),
        "matrix_definition": ("M_ab = mean over prompts of the item mean of "
                              "P_k(a > b); one orientation measured with order "
                              "balancing, M_ba = 1 - M_ab, diagonal not measured"),
        "matrix": {bank[i]: {bank[j]: (None if i == j else round(float(M[i, j]), 4))
                             for j in range(len(bank))} for i in range(len(bank))},
        "within_bank_vulnerability": ci("B", B),
        "common_comparator_variant": ci("C", C),
        "common_comparator_note": ("self value fixed at .5, so this statistic is capped "
                                   "at .5 for any row that loses to no one"),
        "local_worst_item": ci("W", Wm),
        "bootstrap": {"replicates": args.bootstrap,
                      "unit": "whole prompt, full bank together, every minimum recomputed"},
        "absent_from_bank": ("PROSPER: its own leave-two-out estimator is not implemented, "
                             "and the contract forbids putting the earlier max-min "
                             "adaptation under that name"),
        "verdicts_sha256": file_hash(root / "verdicts.jsonl"),
        "source_sha256": file_hash(__file__),
    }
    (root / "matrix.json").write_text(json.dumps(report, indent=1) + "\n")
    (root / "crossplay_plot.tex").write_text(plot + "\n")
    print(json.dumps({k: report[k] for k in
                      ("prompts_complete_across_all_pairs", "status_counts",
                       "within_bank_vulnerability", "local_worst_item")}, indent=1)[:1100])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
