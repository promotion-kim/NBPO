#!/usr/bin/env python3
"""Reproduce the primary controlled figure from archived per-seed observations.

Run from any directory: python scripts/plot_controlled.py
Points average five independent controlled seeds at each declared alpha. The
x coordinate is mean BT deviance; vertical error bars are sample SD (ddof=1),
not standard errors or confidence intervals. Lines only connect observed means.
The non-converged legacy map and reference/solver-oracle rows remain in the CSV
for provenance, but are not plotted as competing converged methods.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METHODS = [
    ("nbpo_direct", "NBPO", "#0072B2", "o", "-"),
    ("game_ks", "Game-KS", "#009E73", "s", "--"),
    ("game_utilitarian", "Game-utilitarian", "#7B4AB2", "^", ":"),
    ("fixed_reference_nash", "Fixed-reference Nash", "#D55E00", "D", "-"),
    ("bt_rm_nash", "BT-RM–Nash", "#B9476B", "v", "--"),
    ("bt_rm_utilitarian", "BT-RM–utilitarian", "#6B6B6B", "x", ":"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=ROOT / "data/controlled_per_seed.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "figures/controlled_primary")
    args = parser.parse_args()
    with args.csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    alphas = sorted({float(r["alpha"]) for r in rows})
    if alphas != [0.0, 0.25, 0.5, 0.75, 1.0]:
        raise ValueError(f"Unexpected controlled alpha grid: {alphas}")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.labelsize": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42,
                         "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(6.2, 2.7))
    fig.subplots_adjust(left=0.115, right=0.985, bottom=0.185, top=0.73)
    ax.axhline(0, color="#777777", lw=0.65, zorder=0)
    for method, label, color, marker, linestyle in METHODS:
        xs, means, sds = [], [], []
        for alpha in alphas:
            group = [r for r in rows if r["method"] == method and float(r["alpha"]) == alpha]
            if len(group) != 5 or len({r["seed"] for r in group}) != 5:
                raise ValueError(f"Expected five unique seeds for {method}, alpha={alpha}")
            if any(r["converged"].lower() != "true" for r in group):
                raise ValueError(f"Non-converged row found for plotted method {method}")
            x = np.asarray([float(r["bt_deviance_per_edge"]) for r in group])
            y = np.asarray([float(r["normalized_min_surplus"]) for r in group])
            if not (np.isfinite(x).all() and np.isfinite(y).all()):
                raise ValueError("Non-finite plotted values")
            xs.append(x.mean()); means.append(y.mean()); sds.append(y.std(ddof=1))
        ax.errorbar(xs, means, yerr=sds, label=label, color=color,
                    marker=marker, linestyle=linestyle, lw=1.35, markersize=4,
                    markerfacecolor="white", markeredgewidth=0.9,
                    elinewidth=0.65, capsize=1.7, capthick=0.65)
    ax.set_xlabel("Bradley–Terry deviance per edge")
    ax.set_ylabel("Normalized worst surplus")
    ax.set_xlim(-0.012, 0.37)
    ax.set_xticks([0, 0.1, 0.2, 0.3])
    ax.set_ylim(-0.76, 1.09)
    ax.set_yticks([-0.5, 0, 0.5, 1.0])
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.55)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.015, 1.045), ncol=3,
              frameon=False, fontsize=7.8, columnspacing=1.15,
              handlelength=2.2, handletextpad=0.5, borderaxespad=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        fig.savefig(args.output.with_suffix(f".{extension}"), dpi=240)
    plt.close(fig)
    print(f"Rendered 6 methods × 5 alpha values × 5 seeds from {args.csv}")
    print(f"Saved {args.output}.pdf and .png; vertical bars = seed SD, ddof=1")


if __name__ == "__main__":
    main()
