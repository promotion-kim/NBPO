#!/usr/bin/env python3
"""Plot measured objective trade-offs with supplied confidence intervals.

The supplied CSV template contains only a header: no observations are invented.
One row denotes one method under a common protocol and split. Supply objective
means and their confidence-interval bounds, plus CI provenance in ci_method
(e.g., hierarchical_seed_prompt_bootstrap_95pct). All rows must share protocol,
split, and CI method. This plot does not infer a Pareto frontier or connect means.

Example after filling real measurements:
  python scripts/plot_objective_tradeoffs.py --csv data/objective_tradeoffs.csv \
      --x-label 'Helpfulness surplus' --y-label 'Harmlessness surplus' --surplus

Column contract:
protocol_id, split: immutable evaluation identifiers (judge/decoder/pool in the
protocol manifest); method: displayed label; n_seeds and n_prompts: measured
sample counts; ci_method: construction and confidence level of supplied CIs;
objective_1/2_mean, objective_1/2_ci_low/high: numeric measurement fields.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, default=ROOT / "data/objective_tradeoffs_template.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "figures/objective_tradeoffs")
    parser.add_argument("--x-label", default="Objective 1")
    parser.add_argument("--y-label", default="Objective 2")
    parser.add_argument("--surplus", action="store_true", help="Add zero-surplus reference lines")
    args = parser.parse_args()
    with args.csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit("No measurements supplied: no figure was created. Fill the CSV with measured means and confidence intervals.")
    for name in ("protocol_id", "dataset", "evaluator_id", "evaluator_revision", "split", "ci_method"):
        if len({r[name].strip() for r in rows}) != 1 or not rows[0][name].strip():
            raise ValueError(f"Rows must have one common, nonempty {name}")
    labels = [r["method"].strip() for r in rows]
    if any(not label for label in labels) or len(labels) != len(set(labels)):
        raise ValueError("Method labels must be nonempty and unique")
    points = []
    for row in rows:
        if int(row["n_seeds"]) < 1 or int(row["n_prompts"]) < 1:
            raise ValueError("Sample counts must be positive integers")
        axes = []
        for objective in ("objective_1", "objective_2"):
            mean, low, high = [float(row[objective + suffix]) for suffix in ("_mean", "_ci_low", "_ci_high")]
            if not all(math.isfinite(v) for v in (mean, low, high)) or not low <= mean <= high:
                raise ValueError(f"Invalid mean/CI for {row['method']}, {objective}")
            axes.append((mean, low, high))
        points.append(axes)
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(5.0, 3.5), constrained_layout=True)
    if args.surplus:
        ax.axhline(0, color="0.7", lw=0.7)
        ax.axvline(0, color="0.7", lw=0.7)
    for label, ((x, xl, xh), (y, yl, yh)) in zip(labels, points):
        ax.errorbar(x, y, xerr=[[x-xl], [xh-x]], yerr=[[y-yl], [yh-y]],
                    fmt="o", capsize=2, markersize=4, label=label)
    ax.set_xlabel(args.x_label); ax.set_ylabel(args.y_label)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(color="0.92", lw=0.5); ax.set_axisbelow(True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        fig.savefig(args.output.with_suffix(f".{extension}"), dpi=240)
    print(f"Plotted measured points only; split={rows[0]['split']}; CI={rows[0]['ci_method']}")


if __name__ == "__main__":
    main()
