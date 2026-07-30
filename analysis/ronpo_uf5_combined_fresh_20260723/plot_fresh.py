#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(); p.add_argument("--summary", type=Path, required=True); p.add_argument("--out", type=Path, required=True); a = p.parse_args()
    d = json.loads(a.summary.read_text()); m = d["methods"]
    base = m["Base"]; stages = []
    for s in range(1, 5):
        label = f"RONPO-SS-S{s}"
        if label in m: stages.append((s, label))
    for s in (5, 6, 7):
        label = f"RONPO-COMB-S{s}"
        if label in m: stages.append((s, label))
    xs = [0] + [s for s, _ in stages]
    avg = [base["average_of_objective_means"]] + [m[x]["average_of_objective_means"] for _, x in stages]
    worst = [base["worst_objective_mean"]] + [m[x]["worst_objective_mean"] for _, x in stages]
    fig, ax = plt.subplots(figsize=(4.3, 3.0))
    ax.plot(xs, avg, "-o", label="RONPO average", color="#5077bb")
    ax.plot(xs, worst, "--s", label="RONPO worst", color="#b14e8f")
    if "RMOD K=16" in m:
        ax.axhline(m["RMOD K=16"]["worst_objective_mean"], ls=":", color="#dc7c1f", label="RMOD worst")
    if "RONPO-FB-S5" in m:
        ax.scatter([5], [m["RONPO-FB-S5"]["worst_objective_mean"]], marker="x", s=45, color="#555", label="fixed-Base S5 worst")
    ax.set(xlabel="Training stage", ylabel="Raw reward", xticks=range(max(xs) + 1), title="Fresh UF5 stage trajectory")
    ax.grid(ls=":", alpha=.45); ax.legend(fontsize=7); fig.tight_layout()
    a.out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(a.out); fig.savefig(a.out.with_suffix(".png"), dpi=180)


if __name__ == "__main__": main()
