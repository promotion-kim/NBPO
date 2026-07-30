#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SHORT = {"instruction_following": "Instr.\nfollow.", "truthfulness": "Truthful", "honesty": "Honest", "helpfulness": "Helpful", "safety": "Safe"}


def stages(methods, prefix):
    found = []
    for label in methods:
        match = re.fullmatch(re.escape(prefix) + r"S(\d+)", label)
        if match:
            found.append((int(match.group(1)), label))
    return [label for _, label in sorted(found)]


def main():
    p = argparse.ArgumentParser(); p.add_argument("--summary", type=Path, required=True); p.add_argument("--out", type=Path, required=True); a = p.parse_args()
    report = json.loads(a.summary.read_text()); methods = report["methods"]; objectives = report["protocol"]["objectives"]
    series = {"RONPO ": "#5077bb", "RONPO-MA-": "#159d8c", "RONPO-SS-": "#b14e8f"}
    fig = plt.figure(figsize=(7.2, 3.15)); grid = fig.add_gridspec(1, 2, width_ratios=(1.0, 1.2), wspace=0.35)
    radar = fig.add_subplot(grid[0, 0], polar=True); ax = fig.add_subplot(grid[0, 1])
    angles = list(np.linspace(0, 2*np.pi, len(objectives), endpoint=False)); closed = angles + angles[:1]
    radar_models = ["Base", "RMOD K=16", "RONPO S5", "RONPO-MA-S4", "RONPO-SS-S4"]
    colors = {"Base":"#222222", "RMOD K=16":"#dc7c1f", "RONPO S5":"#5077bb", "RONPO-MA-S4":"#159d8c", "RONPO-SS-S4":"#b14e8f"}
    for label in radar_models:
        if label not in methods: continue
        vals = [methods[label]["mean_by_objective"][o] for o in objectives]
        radar.plot(closed, vals+vals[:1], marker="o", ms=3, lw=1.8, color=colors[label], label=label)
    radar.set_ylim(0,1); radar.set_xticks(angles); radar.set_xticklabels([SHORT[o] for o in objectives], fontsize=7); radar.set_yticks([.25,.5,.75,1]); radar.tick_params(labelsize=6); radar.grid(ls=":", alpha=.45); radar.set_title("Per-objective raw reward", fontsize=9, pad=12)
    base = methods["Base"]
    for prefix, color in series.items():
        labels = stages(methods, prefix)
        if not labels: continue
        xs = [0] + list(range(1, len(labels)+1))
        avg = [base["average_of_objective_means"]] + [methods[x]["average_of_objective_means"] for x in labels]
        worst = [base["worst_objective_mean"]] + [methods[x]["worst_objective_mean"] for x in labels]
        name = prefix.strip(" -") or "RONPO"
        ax.plot(xs, avg, "-o", color=color, lw=1.8, ms=3.5, label=f"{name} avg.")
        ax.plot(xs, worst, "--s", color=color, lw=1.6, ms=3.2, label=f"{name} worst")
    if "RMOD K=16" in methods:
        ax.axhline(methods["RMOD K=16"]["worst_objective_mean"], color="#dc7c1f", ls=":", lw=1.4, label="RMOD worst")
    ax.set_xticks(range(0,6)); ax.set_xlabel("Training stage", fontsize=8); ax.set_ylabel("Raw reward", fontsize=8); ax.tick_params(labelsize=7); ax.grid(ls=":", alpha=.45); ax.set_title("Average and weakest objective", fontsize=9); ax.legend(fontsize=5.8, ncol=2)
    h,l = radar.get_legend_handles_labels(); fig.legend(h,l,ncol=3,loc="lower center",bbox_to_anchor=(.5,-.03),fontsize=6.4,frameon=False); fig.subplots_adjust(bottom=.23,top=.88,left=.05,right=.98)
    a.out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(a.out,bbox_inches="tight"); fig.savefig(a.out.with_suffix(".png"),dpi=180,bbox_inches="tight")


if __name__ == "__main__": main()
