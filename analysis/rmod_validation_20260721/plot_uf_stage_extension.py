#!/usr/bin/env python3
"""Plot locked and preregistered UltraFeedback RONPO stage trajectories."""
import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SHORT = {
    "instruction_following": "Instr.\nfollow.", "truthfulness": "Truthful",
    "honesty": "Honest", "helpfulness": "Helpful", "safety": "Safe",
}
SERIES = (
    ("RONPO", re.compile(r"RONPO(?: |-)S(\d+)$"), "#5077bb"),
    ("RONPO-MA", re.compile(r"RONPO-MA-S(\d+)$"), "#159d8c"),
    ("RONPO-SS", re.compile(r"RONPO-SS-S(\d+)$"), "#b14e8f"),
)


def first(methods, *labels):
    return next((x for x in labels if x in methods), None)


def stage_series(methods, pattern):
    rows = [(int(m.group(1)), label) for label in methods if (m := pattern.fullmatch(label))]
    return sorted(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    report = json.loads(a.summary.read_text())
    methods, objectives = report["methods"], report["protocol"]["objectives"]
    base_label = first(methods, "Base", "Base (K=1)")
    rmod_label = first(methods, "RMOD K=16", "RMOD (K=16)", "RMOD")
    if not base_label:
        raise RuntimeError("Base row missing")
    trajectories = [(name, stage_series(methods, regex), color) for name, regex, color in SERIES]

    fig = plt.figure(figsize=(7.2, 3.15))
    grid = fig.add_gridspec(1, 2, width_ratios=(1.0, 1.2), wspace=0.35)
    radar, ax = fig.add_subplot(grid[0, 0], polar=True), fig.add_subplot(grid[0, 1])
    angles = list(np.linspace(0, 2*np.pi, len(objectives), endpoint=False)); closed = angles + angles[:1]
    radar_rows = [(base_label, "#222222")]
    if rmod_label:
        radar_rows.append((rmod_label, "#dc7c1f"))
    radar_rows += [(rows[-1][1], color) for _, rows, color in trajectories if rows]
    for label, color in radar_rows:
        vals = [methods[label]["mean_by_objective"][o] for o in objectives]
        radar.plot(closed, vals + vals[:1], marker="o", ms=3, lw=1.8, color=color, label=label)
    radar.set_ylim(0, 1); radar.set_xticks(angles)
    radar.set_xticklabels([SHORT[o] for o in objectives], fontsize=7)
    radar.set_yticks([.25, .5, .75, 1]); radar.tick_params(labelsize=6)
    radar.grid(ls=":", alpha=.45); radar.set_title("Per-objective raw reward", fontsize=9, pad=12)

    base = methods[base_label]
    xmax = 0
    for name, rows, color in trajectories:
        if not rows:
            continue
        xs = [0] + [stage for stage, _ in rows]; xmax = max(xmax, max(xs))
        avg = [base["average_of_objective_means"]] + [methods[x]["average_of_objective_means"] for _, x in rows]
        worst = [base["worst_objective_mean"]] + [methods[x]["worst_objective_mean"] for _, x in rows]
        ax.plot(xs, avg, "-o", color=color, lw=1.8, ms=3.5, label=f"{name} avg.")
        ax.plot(xs, worst, "--s", color=color, lw=1.6, ms=3.2, label=f"{name} worst")
    if rmod_label:
        ax.axhline(methods[rmod_label]["worst_objective_mean"], color="#dc7c1f", ls=":", lw=1.4,
                   label="RMOD worst")
    ax.set_xticks(range(xmax + 1)); ax.set_xlabel("Training stage", fontsize=8)
    ax.set_ylabel("Raw reward", fontsize=8); ax.tick_params(labelsize=7); ax.grid(ls=":", alpha=.45)
    ax.set_title("Average and weakest objective", fontsize=9); ax.legend(fontsize=5.8, ncol=2)
    handles, labels = radar.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="lower center", bbox_to_anchor=(.5, -.03),
               fontsize=6.4, frameon=False)
    fig.subplots_adjust(bottom=.23, top=.88, left=.05, right=.98)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight")
    fig.savefig(a.out.with_suffix(".png"), dpi=180, bbox_inches="tight")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
