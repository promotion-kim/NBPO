#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LABELS = {
    "instruction_following": "Instruction\nfollowing",
    "truthfulness": "Truthfulness",
    "honesty": "Honesty",
    "helpfulness": "Helpfulness",
    "safety": "Safety",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    report = json.loads(a.summary.read_text())
    methods = report["methods"]
    objectives = report["protocol"]["objectives"]
    base = methods["Base"]["mean_by_objective"]

    stages = []
    for stage in range(1, 5):
        stages.append((stage, methods[f"RONPO-SS-S{stage}"]))
    for stage in (5, 6):
        stages.append((stage, methods[f"RONPO-COMB-S{stage}"]))

    angles = np.linspace(0, 2 * np.pi, len(objectives), endpoint=False)
    closed = np.r_[angles, angles[0]]
    rmin, rmax = -0.06, 0.025
    shift = -rmin

    def delta(row):
        values = [row["mean_by_objective"][o] - base[o] for o in objectives]
        return np.r_[values, values[0]] + shift

    fig, ax = plt.subplots(figsize=(4.25, 3.6), subplot_kw={"polar": True})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    base_ring = np.full(len(closed), shift)
    ax.plot(closed, base_ring, color="#222222", lw=2.0, ls="--", label="Base")
    ax.plot(closed, delta(methods["RMOD K=16"]), color="#d55e00", lw=2.0,
            marker="o", ms=2.8, label="RMOD")

    colors = plt.cm.Blues(np.linspace(0.38, 0.95, len(stages)))
    for color, (stage, row) in zip(colors, stages):
        values = delta(row)
        ax.plot(closed, values, color=color, lw=1.3 if stage < 6 else 2.3,
                marker="o", ms=2.2, label=f"RONPO-S{stage}")
        if stage == 6:
            ax.fill(closed, values, color=color, alpha=0.08)

    ax.set_ylim(0, rmax - rmin)
    ticks = [-0.04, -0.02, 0.0, 0.02]
    ax.set_yticks([x + shift for x in ticks])
    ax.set_yticklabels([f"{x:+.2f}" for x in ticks], fontsize=6.5)
    ax.set_rlabel_position(180)
    ax.tick_params(axis="y", pad=-4)
    ax.set_xticks(angles)
    ax.set_xticklabels([LABELS[o] for o in objectives], fontsize=7.5)
    ax.grid(ls=":", alpha=0.5)
    ax.set_title("Per-objective raw-reward change from Base", fontsize=9.5, pad=14)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=4,
              fontsize=6.6, frameon=False, handlelength=2.2, columnspacing=1.0)
    fig.subplots_adjust(left=0.08, right=0.92, top=0.88, bottom=0.25)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight")
    fig.savefig(a.out.with_suffix(".png"), dpi=220, bbox_inches="tight")


if __name__ == "__main__":
    main()
