#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HEADS = ("instruction_following", "truthfulness", "honesty", "helpfulness", "safety")
TICKS = ("Instr.\nfollowing", "Truthful.", "Honesty", "Helpful.", "Safety")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    data = json.loads(a.summary.read_text())["figure5_normalized"]
    angles = np.linspace(0, 2 * np.pi, len(HEADS), endpoint=False).tolist()
    angles += angles[:1]

    plt.rcParams.update({"font.family": "serif", "font.size": 9, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.5, 3.1), subplot_kw={"polar": True})
    for label in ("INPO", "SPPO", "DPO", "IPO", "SimPO"):
        x = [data[label][h] for h in HEADS]
        ax.plot(angles, x + x[:1], color="#b9bcc2", lw=0.8)
    ax.plot([], [], color="#b9bcc2", lw=0.8, label="baselines")
    styles = (
        ("Base", "Base", "#111111", "--", 1.3),
        ("RMOD K=16", "RMOD $K{=}16$", "#e6952a", "-", 1.5),
        ("RONPO-S2", "RONPO stage 2", "#85a9ff", "-", 1.2),
        ("RONPO-COMB-S6", "RONPO stage 6", "#1644c8", "-", 2.0),
    )
    for key, label, color, linestyle, width in styles:
        x = [data[key][h] for h in HEADS]
        ax.plot(angles, x + x[:1], color=color, linestyle=linestyle, lw=width, label=label, zorder=5)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(TICKS, fontsize=7)
    ax.set_yticklabels([])
    ax.set_title("Frozen joint comparison", fontsize=9, pad=12)
    ax.legend(loc="lower right", bbox_to_anchor=(1.34, -0.12), fontsize=6.2, labelspacing=0.2)
    fig.tight_layout(pad=0.3)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.output)


if __name__ == "__main__":
    main()

