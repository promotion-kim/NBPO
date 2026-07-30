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
    methods = json.loads(a.summary.read_text())["methods"]
    labels = list(methods)
    means = np.asarray([[methods[x]["mean_by_objective"][h] for h in HEADS] for x in labels])
    lo, hi = means.min(0), means.max(0)
    normalized = (means - lo) / np.where(hi > lo, hi - lo, 1.0)
    values = dict(zip(labels, normalized))
    angles = np.linspace(0, 2 * np.pi, len(HEADS), endpoint=False).tolist()
    angles += angles[:1]

    plt.rcParams.update({"font.family": "serif", "font.size": 9, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.5, 3.1), subplot_kw={"polar": True})
    for label in ("DPO", "IPO", "SPPO", "SimPO"):
        x = values[label].tolist()
        ax.plot(angles, x + x[:1], color="#b9bcc2", lw=0.8)
    ax.plot([], [], color="#b9bcc2", lw=0.8, label="Stage-1 baselines")
    styles = (
        ("Base", "Base", "#111111", "--", 1.2),
        ("INPO-S1-PARENT", "INPO stage 1", "#777777", "-", 1.1),
        ("INPO-CONTROL-S2", "INPO continuation", "#e6952a", "-", 1.6),
        ("RONPO-FT-S2", "RONPO robust fine-tune", "#1644c8", "-", 2.0),
    )
    for key, label, color, line, width in styles:
        x = values[key].tolist()
        ax.plot(angles, x + x[:1], color=color, linestyle=line, lw=width, label=label, zorder=5)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(TICKS, fontsize=7)
    ax.set_yticklabels([])
    ax.set_title("Matched average warm start", fontsize=9, pad=12)
    ax.legend(loc="lower right", bbox_to_anchor=(1.38, -0.14), fontsize=6.1, labelspacing=0.2)
    fig.tight_layout(pad=0.3)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.output)


if __name__ == "__main__":
    main()
