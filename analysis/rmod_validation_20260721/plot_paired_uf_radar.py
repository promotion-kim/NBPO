#!/usr/bin/env python3
"""Plot the exact-prompt-aligned UltraFeedback RMOD comparison."""

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
DISPLAY = {"K1_reference": "Base (K=1)", "RONPO": "RONPO", "RMOD": "RMOD (K=16)"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.summary.read_text(encoding="utf-8"))
    objectives = report["protocol"]["objectives"]
    angles = np.linspace(0, 2 * np.pi, len(objectives), endpoint=False).tolist()
    angles += angles[:1]
    colors = {"K1_reference": "#222222", "RONPO": "#1f4fd6", "RMOD": "#e07b16"}
    fig, ax = plt.subplots(figsize=(5.3, 4.5), subplot_kw={"polar": True})
    for method, record in report["methods"].items():
        values = [record["mean_by_objective"][objective] for objective in objectives]
        values += values[:1]
        ax.plot(angles, values, lw=2, marker="o", ms=4,
                color=colors.get(method), label=DISPLAY.get(method, method))
        ax.fill(angles, values, color=colors.get(method), alpha=0.06)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([LABELS[objective] for objective in objectives], fontsize=8)
    ax.set_title("UltraFeedback: five-objective raw reward", fontsize=11, pad=16)
    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.10), fontsize=8)
    ax.grid(ls=":", alpha=0.45)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".png"), dpi=160, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
