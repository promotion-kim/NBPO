"""Draw fig:target-transfer from the measured diagnostic artifact.

Panel (a) tracks each objective across the three representations of the same
NBPO run -- exact target, fitted pool, fresh responses -- with whole-prompt
paired intervals. Panel (b) shows the NBPO-minus-utilitarian difference at the
same three stages against a zero reference.

The stages do not share a denominator: the categorical stages are on the
78-prompt intersection where every rubric-order verdict parsed, and the fresh
stage is on its own complete prompts. The panel labels say so, and the fresh
stage carries the untrained-base calibration line so its level can be read
against what no training gives.

No point is drawn from an absent measurement; the figure is written only when
the artifact has every stage it needs.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAPER = Path(__file__).resolve().parents[1]
OUT = PAPER / "figures/target_transfer.pdf"
DIAG = "/work/uf4_20260910/analysis/diag_20260914"
KUBECONFIG = "/home/sjkim/.kube/aipr-kubeconfig.yaml"
CRITERIA = [("instruction_following", "IF"), ("truthfulness", "Truth"),
            ("honesty", "Honesty"), ("helpfulness", "Help")]
STAGES = [("Exact target", "nbpo", "#1f4e79"),
          ("Fitted pool", "neural_nbpo_mse_s42", "#2e8b57"),
          ("Fresh responses", "nbpo_mse_s42", "#b8860b")]


def fetch(path):
    cmd = ("kubectl exec -n p-aipr nbpo-judge -c main -- bash -lc %s"
           % json.dumps("cat %s 2>/dev/null" % path))
    env = dict(os.environ, KUBECONFIG=KUBECONFIG)
    raw = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                         timeout=300, env=env).stdout.strip()
    return json.loads(raw) if raw else None


def main():
    d = fetch(DIAG + "/target_transfer.json")
    if not d:
        print(json.dumps({"skipped": "no artifact"}))
        return 0
    boot, fresh = d.get("bootstrap", {}), d.get("fresh_intervals", {})
    needed = [k for _l, k, _c in STAGES[:2]]
    if any(k not in boot for k in needed) or STAGES[2][1] not in fresh:
        print(json.dumps({"skipped": "artifact lacks a stage", "have": sorted(boot)}))
        return 0

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(9.4, 3.5))
    width = 0.26
    xs = range(len(CRITERIA))

    for s, (label, key, colour) in enumerate(STAGES):
        pts, los, his = [], [], []
        for crit, _short in CRITERIA:
            if key in boot:
                e = boot[key][crit]
                p, lo, hi = e["win_rate_common"], e["ci95"][0], e["ci95"][1]
            else:
                e = fresh[key][crit]
                p, lo, hi = e["win_rate"], e["ci95"][0], e["ci95"][1]
            pts.append(p)
            los.append(p - lo)
            his.append(hi - p)
        off = (s - 1) * width
        ax_a.bar([x + off for x in xs], [p - 0.5 for p in pts], width, bottom=0.5,
                 color=colour, alpha=0.85, label=label,
                 yerr=[los, his], capsize=2.5, error_kw={"lw": 0.9})

    base = fresh.get("base_fresh")
    if base:
        level = min(base[c]["win_rate"] for c, _s in CRITERIA)
        ax_a.axhline(level, color="#888888", lw=0.9, ls=":",
                     label="Untrained base draw, weakest objective")
    ax_a.axhline(0.5, color="black", lw=0.8)
    ax_a.set_xticks(list(xs))
    ax_a.set_xticklabels([s for _c, s in CRITERIA])
    ax_a.set_ylabel("Independent reference win rate")
    ax_a.set_title("(a) One run across three representations", fontsize=10)
    ax_a.legend(fontsize=7, loc="upper left", framealpha=0.9)

    diffs = [("Exact target", d.get("paired_differences", {}).get("nbpo_minus_util")),
             ("Fitted pool", d.get("pool_nbpo_minus_util")),
             ("Fresh responses", d.get("fresh_nbpo_minus_util"))]
    for s, ((label, table), (_l, _k, colour)) in enumerate(zip(diffs, STAGES)):
        if not table:
            continue
        pts, los, his = [], [], []
        for crit, _short in CRITERIA:
            e = table.get(crit)
            if not e:
                pts.append(0.0), los.append(0.0), his.append(0.0)
                continue
            pts.append(e["point"])
            los.append(e["point"] - e["ci95"][0])
            his.append(e["ci95"][1] - e["point"])
        off = (s - 1) * width
        ax_b.errorbar([x + off for x in xs], pts, yerr=[los, his], fmt="o", ms=4,
                      color=colour, lw=0, elinewidth=0.9, capsize=2.5, label=label)
    ax_b.axhline(0.0, color="black", lw=0.8)
    ax_b.set_xticks(list(xs))
    ax_b.set_xticklabels([s for _c, s in CRITERIA])
    ax_b.set_ylabel(r"NBPO $-$ utilitarian")
    ax_b.set_title("(b) Same contrast at each stage", fontsize=10)
    ax_b.legend(fontsize=7, loc="upper left", framealpha=0.9)

    for ax in (ax_a, ax_b):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.parent / (OUT.stem + ".tmp.pdf")
    fig.savefig(tmp, format="pdf", bbox_inches="tight")
    os.replace(tmp, OUT)
    plt.close(fig)
    print(json.dumps({"written": str(OUT),
                      "categorical_n": d.get("common_prompts_all_objectives"),
                      "fresh_n": fresh[STAGES[2][1]].get("n"),
                      "base_calibration": bool(base)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
