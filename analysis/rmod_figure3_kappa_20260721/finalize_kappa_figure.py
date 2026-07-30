#!/usr/bin/env python3
import argparse, csv, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    raw = {}
    with (a.results / "per_objective_scores.csv").open() as f:
        for r in csv.DictReader(f):
            if "objective" in r:
                raw.setdefault(r["model"], {})[r["objective"]] = float(r["mean_raw_score"])
            else:
                raw[r["model"]] = {
                    "helpfulness": float(r["helpfulness_raw"]),
                    "harmlessness": float(r["harmlessness_raw"]),
                }
    colors = {"k0p01": "#1976d2", "k0p1": "#c62828", "k0p5": "#2e7d32", "k1": "#6a1b9a", "k2": "#ef6c00"}
    labels = {"k0p01": r"$\kappa=0.01$", "k0p1": r"$\kappa=0.1$", "k0p5": r"$\kappa=0.5$", "k1": r"$\kappa=1$", "k2": r"$\kappa=2$"}
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    rows = []
    for k in ("k0p01", "k0p1", "k0p5", "k1", "k2"):
        pts = [(0, raw["base"]["helpfulness"], raw["base"]["harmlessness"])] if "base" in raw else []
        for stage in range(1, 5):
            key = f"{k}_stage{stage}"
            if key not in raw:
                rows.append({"kappa": k, "stage": stage, "status": "failed_or_missing"})
                continue
            x = raw[key]["helpfulness"]; y = raw[key]["harmlessness"]
            pts.append((stage, x, y)); rows.append({"kappa": k, "stage": stage, "status": "eligible", "helpfulness": x, "harmlessness": y})
        if pts:
            ax.plot([p[1] for p in pts], [p[2] for p in pts], "-o", lw=2, ms=5, color=colors[k], label=labels[k])
            for stage, x, y in pts:
                if stage:
                    ax.annotate(str(stage), (x, y), xytext=(5, 5), textcoords="offset points", fontsize=8, color=colors[k])
    if "base" in raw:
        bx, by = raw["base"]["helpfulness"], raw["base"]["harmlessness"]
        ax.scatter([bx], [by], marker="*", s=180, color="black", label="_nolegend_", zorder=6)
        ax.annotate("Base", (bx, by), xytext=(7, -2), textcoords="offset points", fontsize=9, color="black")
    ax.set_xlabel("Helpfulness (Beaver reward)")
    ax.set_ylabel("Harmlessness (negated Beaver cost)")
    ax.set_title("SafeRLHF RONPO trajectories by adversary temperature")
    ax.grid(ls=":", alpha=.4); ax.legend(frameon=False)
    fig.tight_layout(); a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight"); fig.savefig(a.out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    (a.out.parent / "kappa_stage_points.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
