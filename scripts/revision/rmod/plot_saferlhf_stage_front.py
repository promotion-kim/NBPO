"""SafeRLHF helpfulness-harmlessness plane: RONPO stage 1-4 trajectory advancing
from the base policy toward the Pareto frontier, with final-stage baselines as
reference. Mirrors RMOD Fig. 3(b); RONPO's advance axis is training stages
(no decode-time overhead) rather than the candidate count K.
"""
import argparse, json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# raw Beaver reward (helpful) and negated Beaver cost (harmless), 1000-prompt panel
RONPO = [("Base", -3.007, 10.882), ("1", 2.241, 11.975), ("2", 4.907, 11.489),
         ("3", 5.738, 11.755), ("4", 6.614, 10.907)]
BASELINES = [("IPO", 3.370, 12.382), ("DPO", 2.635, 12.599), ("SimPO", 1.951, 13.749),
             ("INPO", 2.131, 12.461), ("SPPO", 2.739, 10.016),
             ("HT-MNPO (help.)", 4.200, 8.725), ("HT-MNPO (harm.)", 3.648, 9.077)]


def load_series(path):
    if not path:
        return None
    pts = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        pts.append((str(r["k"]), r["helpful"], r["harmless"]))
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rmod", default=None, help="jsonl of {k,helpful,harmless} for RMOD K-sweep")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    # RONPO trajectory
    xs = [p[1] for p in RONPO]; ys = [p[2] for p in RONPO]
    ax.plot(xs, ys, "-o", color="#1f4fd6", lw=2.0, ms=7, zorder=5, label="RONPO (stages 1-4)")
    for lab, x, y in RONPO:
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(6, 6),
                    fontsize=9, color="#1f4fd6", fontweight="bold")
    for i in range(len(xs) - 1):
        ax.annotate("", xy=(xs[i + 1], ys[i + 1]), xytext=(xs[i], ys[i]),
                    arrowprops=dict(arrowstyle="->", color="#1f4fd6", lw=1.6))
    # RMOD K-sweep (optional)
    rmod = load_series(args.rmod)
    if rmod:
        rx = [p[1] for p in rmod]; ry = [p[2] for p in rmod]
        ax.plot(rx, ry, "-s", color="#e07b16", lw=2.0, ms=6, zorder=4,
                label="RMOD (same-decoder K sweep)")
        for lab, x, y in rmod:
            text = "reference (K=1)" if lab == "1" else f"K={lab}"
            ax.annotate(text, (x, y), textcoords="offset points", xytext=(6, -10),
                        fontsize=8, color="#e07b16")
        if rmod[0][0] == "1":
            ax.scatter([rmod[0][1]], [rmod[0][2]], marker="*", s=160,
                       facecolors="none", edgecolors="#e07b16", linewidths=1.5, zorder=6)
    # base star + baselines
    ax.scatter([RONPO[0][1]], [RONPO[0][2]], marker="*", s=230, color="black", zorder=6)
    for lab, x, y in BASELINES:
        ax.scatter([x], [y], marker="^", s=42, color="#888", zorder=3)
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(5, -3), fontsize=7, color="#555")
    ax.set_xlabel("Helpfulness (Beaver reward)")
    ax.set_ylabel("Harmlessness (negated Beaver cost)")
    ax.set_title("SafeRLHF: helpfulness--harmlessness trade-off", fontsize=11)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    ax.grid(True, ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(args.out.replace(".pdf", ".png"), dpi=160, bbox_inches="tight")
    print("[stage-front] wrote", args.out)


if __name__ == "__main__":
    main()
