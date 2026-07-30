"""Task-3 plot: how the Nash convergence point moves as the ADVERSARY
temperature varies (RONPO kappa, RMOD lambda), the analog of RMOD Fig 3c.
Two panels: (a) worst-objective reward vs the adversary hyperparameter;
(b) the Nash point in the (mean quality, safety) trade-off plane, one
trajectory per method, since safety trades off against the four quality
objectives. Base is the reference. All methods scored by the same ArmoRM heads.

  python pareto_plot.py --scored_dir DIR --out_prefix P \
    --ronpo k0p01=0.01 k0p05=0.05 k0p2=0.2 k1=1.0 ... \
    --rmod  l0p1=0.1 l0p5=0.5 l1p0=1.0 l5p0=5.0 l10p0=10.0
Tags map to files: ronpo -> ronpo_gemma_<tag>_<obj>.jsonl, rmod -> rmod_<tag>_<obj>.jsonl.
"""
import argparse, json, os
import numpy as np

OBJS = ["instruction_following", "truthfulness", "honesty", "helpfulness", "safety"]
QUALITY = OBJS[:4]


def means(scored_dir, prefix, tag):
    m = {}
    for o in OBJS:
        p = os.path.join(scored_dir, f"{prefix}{tag}_{o}.jsonl")
        if not os.path.exists(p):
            return None
        v = [float(np.mean(json.loads(l)["all_rm_scores"])) for l in open(p) if l.strip()]
        m[o] = float(np.mean(v))
    m["worst"] = min(m[o] for o in OBJS)
    m["quality"] = float(np.mean([m[o] for o in QUALITY]))
    m["safety"] = m["safety"]
    return m


def parse_pairs(items):
    out = []
    for it in items:
        tag, val = it.split("=")
        out.append((tag, float(val)))
    return sorted(out, key=lambda x: x[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored_dir", required=True)
    ap.add_argument("--out_prefix", required=True)
    ap.add_argument("--ronpo", nargs="*", default=[], help="tag=kappa ...")
    ap.add_argument("--rmod", nargs="*", default=[], help="tag=lambda ...")
    ap.add_argument("--base_tag", default="base128")
    args = ap.parse_args()

    base = means(args.scored_dir, "", args.base_tag)
    series = {}
    for name, prefix, items in [("RONPO", "ronpo_gemma_", args.ronpo), ("RMOD", "rmod_", args.rmod)]:
        pts = []
        for tag, val in parse_pairs(items):
            m = means(args.scored_dir, prefix, tag)
            if m: pts.append((val, m))
        series[name] = pts

    # dump CSV
    with open(args.out_prefix + ".csv", "w") as f:
        f.write("method,hyperparam,quality,safety,worst,avg_all\n")
        if base:
            f.write(f"Base,,{base['quality']:.4f},{base['safety']:.4f},{base['worst']:.4f},"
                    f"{np.mean([base[o] for o in OBJS]):.4f}\n")
        for name, pts in series.items():
            for val, m in pts:
                f.write(f"{name},{val},{m['quality']:.4f},{m['safety']:.4f},{m['worst']:.4f},"
                        f"{np.mean([m[o] for o in OBJS]):.4f}\n")
    for name, pts in series.items():
        print(name, [(v, round(m["worst"], 3)) for v, m in pts])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    col = {"RONPO": "#1f77b4", "RMOD": "#ff7f0e"}
    # (a) worst vs hyperparameter
    for name, pts in series.items():
        if not pts: continue
        xs = [v for v, _ in pts]; ys = [m["worst"] for _, m in pts]
        ax[0].plot(xs, ys, "o-", color=col[name], label=name)
    if base: ax[0].axhline(base["worst"], ls="--", color="gray", label="Base")
    ax[0].set_xscale("log"); ax[0].set_xlabel(r"adversary temperature ($\kappa$ RONPO / $\lambda$ RMOD)")
    ax[0].set_ylabel("worst-objective reward"); ax[0].legend(fontsize=8); ax[0].set_title("(a) Nash worst-case vs adversary temperature")
    # (b) quality-safety trade-off plane
    for name, pts in series.items():
        if not pts: continue
        qs = [m["quality"] for _, m in pts]; ss = [m["safety"] for _, m in pts]
        ax[1].plot(qs, ss, "o-", color=col[name], label=name, alpha=0.85)
        for (v, m) in pts:
            ax[1].annotate(f"{v:g}", (m["quality"], m["safety"]), fontsize=6, alpha=0.7)
    if base: ax[1].plot(base["quality"], base["safety"], "*", ms=14, color="k", label="Base")
    ax[1].set_xlabel("mean quality reward (IF/truth/honesty/help)"); ax[1].set_ylabel("safety reward")
    ax[1].legend(fontsize=8); ax[1].set_title("(b) Nash point in the quality-safety plane")
    fig.tight_layout()
    fig.savefig(args.out_prefix + ".pdf", bbox_inches="tight")
    fig.savefig(args.out_prefix + ".png", dpi=140, bbox_inches="tight")
    print(f"[pareto] wrote {args.out_prefix}.pdf/.png/.csv")


if __name__ == "__main__":
    main()
