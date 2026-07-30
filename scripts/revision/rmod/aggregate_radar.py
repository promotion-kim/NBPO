"""Aggregate the ArmoRM 5-head scored files into the Task-1 radar: per method,
the mean reward on each of the 5 objectives plus the worst-case objective, and
a radar-chart PDF. Every method is scored by the same rm_armo_multihead run, so
the axes are comparable.

  python aggregate_radar.py --scored_dir <dir> --out_prefix <path> \
     --methods base_ref=Base rmod_l0p5=RMOD ronpo_gemma=RONPO ...

Each method M has files <scored_dir>/M_<obj>.jsonl with `all_rm_scores` per
prompt (one or more responses). We reduce multiple responses by mean.
"""
import argparse, json, os
import numpy as np

OBJS = ["instruction_following", "truthfulness", "honesty", "helpfulness", "safety"]


def method_means(scored_dir, tag):
    per_obj = {}
    for o in OBJS:
        path = os.path.join(scored_dir, f"{tag}_{o}.jsonl")
        if not os.path.exists(path):
            return None
        vals = []
        for line in open(path):
            s = json.loads(line)["all_rm_scores"]
            if s:
                vals.append(float(np.mean(s)))
        per_obj[o] = float(np.mean(vals)) if vals else float("nan")
    return per_obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored_dir", required=True)
    ap.add_argument("--out_prefix", required=True)
    ap.add_argument("--methods", nargs="+", required=True, help="tag=Label ...")
    args = ap.parse_args()

    rows = {}
    for spec in args.methods:
        tag, label = spec.split("=", 1)
        m = method_means(args.scored_dir, tag)
        if m is None:
            print(f"[skip] {tag} (missing scored files)")
            continue
        m["worst"] = min(m[o] for o in OBJS)
        m["avg"] = float(np.mean([m[o] for o in OBJS]))
        rows[label] = m

    # CSV
    cols = OBJS + ["avg", "worst"]
    with open(args.out_prefix + ".csv", "w") as f:
        f.write("method," + ",".join(cols) + "\n")
        for label, m in rows.items():
            f.write(label + "," + ",".join(f"{m[c]:.4f}" for c in cols) + "\n")
    print("methods:", list(rows))
    for label, m in rows.items():
        print(f"  {label:16s} worst={m['worst']:.4f} avg={m['avg']:.4f} " +
              " ".join(f"{o[:4]}={m[o]:.3f}" for o in OBJS))

    # radar
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ang = np.linspace(0, 2 * np.pi, len(OBJS), endpoint=False).tolist()
    ang += ang[:1]
    fig, axp = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    for label, m in rows.items():
        v = [m[o] for o in OBJS]
        v += v[:1]
        axp.plot(ang, v, linewidth=1.8, label=label)
        axp.fill(ang, v, alpha=0.08)
    axp.set_xticks(ang[:-1])
    axp.set_xticklabels([o.replace("_", "\n") for o in OBJS], fontsize=9)
    axp.set_title("UltraFeedback 5-objective ArmoRM reward", fontsize=11, pad=18)
    axp.legend(loc="upper right", bbox_to_anchor=(1.28, 1.10), fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out_prefix + ".pdf", bbox_inches="tight")
    fig.savefig(args.out_prefix + ".png", dpi=140, bbox_inches="tight")
    print(f"[radar] wrote {args.out_prefix}.pdf/.png/.csv")


if __name__ == "__main__":
    main()
