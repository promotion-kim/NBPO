#!/usr/bin/env python3
"""The prompt-count generalization curve: does more training data help or hurt?

Every arm on this curve got the SAME gradient budget (1200 updates), the SAME
canonical target, the SAME learning rate and the SAME held-out set. Only the
number of TRAINING PROMPTS differs, so the x axis is the one thing that varies
and the epochs-per-pair fall out of it -- 85 epochs at 8 prompts, 0.98 at 700.

Reading it: if held-out correlation rises with prompt count, the earlier failure
was a data-per-prompt problem. If it falls, per-prompt targets do not transfer
and adding prompts at fixed compute actively costs. Either answer is a result;
the point of fixing compute is that neither can be confused with "trained
longer".
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gates", nargs="+", required=True,
                    help="n_prompts=<gate.json> entries")
    ap.add_argument("--surplus", type=Path, default=None,
                    help="held_out_surplus.json covering the same arms")
    ap.add_argument("--rows-per-prompt", type=int, default=28)
    ap.add_argument("--updates", type=int, default=1200)
    ap.add_argument("--effective-batch", type=int, default=16)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--tex", type=Path, default=None)
    args = ap.parse_args()

    surplus = json.loads(args.surplus.read_text())["arms"] if args.surplus else {}

    pts = []
    for spec in args.gates:
        n, path = spec.split("=", 1)
        n = int(n)
        g = json.loads(Path(path).read_text())
        rows = n * args.rows_per_prompt
        pts.append({
            "n_prompts": n,
            "n_pairs": rows,
            "epochs": args.updates * args.effective_batch / rows,
            "label": g["label"],
            "validation": g["splits"]["validation"]["metrics"],
            "test": g["splits"]["test"]["metrics"],
            "surplus": surplus.get(g["label"], {}),
        })
    pts.sort(key=lambda p: p["n_prompts"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "generalization_curve.json").write_text(json.dumps(pts, indent=2) + "\n")

    hdr = (f"{'prompts':>8} {'pairs':>7} {'epochs':>7} {'val nMSE':>9} {'test nMSE':>10} "
           f"{'test sign':>10} {'test r':>8} {'test rho':>9} {'test worst s':>13}")
    print(hdr); print("-" * len(hdr))
    for p in pts:
        s = p["surplus"].get("test", {}).get("worst_objective_mean_surplus")
        print(f"{p['n_prompts']:>8} {p['n_pairs']:>7} {p['epochs']:>7.2f} "
              f"{p['validation']['normalized_mse_var']:>9.4f} "
              f"{p['test']['normalized_mse_var']:>10.4f} "
              f"{p['test']['sign_agreement']:>10.4f} {p['test']['pearson']:>+8.4f} "
              f"{p['test']['spearman']:>+9.4f} "
              f"{(f'{s:+.5f}' if s is not None else 'n/a'):>13}")

    if args.tex:
        L = [r"\begin{tabular}{rrrrrrr}", r"\toprule",
             r"train prompts & pairs & epochs & test nMSE & sign agr. & Pearson & "
             r"worst surplus\\", r"\midrule"]
        for p in pts:
            s = p["surplus"].get("test", {}).get("worst_objective_mean_surplus")
            L.append(" & ".join([
                f"${p['n_prompts']}$", f"${p['n_pairs']}$", f"${p['epochs']:.2f}$",
                f"${p['test']['normalized_mse_var']:.3f}$",
                f"${p['test']['sign_agreement']:.3f}$",
                f"${p['test']['pearson']:+.3f}$",
                (f"${s:+.4f}$" if s is not None else "--")]) + r"\\")
        L += [r"\bottomrule", r"\end{tabular}"]
        args.tex.parent.mkdir(parents=True, exist_ok=True)
        args.tex.write_text("\n".join(L) + "\n")
        print(f"\nwrote {args.tex}")

    # the figure: two panels sharing the prompt-count axis
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [p["n_prompts"] for p in pts]
    fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.6))
    ax[0].plot(xs, [p["test"]["pearson"] for p in pts], "o-", label="test")
    ax[0].plot(xs, [p["validation"]["pearson"] for p in pts], "s--", label="validation")
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_ylabel("Pearson$(h,\\ T)$ on held-out prompts")
    ax[0].legend(fontsize=8)
    ax[1].plot(xs, [p["test"]["normalized_mse_var"] for p in pts], "o-", label="test")
    ax[1].plot(xs, [p["validation"]["normalized_mse_var"] for p in pts], "s--",
               label="validation")
    ax[1].axhline(1.0, color="r", lw=0.9, ls=":", label="$h=0$")
    ax[1].axhline(0.90, color="g", lw=0.9, ls="--", label="gate")
    ax[1].set_ylabel("normalized MSE")
    ax[1].legend(fontsize=8)
    for a in ax:
        a.set_xscale("log")
        a.set_xlabel("training prompts (equal gradient budget: 1200 updates)")
        a.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.out_dir / "generalization_curve.pdf")
    fig.savefig(args.out_dir / "generalization_curve.png", dpi=170)
    print(f"wrote {args.out_dir}/generalization_curve.pdf")


if __name__ == "__main__":
    main()
