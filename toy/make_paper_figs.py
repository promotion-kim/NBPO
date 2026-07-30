"""Regenerate the toy-experiment paper figures as vector PDFs.

Reuses the deterministic computation in toy_v2.py so the figure content is
identical to the current raster versions, but renders as vector PDF (sharp at
any zoom). Produces, into the target figures directory:

  ronpo_toy_curves.pdf   (single decoy game: 2x2 dynamics panels)
  ronpo_decoy_sweep.pdf  (decoy-badness sweep: 2x2 panels)
  kappa_tradeoff_avg_vs_min.pdf  (average-vs-minimum trade-off over kappa)

Fig. 1 (ronpo_lastiter_validation) is produced by validate_theory.py.

Usage:
    python make_paper_figs.py --outdir ../ronpo_aaai/figures
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import toy_v2 as T


def _panel_tag(ax, tag, text=None):
    ax.text(-0.02, 1.04, f"({tag})", transform=ax.transAxes,
            fontsize=12, fontweight="bold", ha="left", va="bottom")


def make_single(args, outdir):
    P = T.make_conflicting_preferences(
        n_actions=args.n_actions, n_objectives=args.n_objectives, seed=args.seed,
        specialist_strength=args.specialist_strength, decoy_good=args.decoy_good,
        decoy_bad=args.decoy_bad, background_strength=args.background_strength)
    ref = np.ones(args.n_actions) / args.n_actions
    histories, _ = T.run_all_methods_on_game(P, args, ref)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))

    curves = [
        (axes[0, 0], "a", "Average reference win rate", "avg_ref",
         "Average Reference Win Rate", "Average win rate against uniform reference"),
        (axes[0, 1], "b", "Minimum objective-action win rate", "worst_case",
         "Worst Objective-Action Win Rate", "Minimum win rate"),
        (axes[1, 0], "c", "Gap to robust LP optimum", "robust_gap",
         "Gap to Robust LP Optimum", "Gap to LP optimum"),
    ]
    for ax, tag, cap, metric, title, ylabel in curves:
        for name, hist in histories.items():
            ax.plot(getattr(hist, metric), label=name)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        _panel_tag(ax, tag, cap)

    ax = axes[1, 1]
    names = list(histories.keys())
    n_actions = len(histories[names[0]].policies[-1])
    x = np.arange(n_actions)
    width = 0.8 / max(len(names), 1)
    for idx, name in enumerate(names):
        ax.bar(x + idx * width, histories[name].policies[-1], width=width, label=name)
    ax.set_xticks(x + width * (len(names) - 1) / 2)
    ax.set_xticklabels([f"a{i}" for i in range(n_actions)])
    ax.set_xlabel("Action")
    ax.set_ylabel("Final policy probability")
    ax.set_title("Final Policy Distributions")
    ax.legend(fontsize=8)
    _panel_tag(ax, "d", "Final policy distributions")

    fig.tight_layout()
    out = outdir / "ronpo_toy_curves.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


def _sweep_agg(args):
    values = [float(x) for x in args.decoy_bad_values.split(",")]
    rows = []
    for b in values:
        for seed in range(args.num_seeds):
            P = T.make_conflicting_preferences(
                n_actions=args.n_actions, n_objectives=args.n_objectives,
                seed=args.seed + seed, specialist_strength=args.specialist_strength,
                decoy_good=args.decoy_good, decoy_bad=b,
                background_strength=args.background_strength)
            ref = np.ones(args.n_actions) / args.n_actions
            histories, _ = T.run_all_methods_on_game(P, args, ref)
            decoy_index = args.n_objectives
            for name, hist in histories.items():
                pi = hist.policies[-1]
                rows.append({"decoy_bad": b, "seed": seed, "method": name,
                             "worst_case": hist.worst_case[-1],
                             "robust_gap": hist.robust_gap[-1],
                             "avg_ref": hist.avg_ref[-1],
                             "decoy_mass": float(pi[decoy_index])})
    agg = T.aggregate_mean_se(rows, ["decoy_bad", "method"],
                              ["worst_case", "robust_gap", "avg_ref", "decoy_mass"])
    return values, agg


def make_decoy_sweep(args, outdir):
    values, agg = _sweep_agg(args)
    methods = sorted({a["method"] for a in agg})
    xlabel = "Decoy badness under hidden objective (lower = more dangerous decoy)"

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    panels = [
        (axes[0, 0], "a", "Average reference win rate", "avg_ref",
         "Average reference win rate", "Average reference win rate vs Decoy Badness"),
        (axes[0, 1], "b", "Final decoy mass", "decoy_mass",
         "Final decoy mass", "Final decoy mass vs Decoy Badness"),
        (axes[1, 0], "c", "Gap to robust LP optimum", "robust_gap",
         "Gap to LP optimum", "Gap to LP optimum vs Decoy Badness"),
        (axes[1, 1], "d", "Minimum win rate", "worst_case",
         "Minimum win rate", "Minimum win rate vs Decoy Badness"),
    ]
    for ax, tag, cap, metric, ylabel, title in panels:
        for method in methods:
            xs, ys, es = [], [], []
            for b in values:
                found = [a for a in agg if a["method"] == method and float(a["decoy_bad"]) == b]
                if found:
                    xs.append(b)
                    ys.append(float(found[0][f"{metric}_mean"]))
                    es.append(float(found[0][f"{metric}_se"]))
            ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=method)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        _panel_tag(ax, tag, cap)

    fig.tight_layout()
    out = outdir / "ronpo_decoy_sweep.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


def make_kappa_tradeoff(args, outdir):
    kappas = [float(x) for x in args.kappa_values.split(",")]
    ronpo_rows, base_rows = [], []
    for seed in range(args.num_seeds):
        P = T.make_conflicting_preferences(
            n_actions=args.n_actions, n_objectives=args.n_objectives,
            seed=args.seed + seed, specialist_strength=args.specialist_strength,
            decoy_good=args.decoy_good, decoy_bad=args.decoy_bad,
            background_strength=args.background_strength)
        ref = np.ones(args.n_actions) / args.n_actions
        _, robust_opt_value = T.solve_robust_optimum(P)
        base_histories, _ = T.run_all_methods_on_game(P, args, ref)
        for name in ["SPPO-avg", "INPO-avg", "MNPO-hist-avg", "HT-MNPO-mix"]:
            h = base_histories[name]
            base_rows.append({"method": name, "avg_ref": h.avg_ref[-1],
                              "worst_case": h.worst_case[-1]})
        for kappa in kappas:
            policies, _ = T.run_ronpo(P, args.n_iter, args.alpha_ronpo_pi,
                                      args.alpha_ronpo_sigma, args.tau, kappa, ref)
            h = T.record_metrics(f"RONPO-kappa={kappa:g}", policies, P, robust_opt_value, ref)
            ronpo_rows.append({"kappa": kappa, "avg_ref": h.avg_ref[-1],
                               "worst_case": h.worst_case[-1]})

    fig, ax = plt.subplots(figsize=(7, 6))
    for kappa in kappas:
        vals = [r for r in ronpo_rows if float(r["kappa"]) == kappa]
        x = np.mean([r["avg_ref"] for r in vals])
        y = np.mean([r["worst_case"] for r in vals])
        ax.scatter([x], [y])
        ax.text(x, y, f"κ={kappa:g}", fontsize=8)
    for method in ["SPPO-avg", "INPO-avg", "MNPO-hist-avg", "HT-MNPO-mix"]:
        vals = [r for r in base_rows if r["method"] == method]
        x = np.mean([r["avg_ref"] for r in vals])
        y = np.mean([r["worst_case"] for r in vals])
        ax.scatter([x], [y], marker="x")
        ax.text(x, y, method, fontsize=8)
    ax.set_xlabel("Average reference win rate")
    ax.set_ylabel("Minimum win rate")
    ax.set_title("Average-vs-Minimum Trade-off")
    fig.tight_layout()
    out = outdir / "kappa_tradeoff_avg_vs_min.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=str, required=True)
    a = p.parse_args()
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    args = T.build_parser().parse_args([])  # defaults matching the paper runs
    make_single(args, outdir)
    make_decoy_sweep(args, outdir)
    make_kappa_tradeoff(args, outdir)


if __name__ == "__main__":
    main()
