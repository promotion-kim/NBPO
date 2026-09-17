#!/usr/bin/env python3
"""Collect IFEval/GSM8K into the numbers tab:general-capability prints.

Reports, per method, the seed mean and sample standard deviation, every per-seed
estimate, and the per-seed paired prompt-bootstrap interval against the base --
the interval is the one written beside the responses, never recomputed here.

It also records the base rescoring spread. The official IFEval language check
calls a language detector whose seed this harness does not pin, so rescoring one
fixed response set moves strict-prompt accuracy by a prompt or two; the paper
quotes the frozen first scoring and discloses the spread.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

METHODS = {"nbpo": "nbpo", "util": "game_utilitarian"}
SEEDS = (42, 43, 44)
METRIC = {"ifeval": "strict_prompt_accuracy", "gsm8k": "exact_match"}


def load(root, label, bench, version, who):
    path = root / "evaluations" / f"deterministic_{label}_v{version}" / f"{bench}_{who}.summary.json"
    return json.loads(path.read_text())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("/work/nbpo_repair_20260909"))
    ap.add_argument("--version", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    report = {"frozen_base": {}, "methods": {}, "base_rescoring_spread": {}}
    frozen = args.root / "evaluations" / "base_deterministic_v1"
    for bench, metric in METRIC.items():
        s = json.loads((frozen / f"{bench}_base.summary.json").read_text())
        report["frozen_base"][bench] = s["metrics"][metric]["estimate"]

    for short, name in METHODS.items():
        entry = {"seeds": list(SEEDS)}
        for bench, metric in METRIC.items():
            vals, deltas, base_vals = [], [], []
            for seed in SEEDS:
                label = f"uf4_{short}_mse_s{seed}"
                s = load(args.root, label, bench, args.version, label)
                b = load(args.root, label, bench, args.version, "base")
                vals.append(s["metrics"][metric]["estimate"])
                base_vals.append(b["metrics"][metric]["estimate"])
                d = s["paired_delta_vs_base"][metric]
                deltas.append({"seed": seed, "estimate": d["estimate"], "ci95": d["ci95"],
                               "excludes_zero": d["ci95"][0] > 0 or d["ci95"][1] < 0})
            entry[bench] = {
                "per_seed": vals,
                "mean": st.mean(vals),
                "sample_sd": st.stdev(vals),
                "mean_minus_frozen_base": st.mean(vals) - report["frozen_base"][bench],
                "paired_delta_vs_base": deltas,
                "n_seeds_excluding_zero": sum(d["excludes_zero"] for d in deltas),
            }
            spread = report["base_rescoring_spread"].setdefault(bench, {"values": []})
            spread["values"].extend(base_vals)
        report["methods"][name] = entry

    for bench, spread in report["base_rescoring_spread"].items():
        vs = sorted(set(spread["values"]))
        spread["distinct_values"] = vs
        spread["range"] = vs[-1] - vs[0]
        spread["deterministic"] = len(vs) == 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    for name, entry in report["methods"].items():
        for bench in METRIC:
            e = entry[bench]
            print(f"{name:18s} {bench:7s} {e['mean']:.4f} +/- {e['sample_sd']:.4f} "
                  f"(base {report['frozen_base'][bench]:.4f}, "
                  f"{e['n_seeds_excluding_zero']}/3 seeds exclude zero)")


if __name__ == "__main__":
    main()
