#!/usr/bin/env python3
"""Multi-seed UF5 significance. For each seed we min-max normalize per objective
over that seed's pool {nbpo, os, uniform, base}, then report each method's
per-seed Avg/Worst, the mean +/- SD across seeds, and a paired bootstrap on the
Worst metric pooled over (seed, prompt) pairs. Usage: compute_uf5_seeds.py ROOT"""
import json, sys
import numpy as np

R = sys.argv[1]
OBJ = ["instruction_following", "truthfulness", "honesty", "helpfulness", "safety"]
METHODS = ["nbpo", "os", "uniform"]
SEEDS = [42, 43, 44]

def evdir(method, seed):
    return f"{R}/eval/{method}" if seed == 42 else f"{R}/eval/s{seed}_{method}"

def col(d, o):
    return np.array([float(json.loads(l)["all_rm_scores"][0]) for l in open(f"{d}/eval_{o}.jsonl") if l.strip()])

# per seed: normalize over {nbpo,os,uniform,base} and get per-prompt worst per method
worst_by_seed = {}   # seed -> dict(method -> n-vector of per-prompt worst)
per_seed = {m: {"Avg": [], "Worst": []} for m in METHODS}
for seed in SEEDS:
    arms = METHODS + ["base"]
    dirs = {m: evdir(m, seed) for m in METHODS}; dirs["base"] = f"{R}/eval/base"
    n = min(min(len(col(dirs[a], o)) for a in arms) for o in OBJ)
    norms = {}
    for o in OBJ:
        M = np.stack([col(dirs[a], o)[:n] for a in arms], 1)
        lo, hi = M.min(1, keepdims=True), M.max(1, keepdims=True)
        norms[o] = np.where(hi > lo, (M - lo) / np.where(hi > lo, hi - lo, 1.0), 0.5)
    stack = np.stack([norms[o] for o in OBJ], 0)          # K x n x A
    avg, worst = stack.mean(0), stack.min(0)              # n x A
    worst_by_seed[seed] = {m: worst[:, j] for j, m in enumerate(arms)}
    for j, m in enumerate(METHODS):
        per_seed[m]["Avg"].append(float(avg[:, j].mean()))
        per_seed[m]["Worst"].append(float(worst[:, j].mean()))

print(f"UF5 multi-seed (seeds={SEEDS})")
print(f"{'method':10} {'Avg mean+-SD':>18} {'Worst mean+-SD':>18}")
for m in METHODS:
    a, w = np.array(per_seed[m]["Avg"]), np.array(per_seed[m]["Worst"])
    print(f"{m:10} {a.mean():>8.4f}+-{a.std(ddof=1):.4f}   {w.mean():>8.4f}+-{w.std(ddof=1):.4f}")

# pooled per-(seed,prompt) paired bootstrap on Worst: NBPO vs uniform, NBPO vs os
rng = np.random.default_rng(0)
for base_m in ["uniform", "os"]:
    diffs = np.concatenate([worst_by_seed[s]["nbpo"] - worst_by_seed[s][base_m] for s in SEEDS])
    bs = np.array([diffs[rng.integers(0, len(diffs), len(diffs))].mean() for _ in range(4000)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    sig = "SIGNIFICANT" if lo > 0 or hi < 0 else "n.s."
    print(f"Worst  NBPO - {base_m:8}: {diffs.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  {sig}  (pooled n={len(diffs)})")
