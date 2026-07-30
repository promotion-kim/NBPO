#!/usr/bin/env python3
"""Per-prompt min-max normalized Avg/Worst over the arm pool for UltraFeedback-5obj.
Reads the 5 ArmoRM per-objective files eval_<name>.jsonl per arm.
Usage: compute_uf5.py ROOT arm1 arm2 ..."""
import json, sys
import numpy as np

R = sys.argv[1]; arms = sys.argv[2:]
OBJ = ["instruction_following", "truthfulness", "honesty", "helpfulness", "safety"]

def col(arm, obj):
    rows = [json.loads(l) for l in open(f"{R}/{arm}/eval_{obj}.jsonl") if l.strip()]
    return np.array([float(r["all_rm_scores"][0]) for r in rows])

n = min(min(len(col(a, o)) for a in arms) for o in OBJ)
norms = {}
for o in OBJ:
    M = np.stack([col(a, o)[:n] for a in arms], axis=1)          # n x A
    lo = M.min(1, keepdims=True); hi = M.max(1, keepdims=True)
    norms[o] = np.where(hi > lo, (M - lo) / np.where(hi > lo, hi - lo, 1.0), 0.5)

stacked = np.stack([norms[o] for o in OBJ], axis=0)              # K x n x A
avg = stacked.mean(0); worst = stacked.min(0)                   # n x A each
print(f"UF5 n={n}, arms={arms}")
print("arm".ljust(10) + "".join(o[:5].rjust(7) for o in OBJ) + "    Avg  Worst")
rows = []
for j, a in enumerate(arms):
    per = {o: float(norms[o][:, j].mean()) for o in OBJ}
    r = dict(arm=a, **per, Avg=float(avg[:, j].mean()), Worst=float(worst[:, j].mean()))
    rows.append(r)
    print(f"{a:10}" + "".join(f"{per[o]:7.3f}" for o in OBJ) + f" {r['Avg']:6.3f} {r['Worst']:6.3f}")
json.dump(rows, open(f"{R}/uf5_avg_worst_summary.json", "w"), indent=2)
print(f"[saved] {R}/uf5_avg_worst_summary.json")
