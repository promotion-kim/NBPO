"""Cross-objective conflict on the same development panel, separated from cycles.

Two different things are often called conflict and are measured separately here.

* Between objectives: do the objectives disagree about which candidate is best?
  Measured by Spearman rank correlation between objectives over each prompt's
  eight candidates, by top-candidate agreement, and by how often the candidate
  maximizing one objective still clears the pool mean of another.

* Within an objective: does the teacher's own preference relation cycle? The
  weak no-Condorcet rate over each prompt's learner pool answers that and is
  reported with its denominator, not merged into the number above.

Candidate values come from the teacher's cross block against the shared
comparator pool, which is the quantity the solver optimizes.
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path("/work/uf4_20260910")
DIAG = ROOT / "analysis/diag_20260914"
TARGET = "nash_v1"
CRIT = ("instruction_following", "truthfulness", "honesty", "helpfulness")


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def main():
    panel = json.loads((DIAG / "panel_dev200.json").read_text())
    ids = panel["panel_prompt_ids"]
    meta = json.loads((ROOT / "targets" / TARGET / "dev/tensor/meta.json").read_text())
    index = {pid: k for k, pid in enumerate(meta["prompt_ids"])}
    A = np.load(ROOT / "targets" / TARGET / "dev/tensor/tensor_policy.npz")["A"]

    rows = [index[pid] for pid in ids if pid in index]
    # candidate value: mean margin against the shared comparator pool
    values = A[:, rows].mean(axis=-1)               # objectives x prompts x learners

    pairs = {}
    for a, b in combinations(range(len(CRIT)), 2):
        rho = [spearman(values[a, k], values[b, k]) for k in range(values.shape[1])]
        top_a = values[a].argmax(axis=1)
        top_b = values[b].argmax(axis=1)
        clears = float(np.mean([values[b, k, top_a[k]] > values[b, k].mean()
                                for k in range(values.shape[1])]))
        pairs["%s|%s" % (CRIT[a], CRIT[b])] = {
            "n_prompts": int(values.shape[1]),
            "spearman_mean": float(np.nanmean(rho)),
            "spearman_median": float(np.nanmedian(rho)),
            "spearman_p10": float(np.nanpercentile(rho, 10)),
            "spearman_negative_fraction": float(np.mean(np.array(rho) < 0)),
            "spearman_below_minus_half_fraction": float(np.mean(np.array(rho) < -0.5)),
            "top_candidate_agreement": float(np.mean(top_a == top_b)),
            "argmax_of_first_clears_second_pool_mean": clears,
        }

    # within-objective cycles on the learner pool, from the learner-vs-learner
    # relation the teacher induces through the shared comparator pool
    cycles = {}
    for o, crit in enumerate(CRIT):
        no_condorcet = 0
        triangles = 0
        cyclic_triangles = 0
        for k in range(values.shape[1]):
            v = values[o, k]
            beats = v[:, None] > v[None, :]
            # a Condorcet winner beats every other candidate
            has_winner = any(all(beats[i, j] for j in range(8) if j != i) for i in range(8))
            no_condorcet += int(not has_winner)
            for i, j, m in combinations(range(8), 3):
                triangles += 1
                cyc = (beats[i, j] and beats[j, m] and beats[m, i]) or \
                      (beats[j, i] and beats[m, j] and beats[i, m])
                cyclic_triangles += int(cyc)
        cycles[crit] = {
            "prompts": int(values.shape[1]),
            "weak_no_condorcet_rate": no_condorcet / values.shape[1],
            "triangles": triangles,
            "cyclic_triangle_rate": cyclic_triangles / triangles,
            "note": ("values are scalar means against the shared comparator pool, so a strict "
                     "order exists unless two values tie exactly; this measures the induced "
                     "relation, not the full pairwise tensor"),
        }

    # target separation between the three aggregation rules on this panel
    tv = {}
    for name in ("util_l1matched_v1", "fixedref_nash_v1"):
        base = np.load(ROOT / "targets" / TARGET / "dev/solver/pi_star.npz")["pi"]
        other = np.load(ROOT / "targets" / name / "dev/solver/pi_star.npz")["pi"]
        idx2 = {pid: k for k, pid in enumerate(
            json.loads((ROOT / "targets" / name / "dev/tensor/meta.json").read_text())["prompt_ids"])}
        shared = [(index[pid], idx2[pid]) for pid in ids if pid in index and pid in idx2]
        vals = [0.5 * np.abs(base[i] - other[j]).sum() for i, j in shared]
        tv["nash_vs_" + name] = {"n": len(vals), "mean_tv": float(np.mean(vals)),
                                 "max_tv": float(np.max(vals))}

    payload = {"panel_sha256": panel["panel_sha256"], "target_set": TARGET,
               "objective_pairs": pairs, "within_objective": cycles, "target_tv": tv}
    (DIAG / "conflict_report.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
