"""Assemble the cross-play matrices and the regularized bank diagnostic.

Reads the per-pair judgment files, builds one win-rate matrix per rubric, and
computes the three main-text statistics for every bank policy:

    W_ref^min(pi) = min_k W_k(pi, base)
    W_B^min(pi)   = min over k and over q != pi of W_k(pi, q)
    s_k^B(pi)     = V_k^B(pi) - V_k^B(base),  min over k reported

with V_k^B the regularized finite-bank value from app:crossplay. Its inner
minimisation has a closed form,

    V_k^B(pi) = -beta_k * log sum_q a_q exp(-J_k(pi,q) / beta_k),

which was checked against direct constrained optimisation to 1e-17 before being
used here rather than trusted from the algebra.

Uncertainty is the whole-prompt paired bootstrap the manuscript specifies: a
replicate resamples PROMPTS, carries every policy pair, rubric and presentation
order for a resampled prompt together, and recomputes both minima and the bank
solve inside the replicate rather than bootstrapping a minimum of point
estimates.

Two conventions are applied and recorded rather than silently chosen. The
diagonal is 0.5, which the recorded antisymmetry convention forces rather than
measures: W(B,A) = 1 - W(A,B) at A = B gives W = 0.5 exactly. And because a
policy appearing in its own comparator bank would otherwise score itself,
W_B^min excludes the self cell; the bank prior a stays uniform over all four, so
it is shared by every evaluated policy as the contract requires, and the
exclude-self variant of the bank value is reported alongside as a sensitivity.

This diagnostic regularizes over COMPARATOR WEIGHTS, not over response
distributions. It is not an estimate of the population game value or of
population exploitability, and it is never compared with training-teacher
surplus.
"""
from __future__ import annotations

import argparse, itertools, json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/work/uf4_20260910")
CRITERIA = ("instruction_following", "truthfulness", "honesty", "helpfulness")
PAIRS = ROOT / "evaluation/crossplay/pairs"


def bank_value(J, a, beta):
    """-beta log sum_q a_q exp(-J_q/beta), stabilised. J may be (..., Q)."""
    x = -np.asarray(J, dtype=np.float64) / beta
    m = np.max(x, axis=-1, keepdims=True)
    return -beta * (np.squeeze(m, -1) + np.log(np.sum(np.asarray(a) * np.exp(x - m), axis=-1)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference", default="base")
    ap.add_argument("--beta", type=float, default=0.25)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--out", default=str(ROOT / "analysis/crossplay_summary.json"))
    args = ap.parse_args()

    # ---- load every judged pair -----------------------------------------
    per_prompt = defaultdict(dict)          # (criterion, pid) -> {(a,b): value_for_a}
    policies, pair_reports = set(), {}
    for d in sorted(PAIRS.glob("*__vs__*")):
        complete = d / "complete.json"
        if not complete.exists():
            continue
        rep = json.loads(complete.read_text())
        a, b = rep["policy_a"], rep["policy_b"]
        policies.update((a, b))
        pair_reports[(a, b)] = rep
        got = defaultdict(dict)
        with (d / "judgments.jsonl").open() as stream:
            for line in stream:
                r = json.loads(line)
                if r["status"] == "ok":
                    got[(r["criterion"], r["prompt_id"])][r["order"]] = r["value_for_a"]
        for key, orders in got.items():
            if len(orders) == 2:                 # both presentation orders required
                per_prompt[key][(a, b)] = 0.5 * (orders[0] + orders[1])
    if not pair_reports:
        print(json.dumps({"status": "no judged pairs yet"})); return 0

    order = sorted(policies)
    needed = set(itertools.combinations(order, 2))
    missing = sorted(needed - set(pair_reports))
    # a prompt counts only when every judged pair and every rubric has it
    prompts = sorted({pid for (_, pid) in per_prompt}
                     - {pid for (c, pid), v in per_prompt.items()
                        if len(v) != len(pair_reports)}
                     - {pid for pid in {p for (_, p) in per_prompt}
                        if any((c, pid) not in per_prompt for c in CRITERIA)})
    if not prompts:
        print(json.dumps({"status": "no prompt is complete across every judged pair"})); return 0

    # ---- per-prompt tensor: (P, K, N, N) win rate of row over column -----
    idx = {p: i for i, p in enumerate(order)}
    P, K, N = len(prompts), len(CRITERIA), len(order)
    W = np.full((P, K, N, N), 0.5, dtype=np.float64)     # diagonal fixed by convention
    for pi, pid in enumerate(prompts):
        for ki, c in enumerate(CRITERIA):
            for (a, b), v in per_prompt[(c, pid)].items():
                W[pi, ki, idx[a], idx[b]] = v
                W[pi, ki, idx[b], idx[a]] = 1.0 - v      # recorded antisymmetry

    a_prior = np.full(N, 1.0 / N)
    ref = idx[args.reference]

    def statistics(Wm):
        """Wm: (K, N, N) mean win rates -> the three reported statistics."""
        J = Wm - 0.5
        V = bank_value(J, a_prior, args.beta)                      # (K, N)
        s = V - V[:, ref][:, None]
        off = ~np.eye(N, dtype=bool)
        out = {}
        for name, i in idx.items():
            wb = min(Wm[k, i, j] for k in range(K) for j in range(N) if j != i)
            out[name] = {"W_ref_min": float(min(Wm[k, i, ref] for k in range(K))),
                         "W_bank_min": float(wb),
                         "min_s_bank": float(s[:, i].min()),
                         "s_bank_per_objective": {c: float(s[k, i]) for k, c in enumerate(CRITERIA)}}
        # sensitivity: bank value with the self cell removed from the support
        for name, i in idx.items():
            keep = [j for j in range(N) if j != i]
            Ji = J[:, i, keep]
            Vi = bank_value(Ji, np.full(len(keep), 1.0 / len(keep)), args.beta)
            keep_ref = [j for j in range(N) if j != ref]
            Vr = bank_value(J[:, ref, keep_ref], np.full(len(keep_ref), 1.0 / len(keep_ref)),
                            args.beta)
            out[name]["min_s_bank_excluding_self"] = float((Vi - Vr).min())
        return out

    point = statistics(W.mean(axis=0))
    rng = np.random.default_rng(20260912)
    draws = defaultdict(lambda: defaultdict(list))
    for _ in range(args.bootstrap):
        pick = rng.integers(0, P, size=P)
        rep = statistics(W[pick].mean(axis=0))
        for name, vals in rep.items():
            for stat in ("W_ref_min", "W_bank_min", "min_s_bank"):
                draws[name][stat].append(vals[stat])

    summary = {}
    for name in order:
        summary[name] = dict(point[name])
        for stat in ("W_ref_min", "W_bank_min", "min_s_bank"):
            arr = np.array(draws[name][stat])
            summary[name][stat + "_ci95"] = [float(np.quantile(arr, .025)),
                                             float(np.quantile(arr, .975))]

    matrices = {c: {r: {q: float(W[:, k, idx[r], idx[q]].mean()) for q in order}
                    for r in order}
                for k, c in enumerate(CRITERIA)}

    report = {
        "policies": order, "reference": args.reference,
        "n_common_prompts": len(prompts),
        "pairs_judged": ["%s__vs__%s" % p for p in sorted(pair_reports)],
        "pairs_missing_for_full_bank": ["%s__vs__%s" % p for p in missing],
        "bank_prior": "uniform over all %d bank policies, shared by every evaluated policy" % N,
        "beta_k": args.beta,
        "diagonal": ("0.5, forced by the recorded antisymmetry convention W(B,A)=1-W(A,B) "
                     "at A=B; not measured by self-play"),
        "W_bank_min_excludes_self": True,
        "bootstrap": ("%d whole-prompt paired replicates; every policy pair, rubric and "
                      "presentation order for a resampled prompt moves together, and both "
                      "minima and the bank solve are recomputed inside each replicate"
                      % args.bootstrap),
        "diagnostic_scope": ("the bank value regularizes over comparator weights, not over "
                             "response distributions; it estimates neither the population game "
                             "value nor population exploitability, and is not comparable with "
                             "training-teacher surplus"),
        "statistics": summary, "matrices": matrices,
        "judge": {k: pair_reports[sorted(pair_reports)[0]][k] for k in ("judge", "judge_revision")},
    }
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"n_common_prompts": len(prompts), "policies": order,
                      "pairs_judged": len(pair_reports), "missing": missing,
                      "statistics": {n: {k: round(v, 4) for k, v in s.items()
                                         if isinstance(v, float)}
                                     for n, s in summary.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
