#!/usr/bin/env python3
"""Downstream target stability: does the judge's noise survive into the target?

Judge-level agreement statistics are a proxy. What actually matters is whether
two independent halves of the judge evidence produce the *same finite-pool
training target*, because that is the object the policy is fit to. This builds
two tensors from disjoint halves of the observations, runs the identical
finite-pool solve on each, and compares everything downstream.

Two splits, answering different questions:

``order``     forward-order observations versus reverse-order observations,
              each mapped to the same semantic orientation first. This isolates
              position sensitivity as it reaches the target.
``template``  template ``t0`` versus ``{t1, t2}``. This isolates prompt-wording
              sensitivity. Only pairs scored under both are usable, so on an
              adaptively-adjudicated bank this subset is the *unstable* pairs by
              construction -- the script reports the subset size so that bias is
              visible rather than assumed away.

Both halves are restricted to prompts whose tensor is *complete* in both halves.
A cell missing from one half would otherwise be silently imputed by whatever the
tensor builder defaults to, and the comparison would measure imputation.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from mnpo_scripts.nbpo_core import uniform_policy
from mnpo_scripts.nbpo_generic import solve_finite_pool
from mnpo_scripts.nbpo_representations import (
    AdaptiveGameRepresentation,
    FixedReferenceRepresentation,
)

FORWARD, REVERSE = "learner_first", "comparator_first"


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.size < 3 or x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    def rank(v):
        v = np.asarray(v, float)
        o = v.argsort()
        r = np.empty_like(o, dtype=float)
        r[o] = np.arange(len(v))
        return r
    return pearson(rank(x), rank(y))


def half_delta(obs, mode, which):
    """Delta from one half of the observations, both mapped to learner orientation."""
    if mode == "order":
        sel = [o for o in obs if o["presentation_order"] ==
               (FORWARD if which == "a" else REVERSE)]
    else:
        tids = sorted({o["template_id"] for o in obs})
        if len(tids) < 2:
            return None
        keep = {tids[0]} if which == "a" else set(tids[1:])
        sel = [o for o in obs if o["template_id"] in keep]
    if not sel:
        return None
    # semantic_score is already the learner's win probability in BOTH orders
    return sum(o["semantic_score"] for o in sel) / len(sel) - 0.5


def build_tensors(rows, meta, mode, which, objectives):
    """(A_policy, A_ref, prompt_ids) from one half, keeping only complete prompts."""
    cross, ref = defaultdict(dict), defaultdict(dict)
    for r in rows:
        if not r.get("valid"):
            continue
        m = meta.get(r["pair_id"])
        if m is None:
            continue
        d = half_delta(r.get("observations") or [], mode, which)
        if d is None:
            continue
        key = (m["objective"], m["prompt_id"])
        if m["pool"] == "policy":
            cross[key][(m["learner_id"], m["comparator_id"])] = d
        else:
            ref[key][(m["learner_id"], m["comparator_id"])] = d

    lids = sorted({k[0] for v in cross.values() for k in v})
    cids = sorted({k[1] for v in cross.values() for k in v})
    I, J, K = len(lids), len(cids), len(objectives)
    need_cross, need_ref = I * J, J * (J - 1) // 2
    prompts = sorted({p for (o, p) in cross
                      if all(len(cross.get((ob, p), {})) == need_cross
                             and len(ref.get((ob, p), {})) == need_ref
                             for ob in objectives)})
    if not prompts:
        return None, None, [], lids, cids
    li = {v: i for i, v in enumerate(lids)}
    ci = {v: i for i, v in enumerate(cids)}
    A = np.zeros((K, len(prompts), I, J))
    R = np.zeros((K, len(prompts), J, J))
    for k, ob in enumerate(objectives):
        for x, p in enumerate(prompts):
            for (a, b), v in cross[(ob, p)].items():
                A[k, x, li[a], ci[b]] = v
            for (a, b), v in ref[(ob, p)].items():
                i, j = ci[a], ci[b]
                R[k, x, i, j] = v
                R[k, x, j, i] = -v
    return A, R, prompts, lids, cids


def solve(A, R, beta, eta, M, Rfp, kind):
    mu = uniform_policy(A.shape[1], A.shape[3])
    At, Rt = torch.from_numpy(A), torch.from_numpy(R)
    if kind == "adaptive_game":
        rep = AdaptiveGameRepresentation(At, Rt, mu,
                                         torch.full((A.shape[0],), beta, dtype=torch.float64))
    else:
        rep = FixedReferenceRepresentation(At, Rt, mu)
    return solve_finite_pool(rep, "nash", eta=eta, M=M, R=Rfp, gamma=0.5)


def tv(p, q):
    return 0.5 * float(np.abs(p - q).sum())


def compare(sa, sb, A_a, A_b, objectives):
    da, db = A_a.reshape(-1), A_b.reshape(-1)
    la = (torch.log(sa.pi) - torch.log(sa.pi_t)).numpy()
    lb = (torch.log(sb.pi) - torch.log(sb.pi_t)).numpy()
    # pairwise target: the quantity the trainer actually regresses on
    ta, tb = [], []
    I = la.shape[1]
    for i, j in itertools.combinations(range(I), 2):
        ta.append(la[:, i] - la[:, j])
        tb.append(lb[:, i] - lb[:, j])
    ta, tb = np.concatenate(ta), np.concatenate(tb)
    pa, pb = sa.pi.numpy(), sb.pi.numpy()
    tvs = sorted(tv(pa[x], pb[x]) for x in range(pa.shape[0]))
    ra, rb = pa.argsort(axis=1), pb.argsort(axis=1)
    rank_agree = float(np.mean([spearman(ra[x], rb[x]) or 0.0 for x in range(pa.shape[0])]))
    return {
        "delta_pearson": pearson(da, db),
        "delta_spearman": spearman(da, db),
        "target_log_ratio_pearson": pearson(ta, tb),
        "target_log_ratio_spearman": spearman(ta, tb),
        "target_sign_agreement": float(np.mean(np.sign(ta) == np.sign(tb))),
        "per_objective_value_difference": {o: float(sa.V[k] - sb.V[k])
                                           for k, o in enumerate(objectives)},
        "per_objective_surplus_difference": {o: float(sa.surplus[k] - sb.surplus[k])
                                             for k, o in enumerate(objectives)},
        "policy_tv_median": tvs[len(tvs) // 2],
        "policy_tv_p90": tvs[int(0.9 * (len(tvs) - 1))],
        "candidate_ranking_agreement_spearman": rank_agree,
        "n_prompts": int(pa.shape[0]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--objectives", default="instruction_following,truthfulness,honesty,helpfulness")
    ap.add_argument("--beta", type=float, default=0.25)
    ap.add_argument("--eta", type=float, default=1.0)
    ap.add_argument("--dual-iterations", type=int, default=2000)
    ap.add_argument("--fixed-point-iterations", type=int, default=3)
    args = ap.parse_args()

    objectives = args.objectives.split(",")
    meta = {r["pair_id"]: r for r in read_jsonl(args.pairs)}
    rows = read_jsonl(args.results)

    out = {"config": {"beta": args.beta, "eta": args.eta, "M": args.dual_iterations,
                      "R": args.fixed_point_iterations, "objectives": objectives},
           "splits": {}}
    for mode in ("order", "template"):
        Aa, Ra, pa, lids, cids = build_tensors(rows, meta, mode, "a", objectives)
        Ab, Rb, pb, _, _ = build_tensors(rows, meta, mode, "b", objectives)
        if Aa is None or Ab is None:
            out["splits"][mode] = {"status": "no complete prompt in both halves"}
            continue
        common = sorted(set(pa) & set(pb))
        ia = [pa.index(p) for p in common]
        ib = [pb.index(p) for p in common]
        Aa, Ra = Aa[:, ia], Ra[:, ia]
        Ab, Rb = Ab[:, ib], Rb[:, ib]
        entry = {"n_prompts_complete_in_both_halves": len(common),
                 "learner_ids": lids, "comparator_ids": cids}
        for kind, label in (("adaptive_game", "NBPO"),
                            ("fixed_reference", "Fixed-reference Nash")):
            sa = solve(Aa, Ra, args.beta, args.eta, args.dual_iterations,
                       args.fixed_point_iterations, kind)
            sb = solve(Ab, Rb, args.beta, args.eta, args.dual_iterations,
                       args.fixed_point_iterations, kind)
            entry[label] = compare(sa, sb, Aa, Ab, objectives)
        out["splits"][mode] = entry
        print(f"[{mode}] {len(common)} prompts complete in both halves", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "target_stability.json").write_text(json.dumps(out, indent=2))

    GATES = {"forward_vs_reverse_target_pearson_min": 0.90,
             "forward_vs_reverse_target_sign_agreement_min": 0.85,
             "template_split_half_target_pearson_min": 0.90,
             "template_split_half_target_sign_agreement_min": 0.85}
    fails = []
    for mode, gp, gs in (("order", "forward_vs_reverse_target_pearson_min",
                          "forward_vs_reverse_target_sign_agreement_min"),
                         ("template", "template_split_half_target_pearson_min",
                          "template_split_half_target_sign_agreement_min")):
        e = out["splits"].get(mode, {})
        for label in ("NBPO", "Fixed-reference Nash"):
            m = e.get(label)
            if not m:
                fails.append(f"{mode}/{label}: not measured")
                continue
            if (m["target_log_ratio_pearson"] or -1) < GATES[gp]:
                fails.append(f"{mode}/{label}: target Pearson "
                             f"{m['target_log_ratio_pearson']:.3f} < {GATES[gp]}")
            if m["target_sign_agreement"] < GATES[gs]:
                fails.append(f"{mode}/{label}: target sign agreement "
                             f"{m['target_sign_agreement']:.3f} < {GATES[gs]}")
    out["gates"] = GATES
    out["gate_failures"] = fails
    out["gates_passed"] = not fails
    (args.out_dir / "target_stability.json").write_text(json.dumps(out, indent=2))

    L = ["# Downstream target stability", "",
         f"Gates: **{'PASS' if not fails else 'FAIL'}**", ""]
    for f in fails:
        L.append(f"- FAIL: {f}")
    for mode, e in out["splits"].items():
        if "status" in e:
            L += ["", f"## {mode} split", "", e["status"]]
            continue
        L += ["", f"## {mode} split — {e['n_prompts_complete_in_both_halves']} prompts "
              "complete in both halves", "",
              "| method | Delta r | Delta rho | target r | target rho | sign agree | "
              "TV median | TV p90 | rank agree |", "|---|---|---|---|---|---|---|---|---|"]
        for label in ("NBPO", "Fixed-reference Nash"):
            m = e[label]
            f = lambda v: "n/a" if v is None else f"{v:.3f}"
            L.append(f"| {label} | {f(m['delta_pearson'])} | {f(m['delta_spearman'])} | "
                     f"**{f(m['target_log_ratio_pearson'])}** | "
                     f"{f(m['target_log_ratio_spearman'])} | "
                     f"**{f(m['target_sign_agreement'])}** | {m['policy_tv_median']:.4f} | "
                     f"{m['policy_tv_p90']:.4f} | "
                     f"{f(m['candidate_ranking_agreement_spearman'])} |")
        L += ["", "Per-objective differences (half A minus half B):", ""]
        for label in ("NBPO", "Fixed-reference Nash"):
            m = e[label]
            L.append(f"- {label}: value "
                     + ", ".join(f"{o} {v:+.4f}" for o, v in
                                 m["per_objective_value_difference"].items())
                     + "; surplus "
                     + ", ".join(f"{o} {v:+.4f}" for o, v in
                                 m["per_objective_surplus_difference"].items()))
    (args.out_dir / "target_stability.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    raise SystemExit(0 if not fails else 1)


if __name__ == "__main__":
    main()
