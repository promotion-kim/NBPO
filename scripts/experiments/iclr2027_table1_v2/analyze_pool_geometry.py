#!/usr/bin/env python3
"""Choose the pool geometry on validation stability and compute -- nothing else.

Two candidate geometries, 4+4 and 8+8, where 4+4 is a strict subset of the 8+8
draw. For each we report what the pool looks like, what the frozen oracle does on
it, and how stable the finite-pool target is under oracle noise.

**Target stability** is measured by splitting the ENSEMBLE, not the pool: build
the tensor from one seed, and again from the mean of the other two, solve both
with the direct finite-pool solver, and correlate the resulting log-ratio
targets. Averaged over the three leave-one-out pairings. That asks exactly the
question geometry selection should ask -- if the oracle had been slightly
different, would the training target have been the same?

Downstream policy performance is never consulted.
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

OBJECTIVES = ("helpfulness", "harmlessness")


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def pearson(x, y):
    x, y = np.asarray(x, float).ravel(), np.asarray(y, float).ravel()
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    r = lambda v: np.argsort(np.argsort(np.asarray(v, float).ravel())).astype(float)
    return pearson(r(x), r(y))


def solve_target(A_pol, A_ref, beta, eta, dual_tol, workers):
    """One direct finite-pool solve; returns the target and its diagnostics."""
    from mnpo_scripts.nbpo_core import uniform_policy
    from mnpo_scripts.nbpo_generic import solve_finite_pool
    from mnpo_scripts.nbpo_representations import AdaptiveGameRepresentation
    K, X, I, J = A_pol.shape
    rep = AdaptiveGameRepresentation(
        torch.from_numpy(A_pol), torch.from_numpy(A_ref), uniform_policy(X, I),
        torch.full((K,), float(beta), dtype=torch.float64))
    t0 = time.time()
    sol = solve_finite_pool(rep, "nash", eta=eta, R=1, M=800, inner_solver="exact",
                            dual_solver="root", dual_tol=dual_tol,
                            inner_workers=workers)
    target = (torch.log(sol.pi) - torch.log(sol.pi_t)).numpy()
    return target, {
        "seconds": time.time() - t0,
        "dual_converged": bool(sol.dual_converged),
        "projected_kkt_residual": sol.projected_kkt_residual,
        "inner_extra_map_residual": sol.extra_map_residual,
        "target_identity_residual": sol.target_log_ratio_check(),
        "dual_evaluations": sol.outer_iterations_used,
        "weight_l1": float(sol.weights.sum()),
        "min_surplus": float(sol.surplus.min()),
        "surplus": [float(v) for v in sol.surplus],
    }


def pool_stats(rows, n):
    """What the pool itself looks like, before any model touches it."""
    by = defaultdict(lambda: {"learner": {}, "comparator": {}})
    for r in rows:
        by[r["prompt_id"]][r["role"]][r["sample_index"]] = r
    lens, dup, uniq_frac, tot_pairs = [], 0, [], 0
    for pid, blk in by.items():
        texts = ([blk["learner"][i]["response"] for i in range(n)]
                 + [blk["comparator"][i]["response"] for i in range(n)])
        lens += [len(t) for t in texts]
        u = len(set(texts))
        uniq_frac.append(u / len(texts))
        for a, b in itertools.combinations(range(len(texts)), 2):
            tot_pairs += 1
            if texts[a] == texts[b]:
                dup += 1
    lens = np.asarray(lens, float)
    return {
        "n_prompts": len(by), "responses_per_prompt": 2 * n,
        "exact_duplicate_pair_rate": dup / max(1, tot_pairs),
        "mean_distinct_fraction": float(np.mean(uniq_frac)),
        "response_length_chars": {
            "mean": float(lens.mean()), "sd": float(lens.std()),
            "p10": float(np.percentile(lens, 10)),
            "p50": float(np.percentile(lens, 50)),
            "p90": float(np.percentile(lens, 90))},
    }


def oracle_stats(P, n):
    """P has shape (seed, K, X, m, m); slice to the geometry and describe it."""
    p = P[:, :, :, :n, :n]
    mean = p.mean(axis=0)
    return {
        "predictive_entropy": float(np.mean(
            -(np.clip(mean, 1e-9, 1) * np.log(np.clip(mean, 1e-9, 1))
              + np.clip(1 - mean, 1e-9, 1) * np.log(np.clip(1 - mean, 1e-9, 1))))),
        "ensemble_sd_mean": float(p.std(axis=0).mean()),
        "ensemble_sd_p95": float(np.percentile(p.std(axis=0), 95)),
        "saturation_below_0.02_or_above_0.98": float(
            np.mean((mean < 0.02) | (mean > 0.98))),
        "saturation_below_0.05_or_above_0.95": float(
            np.mean((mean < 0.05) | (mean > 0.95))),
        "mean_abs_margin": float(np.mean(np.abs(mean - 0.5))),
    }


def conflict_and_cycles(P_policy, P_ref, n):
    """Objective conflict, and MODEL-PREDICTED cycles. Never human-observed.

    Conflict is read off the POLICY block: the same (learner, comparator) pair
    judged on two objectives, which is exactly a disagreement between objectives.

    Cycles must NOT be read off that block. ``P_policy[i, j]`` is
    ``P(learner_i > comparator_j)``: rows and columns index different response
    sets, so ``M[i,j], M[j,m], M[m,i]`` are three unrelated comparisons and
    "closing" them is meaningless. Cycles are counted on the **reference block**,
    ``P(comparator_i > comparator_j)``, which is a genuine tournament on one set.

    A chance-level rate is 0.25 -- three independent signs agree with probability
    2/8 -- so a rate near 0.25 on a pool of same-model samples means the margins
    carry no information, not that preferences are cyclic. The mean absolute
    margin is reported beside the rate so the two readings can be told apart.
    """
    mean = P_policy[:, :, :, :n, :n].mean(axis=0)
    s = np.sign(mean - 0.5)
    live = (s[0] != 0) & (s[1] != 0)
    conflict = float((s[0][live] != s[1][live]).mean()) if live.any() else None

    ref = P_ref[:, :, :, :n, :n].mean(axis=0)
    cycles = {}
    for k, o in enumerate(OBJECTIVES):
        tri = tot = 0
        margins = []
        for x in range(ref.shape[1]):
            M = ref[k, x] - 0.5
            M = 0.5 * (M - M.T)                    # the block is skew by definition
            for i, j, m in itertools.combinations(range(n), 3):
                tot += 1
                margins += [abs(M[i, j]), abs(M[j, m]), abs(M[m, i])]
                a, b, c = M[i, j] > 0, M[j, m] > 0, M[m, i] > 0
                if (a and b and c) or (not a and not b and not c):
                    tri += 1
        rate = tri / max(1, tot)
        cycles[o] = {
            "triples": tot, "predicted_cycles": tri, "predicted_cycle_rate": rate,
            "chance_rate": 0.25,
            "mean_abs_margin_on_counted_edges": float(np.mean(margins)),
            "reading": ("at or near the 0.25 chance rate this measures INDIFFERENCE "
                        "on a same-model pool, not cyclic preference"
                        if abs(rate - 0.25) < 0.05 else
                        "materially away from the 0.25 chance rate"),
            "block": "comparator-vs-comparator reference tournament"}
    return conflict, cycles


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool-dir", type=Path, required=True)
    ap.add_argument("--probs-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--geometries", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--beta", type=float, default=0.25)
    ap.add_argument("--eta", type=float, default=1.0)
    ap.add_argument("--dual-tol", type=float, default=1e-6)
    ap.add_argument("--workers", type=int, default=64)
    args = ap.parse_args()

    rows = read_jsonl(args.pool_dir / "pool_responses.jsonl")
    load = lambda n: np.load(args.probs_dir / n)["p"].astype(np.float64)
    P = {"gpm": {"policy": load("probs_gpm_policy.npz"),
                 "ref": load("probs_gpm_ref.npz")},
         "bt": {"policy": load("probs_bt_policy.npz"),
                "ref": load("probs_bt_ref.npz")}}
    n_seeds = P["gpm"]["policy"].shape[0]

    report = {"geometries": {}, "n_ensemble_seeds": n_seeds,
              "cycle_note": ("predicted cycles are a property of the MODEL on "
                             "triples nobody compared; SafeRLHF's human annotation "
                             "graph contains zero triangles")}

    for n in args.geometries:
        key = f"{n}+{n}"
        print(f"=== geometry {key} ===", flush=True)
        blk = {"pool": pool_stats(rows, n),
               "oracle_gpm": oracle_stats(P["gpm"]["policy"], n),
               "oracle_bt": oracle_stats(P["bt"]["policy"], n)}
        g = P["gpm"]["policy"][:, :, :, :n, :n].mean(axis=0)
        b = P["bt"]["policy"][:, :, :, :n, :n].mean(axis=0)
        blk["gpm_bt_agreement"] = {
            "sign_agreement": float(np.mean(np.sign(g - 0.5) == np.sign(b - 0.5))),
            "pearson": pearson(g, b), "spearman": spearman(g, b),
            "mean_abs_difference": float(np.mean(np.abs(g - b)))}
        conflict, cycles = conflict_and_cycles(P["gpm"]["policy"],
                                               P["gpm"]["ref"], n)
        blk["objective_conflict_rate"] = conflict
        blk["predicted_cycles_gpm"] = cycles

        # ---- full-ensemble solve ----
        A_pol = P["gpm"]["policy"][:, :, :, :n, :n].mean(axis=0) - 0.5
        A_ref_raw = P["gpm"]["ref"][:, :, :, :n, :n].mean(axis=0) - 0.5
        A_ref = 0.5 * (A_ref_raw - np.swapaxes(A_ref_raw, -1, -2))
        idx = np.arange(n)
        A_ref[..., idx, idx] = 0.0
        tgt_full, diag = solve_target(A_pol, A_ref, args.beta, args.eta,
                                      args.dual_tol, args.workers)
        blk["solve"] = diag
        blk["target"] = {
            "rms": float(np.sqrt(np.mean(tgt_full ** 2))),
            "prompt_wise_variance_mean": float(np.mean(np.var(tgt_full, axis=-1))),
            "p10": float(np.percentile(tgt_full, 10)),
            "p90": float(np.percentile(tgt_full, 90))}
        blk["pair_count"] = {"policy_pairs_per_prompt": n * n,
                             "reference_pairs_per_prompt": n * n,
                             "total_pairs": 2 * n * n * A_pol.shape[1]}

        # ---- split-half over the ENSEMBLE ----
        halves = []
        for held in range(n_seeds):
            rest = [s for s in range(n_seeds) if s != held]
            for which, sel in (("single", [held]), ("rest", rest)):
                p = P["gpm"]["policy"][sel][:, :, :, :n, :n].mean(axis=0) - 0.5
                r = P["gpm"]["ref"][sel][:, :, :, :n, :n].mean(axis=0) - 0.5
                r = 0.5 * (r - np.swapaxes(r, -1, -2))
                r[..., idx, idx] = 0.0
                t, d = solve_target(p, r, args.beta, args.eta, args.dual_tol,
                                    args.workers)
                halves.append((held, which, t, d))
        pairs = []
        for held in range(n_seeds):
            a = next(t for h, w, t, _ in halves if h == held and w == "single")
            bb = next(t for h, w, t, _ in halves if h == held and w == "rest")
            pairs.append({
                "held_out_seed_index": held,
                "pearson": pearson(a, bb), "spearman": spearman(a, bb),
                "sign_agreement": float(np.mean(np.sign(a) == np.sign(bb))),
                "rms_ratio": float(np.sqrt(np.mean(a ** 2))
                                   / max(1e-12, np.sqrt(np.mean(bb ** 2))))})
        m = lambda k: float(np.mean([p[k] for p in pairs if p[k] is not None]))
        blk["target_split_half"] = {
            "per_pairing": pairs, "pearson_mean": m("pearson"),
            "spearman_mean": m("spearman"), "sign_agreement_mean": m("sign_agreement"),
            "all_halves_converged": all(d["dual_converged"] for _, _, _, d in halves),
            "max_projected_kkt": max(d["projected_kkt_residual"]
                                     for _, _, _, d in halves)}
        blk["total_solve_seconds"] = (diag["seconds"]
                                      + sum(d["seconds"] for _, _, _, d in halves))
        report["geometries"][key] = blk
        print(f"  conflict={conflict:.4f}  ent={blk['oracle_gpm']['predictive_entropy']:.4f}  "
              f"sat={blk['oracle_gpm']['saturation_below_0.02_or_above_0.98']:.4f}  "
              f"split-half r={blk['target_split_half']['pearson_mean']:.4f} "
              f"rho={blk['target_split_half']['spearman_mean']:.4f} "
              f"sign={blk['target_split_half']['sign_agreement_mean']:.4f}  "
              f"solve={blk['total_solve_seconds']:.1f}s", flush=True)

    # ---- selection: stability first, then compute ----
    def score(k):
        s = report["geometries"][k]["target_split_half"]
        return (s["spearman_mean"], s["sign_agreement_mean"])
    keys = list(report["geometries"])
    best = max(keys, key=score)
    other = [k for k in keys if k != best]
    margin = {k: {"spearman_delta": score(best)[0] - score(k)[0],
                  "sign_delta": score(best)[1] - score(k)[1],
                  "solve_seconds_ratio": (report["geometries"][best]["total_solve_seconds"]
                                          / report["geometries"][k]["total_solve_seconds"]),
                  "pair_count_ratio": (report["geometries"][best]["pair_count"]["total_pairs"]
                                       / report["geometries"][k]["pair_count"]["total_pairs"])}
              for k in other}
    report["selection"] = {
        "criterion": ("target split-half Spearman first, then sign agreement; ties "
                      "broken by compute. Validation prompts only; no downstream "
                      "policy performance consulted."),
        "selected": best, "margins_over_alternatives": margin}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "pool_geometry.json").write_text(json.dumps(report, indent=2))
    print(f"\nselected geometry: {best}")
    print(json.dumps(margin, indent=2))


if __name__ == "__main__":
    main()
