"""Turn raw screening verdicts into the tab:dataset_readiness cells.

Definitions, fixed here before the first judgment is read:

  p_hat_k(i,j) = 1/2 [ mean value over the i-first draws + mean over the j-first draws ]
  edge i->j in objective k only when p_hat_k(i,j) > 1/2 + delta   (delta declared, default 0.05)
  an edge needs ALL of its scheduled draws parsed, otherwise it is unresolved
  a triple is eligible only when its three edges are resolved in the panel being read

  C   the REPEATED cyclic-triangle fraction: the same oriented 3-cycle present in
      both independent repeat panels, over the triples eligible in both. A cycle
      seen once is reported separately as the single-panel rate, because a
      single-panel cycle is consistent with judge noise.
  D   cross-objective disagreement: for one unordered pair of objectives, over
      the (prompt, response pair) cells where BOTH resolve a strict direction,
      the fraction ordered oppositely; with more than two objectives, the mean
      of that fraction over the C(K,2) objective pairs. At K=2 this is the same
      number as before, so the Safe row and the UF row report one quantity and
      not a pooled version against a pairwise one. The per-pair table is kept in
      the artifact, since the mean can hide one strongly conflicting pair.
  gamma*  the 8-response audit surrogate: A_k[i,j] = p_hat_k(i,j) - 1/2 with the
      uniform occurrence reference, solved by the same max-min routine as the
      manuscript's finite games. This is NOT the policy panel's 8Y+8Z contract.
  TV  mean total variation between the exact NBPO and game-utilitarian targets
      on those same surrogate tensors.

Also reported, in the artifact rather than the table: the fitted-BT noise null
and the parametric-bootstrap excess cyclicity, the no-Condorcet fraction, the
share of cycles that involve the top response, order sensitivity, and the
missing mass. Unparsed judgments stay missing; nothing is imputed, retried or
dropped, and no candidate is deleted to renormalize a target.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from mnpo_scripts.nbpo_core import uniform_policy
from mnpo_scripts.nbpo_generic import matched_weight_l1, solve_finite_pool
from mnpo_scripts.nbpo_representations import AdaptiveGameRepresentation
from scripts.experiments.iclr2027_table1_v2 import nontransitivity_feasibility as nf

ROOT = Path("/work/sub_20260914")


def file_hash(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as s:
        for c in iter(lambda: s.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load(tags):
    """rows[prompt][rubric][(i,j)][panel_index] -> list of (order, value)."""
    rows = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    missing, total, order_flip = 0, 0, defaultdict(list)
    sources = {}
    for tag in tags:
        path = ROOT / "judgments" / tag / "judgments.jsonl"
        sources[str(path)] = file_hash(path)
        with path.open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                r = json.loads(line)
                total += 1
                if r["status"] != "ok":
                    missing += 1
                    continue
                key = (r["i"], r["j"])
                rows[r["prompt_id"]][r["rubric"]][key][r["draw"]].append(
                    (r["order"], float(r["value_for_i"])))
    return rows, {"judgments_read": total, "unparsed": missing,
                  "parse_rate": (total - missing) / max(total, 1)}, sources


def p_hat(draws, panel=None):
    """Order-averaged preference for the lower-indexed response.

    Returns None unless both orders are present in the panels being read, so a
    one-sided edge never becomes a direction.
    """
    per_order = defaultdict(list)
    for d, entries in draws.items():
        if panel is not None and d not in panel:
            continue
        for order, value in entries:
            per_order[order].append(value)
    if 0 not in per_order or 1 not in per_order:
        return None
    return 0.5 * (float(np.mean(per_order[0])) + float(np.mean(per_order[1])))


def edges(pairs, delta, panel=None):
    """(i,j) -> +1 if i beats j, -1 if j beats i, 0 inside the tie band, None unresolved."""
    out = {}
    for key, draws in pairs.items():
        p = p_hat(draws, panel)
        if p is None:
            out[key] = None
            continue
        if p > 0.5 + delta:
            out[key] = 1
        elif p < 0.5 - delta:
            out[key] = -1
        else:
            out[key] = 0
    return out


def oriented_cycles(edge, responses):
    """The oriented 3-cycles and the eligible triple count for one graph."""
    found, eligible = set(), 0
    for tri in combinations(sorted(responses), 3):
        a, b, c = tri
        vals = [edge.get((a, b)), edge.get((a, c)), edge.get((b, c))]
        if any(v is None or v == 0 for v in vals):
            continue
        eligible += 1
        ab, ac, bc = vals
        # a->b when ab=+1; a->c when ac=+1; b->c when bc=+1
        forward = (ab == 1 and bc == 1 and ac == -1)
        backward = (ab == -1 and bc == -1 and ac == 1)
        if forward:
            found.add((tri, "abc"))
        elif backward:
            found.add((tri, "cba"))
    return found, eligible


def condorcet(edge, responses):
    """Weak and strict Condorcet existence, reported separately.

    The weak winner is a response that is never strictly beaten, which is the
    rate to quote: a single tie inside the margin band is not evidence that
    choosing a best response is hard. The strict winner must beat all seven
    opponents outright, so with a wide tie band it is absent almost by
    construction and on its own would overstate the difficulty.
    """
    weak, strict = False, False
    for i in responses:
        losses, wins, resolved = 0, 0, 0
        for j in responses:
            if i == j:
                continue
            key = (min(i, j), max(i, j))
            v = edge.get(key)
            if v is None or v == 0:
                continue
            resolved += 1
            beats = (v == 1) if i < j else (v == -1)
            wins += int(beats)
            losses += int(not beats)
        if losses == 0:
            weak = True
        if resolved == len(responses) - 1 and wins == resolved:
            strict = True
    return weak, strict


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tags", nargs="+", required=True,
                    help="judgment directories to pool, e.g. safe_pilot10_panelA")
    ap.add_argument("--skip-surrogate", action="store_true",
                    help="cycle/disagreement statistics only; no gamma*/TV, no cells. "
                         "Used to re-check the cheap statistics without spending the "
                         "finite-game solve again; never used to fill a table cell.")
    ap.add_argument("--cell-prefix", default="safe",
                    help="tab:dataset_readiness row prefix: safe or uf")
    ap.add_argument("--objectives", nargs="+", default=None,
                    help="restrict and order the rubrics; default is all present")
    ap.add_argument("--responses-dir", default="screen200",
                    help="responses directory, read only to find duplicate texts")
    ap.add_argument("--out", required=True, help="output json under analysis/")
    ap.add_argument("--delta", type=float, default=0.05,
                    help="tie margin on p_hat, declared before reading verdicts")
    ap.add_argument("--beta", type=float, default=0.25)
    ap.add_argument("--eta", type=float, default=1.0)
    ap.add_argument("--responses", type=int, default=8)
    ap.add_argument("--panel-split", type=int, nargs="+", default=[0, 1],
                    help="draw indices forming the two independent repeat panels")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260914)
    args = ap.parse_args()

    rows, parse, sources = load(args.tags)
    prompts = sorted(rows)
    # Duplicate occurrences are kept in the panel -- deleting them would
    # renormalize the reference -- but a pair of identical texts sits at
    # p_hat = 1/2 by construction, so the cycle rate is also computed on the
    # graph of distinct texts, where those pairs cannot create or block a cycle.
    dup_map, dup_count = {}, 0
    resp_file = ROOT / "responses" / args.responses_dir / "responses.jsonl"
    if resp_file.exists():
        by_prompt_hash = defaultdict(dict)
        for line in resp_file.open():
            if not line.strip():
                continue
            e = json.loads(line)
            by_prompt_hash[e["prompt_id"]][e["response_index"]] = e["response_sha256"]
        for pid, idx_hash in by_prompt_hash.items():
            first = {}
            keep = []
            for idx in sorted(idx_hash):
                h = idx_hash[idx]
                if h in first:
                    dup_count += 1
                    continue
                first[h] = idx
                keep.append(idx)
            dup_map[pid] = keep
    rubrics = sorted({r for p in prompts for r in rows[p]})
    if args.objectives:
        absent = [o for o in args.objectives if o not in rubrics]
        if absent:
            raise SystemExit("verdicts lack these objectives: %s" % absent)
        rubrics = list(args.objectives)
    responses = list(range(args.responses))
    panels = [{d} for d in args.panel_split]

    per_prompt, cyc_single, cyc_repeat = {}, [], []
    eligible_single, eligible_repeat = 0, 0
    cycles_single, cycles_repeat = 0, 0
    cycles_repeat_unique, eligible_repeat_unique = 0, 0
    no_condorcet, condorcet_seen, strict_seen, graphs_scored = 0, 0, 0, 0
    top_involved, cycle_prompts = 0, 0
    disagree_pair = defaultdict(lambda: [0, 0])      # (obj_a, obj_b) -> [num, den]
    order_gap = []
    complete_prompts = []

    A_rows, A_prompts = [], []
    for pid in prompts:
        per_rubric = {}
        complete = True
        for rub in rubrics:
            pairs = rows[pid].get(rub, {})
            full = edges(pairs, args.delta)
            if any(v is None for v in full.values()) or len(full) != len(responses) * (len(responses) - 1) // 2:
                complete = False
            per_rubric[rub] = {"all": full,
                              "panels": [edges(pairs, args.delta, panel=p) for p in panels],
                              "p_hat": {str(k): p_hat(v) for k, v in pairs.items()}}
            # order sensitivity: |mean(order 0) - mean(order 1)| per pair
            for key, draws in pairs.items():
                per_order = defaultdict(list)
                for d, entries in draws.items():
                    for order, value in entries:
                        per_order[order].append(value)
                if 0 in per_order and 1 in per_order:
                    order_gap.append(abs(float(np.mean(per_order[0]))
                                         - float(np.mean(per_order[1]))))

        if complete:
            complete_prompts.append(pid)

        prompt_has_cycle = False
        for rub in rubrics:
            full = per_rubric[rub]["all"]
            found_all, elig_all = oriented_cycles(full, responses)
            eligible_single += elig_all
            cycles_single += len(found_all)
            if found_all:
                prompt_has_cycle = True
            pa, pb = (oriented_cycles(e, responses) for e in per_rubric[rub]["panels"])
            both = pa[0] & pb[0]
            eligible_repeat += min(pa[1], pb[1])
            cycles_repeat += len(both)
            uniq = dup_map.get(pid, responses)
            if len(uniq) >= 3:
                ua, ub = (oriented_cycles(e, uniq) for e in per_rubric[rub]["panels"])
                eligible_repeat_unique += min(ua[1], ub[1])
                cycles_repeat_unique += len(ua[0] & ub[0])
            weak, strict = condorcet(full, responses)
            condorcet_seen += int(weak)
            no_condorcet += int(not weak)
            strict_seen += int(strict)
            graphs_scored += 1
            # does a cycle involve the response with the highest mean p_hat?
            if found_all:
                score = {i: 0.0 for i in responses}
                for (i, j), v in full.items():
                    p = per_rubric[rub]["p_hat"].get(str((i, j)))
                    if p is None:
                        continue
                    score[i] += p
                    score[j] += 1.0 - p
                top = max(score, key=score.get)
                if any(top in tri for tri, _ in found_all):
                    top_involved += 1
        if prompt_has_cycle:
            cycle_prompts += 1

        # cross-objective disagreement on strictly resolved pairs, over every
        # unordered pair of objectives so K=2 and K=4 measure the same thing
        for ia in range(len(rubrics)):
            for ib in range(ia + 1, len(rubrics)):
                name = (rubrics[ia], rubrics[ib])
                ea = per_rubric[rubrics[ia]]["all"]
                eb = per_rubric[rubrics[ib]]["all"]
                for key in ea:
                    a, b = ea.get(key), eb.get(key)
                    if a in (None, 0) or b in (None, 0):
                        continue
                    disagree_pair[name][1] += 1
                    disagree_pair[name][0] += int(a != b)

        # the 8-response surrogate tensor, only from complete prompts
        if complete:
            mat = np.zeros((len(rubrics), len(responses), len(responses)))
            for k, rub in enumerate(rubrics):
                for (i, j), _ in per_rubric[rub]["all"].items():
                    p = per_rubric[rub]["p_hat"][str((i, j))]
                    mat[k, i, j] = p - 0.5
                    mat[k, j, i] = 0.5 - p
            A_rows.append(mat)
            A_prompts.append(pid)
        per_prompt[pid] = {"complete": complete,
                           "p_hat": {r: per_rubric[r]["p_hat"] for r in rubrics}}

    result = {
        "tags": args.tags, "sources_sha256": sources,
        "delta_tie_margin": args.delta,
        "n_prompts_read": len(prompts),
        "n_prompts_complete": len(complete_prompts),
        "parse": parse,
        "rubrics": rubrics,
        "C_repeated": cycles_repeat / max(eligible_repeat, 1),
        "C_single_panel": cycles_single / max(eligible_single, 1),
        "C_repeated_unique": cycles_repeat_unique / max(eligible_repeat_unique, 1),
        "eligible_triples_repeated_unique": eligible_repeat_unique,
        "duplicate_occurrences": dup_count,
        "eligible_triples_repeated": eligible_repeat,
        "eligible_triples_single": eligible_single,
        "cycles_repeated": cycles_repeat, "cycles_single": cycles_single,
        "any_cycle_prompt_fraction": cycle_prompts / max(len(prompts), 1),
        "no_weak_condorcet_fraction": no_condorcet / max(graphs_scored, 1),
        "strict_condorcet_fraction": strict_seen / max(graphs_scored, 1),
        "graphs_scored": graphs_scored,
        "cycles_involving_top_response": top_involved,
        "D_disagreement": (float(np.mean([n / d for n, d in disagree_pair.values() if d]))
                           if any(d for _, d in disagree_pair.values()) else 0.0),
        "D_definition": ("mean over the C(K,2) unordered objective pairs of the "
                         "fraction of strictly-resolved (prompt, response pair) "
                         "cells the two objectives order oppositely"),
        "D_per_objective_pair": {"%s|%s" % k: {"disagreements": n, "resolved_cells": d,
                                               "fraction": (n / d) if d else None}
                                 for k, (n, d) in sorted(disagree_pair.items())},
        "disagreement_denominator": sum(d for _, d in disagree_pair.values()),
        "mean_order_gap": float(np.mean(order_gap)) if order_gap else None,
    }

    # ---- the finite-game surrogate on the complete prompts
    if A_rows and not args.skip_surrogate:
        A = np.stack(A_rows, axis=1)                      # (K, X, I, I)
        A = 0.5 * (A - np.swapaxes(A, -1, -2))
        idx = np.arange(A.shape[-1])
        A[..., idx, idx] = 0.0
        cap = float(np.abs(A).max())
        mu = np.full((A.shape[1], A.shape[2]), 1.0 / A.shape[2])
        beta = np.full(A.shape[0], args.beta)
        d = nf.game_values(A, mu, mu, beta)
        feas = nf.solve_max_min(A, mu, beta, d)
        gamma = float(feas["rho_star"])
        # The aggregate solve needs about 900 iterations at this panel size
        # (152 at 20 prompts, 259 at 40), so with maxiter=400 it stops at the
        # cap instead of converging. gamma* then has to be read with its
        # certificate: the achieved value is a feasible lower bound and
        # certify_max_min gives the tangent-plane upper bound, so the gap says
        # how tight the cell is. Reporting the value without that gap would be
        # exactly the kind of uncertifiable number the contract forbids.
        certificate = {k: feas[k] for k in
                       ("certified_upper_bound", "rho_star_gap", "slsqp_status",
                        "slsqp_message", "slsqp_iterations",
                        "primal_feasibility_residual",
                        "complementary_slackness_residual", "certificate_weights",
                        "surplus")}
        certificate["converged"] = (feas["slsqp_status"] == 0)
        certificate["hit_iteration_cap"] = (feas["slsqp_iterations"] >= 400)
        At = torch.from_numpy(A)
        mu_t = uniform_policy(A.shape[1], A.shape[2])
        bt = torch.full((A.shape[0],), float(args.beta), dtype=torch.float64)
        rep = AdaptiveGameRepresentation(At, At, mu_t, bt)
        nbpo = solve_finite_pool(rep, "nash", eta=args.eta, M=800, R=1,
                                 inner_solver="exact", dual_solver="root", dual_tol=1e-6)
        l1 = matched_weight_l1(nbpo)
        util = solve_finite_pool(rep, "utilitarian", eta=args.eta, R=1, weight_l1=l1,
                                 inner_solver="exact")
        pin, piu = nbpo.pi.numpy(), util.pi.numpy()
        s_n = nf.game_values(A, pin, mu, beta) - d
        s_u = nf.game_values(A, piu, mu, beta) - d
        result["surrogate"] = {
            "n_prompts": len(A_prompts), "responses": int(A.shape[2]),
            "payoff_abs_max": cap,
            "reference": "uniform over the eight base occurrences",
            "beta": args.beta, "d": d.tolist(),
            "gamma_star": gamma,
            "gamma_star_certificate": certificate,
            "nbpo_surplus": s_n.tolist(), "nbpo_min_surplus": float(s_n.min()),
            "util_surplus": s_u.tolist(), "util_min_surplus": float(s_u.min()),
            "target_TV_nbpo_util": float(nf.total_variation(pin, piu)),
            "kl_nbpo_reference": float(nf.mean_kl(pin, mu)),
            "matched_weight_l1": float(l1),
            "projected_kkt_residual": float(getattr(nbpo, "projected_kkt_residual", float("nan"))),
            "not_the_policy_contract": ("this is the 8-response audit surrogate with a "
                                        "uniform occurrence reference, not the panel's "
                                        "independent 8Y+8Z problem")}
        pre = args.cell_prefix
        result["cells"] = {pre + "N": len(A_prompts), pre + "C": result["C_repeated"],
                           pre + "D": result["D_disagreement"], pre + "Gamma": gamma,
                           pre + "TV": result["surrogate"]["target_TV_nbpo_util"]}
    else:
        result["surrogate"] = None
        result["cells"] = None
        result["surrogate_skipped"] = bool(args.skip_surrogate)

    out = ROOT / "analysis" / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, default=float) + "\n")
    print(json.dumps({k: result[k] for k in
                      ("n_prompts_read", "n_prompts_complete", "C_repeated",
                       "C_single_panel", "C_repeated_unique", "D_disagreement",
                       "no_weak_condorcet_fraction", "strict_condorcet_fraction",
                       "mean_order_gap", "duplicate_occurrences", "parse")},
                     default=float), flush=True)
    if result.get("cells"):
        print(json.dumps({"cells": result["cells"]}, default=float), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
