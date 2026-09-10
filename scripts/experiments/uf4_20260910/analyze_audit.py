"""Appendix C analysis: cycle rates, the fitted BT noise null, and bootstrap intervals.

Implements the manuscript's definitions and nothing else:

  p_hat^(b)_ij = 1/2 ( mean X over i-first draws + mean X over j-first draws )   Eq. (audit-preference)
  edge i->j only when p_hat > 1/2; an exact tie creates no edge
  an edge needs ALL ten scheduled judgments, otherwise it is unresolved
  a complete tournament needs all six pairs resolved

  C_A  prompts with any directed cycle in panel A, length three or four
  R    prompts carrying the SAME oriented triangle in both panels (repeatability, not confirmation)
  W_A  prompts where every response has an observed loss in panel A

The BT null is Pr(i, tie, j | o) proportional to (e^{z/2}, 2 nu, e^{-z/2}) with
z = r_i - r_j + b o, scores summing to zero within a prompt, a rubric-level b and
nu, fitted on panel A by summed NLL with a fixed 1e-3 squared penalty on the
scores, b and log nu. Synthetic panels preserve the observed presentation orders,
repeat counts and missingness.

Missing comparisons never establish an event: unresolved counts and the
worst-case upper rate are reported beside every rate.
"""
from __future__ import annotations

import argparse, hashlib, itertools, json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path("/work/uf4_20260910")
RUBRICS = ("instruction_following", "truthfulness", "honesty", "helpfulness", "joint_control")
INDIVIDUAL = RUBRICS[:4]
N_RESP = 4
PAIRS = list(itertools.combinations(range(N_RESP), 2))
DRAWS_PER_ORDER = 5


def file_hash(p):
    h = hashlib.sha256()
    with open(p, "rb") as s:
        for c in iter(lambda: s.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load(path):
    """(rubric, prompt, panel, (i,j), order) -> list of X values for i."""
    data = defaultdict(list)
    with open(path) as stream:
        for line in stream:
            r = json.loads(line)
            if r["status"] != "ok":
                data[(r["rubric"], r["prompt_id"], r["panel"], (r["i"], r["j"]), r["order"])]
                continue
            data[(r["rubric"], r["prompt_id"], r["panel"], (r["i"], r["j"]), r["order"])].append(
                float(r["value_for_i"]))
    return data


def edge_estimates(data, rubric, prompt, panel):
    """p_hat for each unordered pair, or None when any scheduled judgment is missing."""
    out = {}
    for pair in PAIRS:
        means = []
        for order in (0, 1):
            vals = data.get((rubric, prompt, panel, pair, order), [])
            if len(vals) != DRAWS_PER_ORDER:
                means = None
                break
            means.append(float(np.mean(vals)))
        out[pair] = None if means is None else 0.5 * (means[0] + means[1])
    return out


def graph_from(p_hat):
    """Directed edges; an exact tie at 1/2 creates none. None if any pair unresolved."""
    if any(v is None for v in p_hat.values()):
        return None
    edges = set()
    for (i, j), v in p_hat.items():
        if v > 0.5:
            edges.add((i, j))
        elif v < 0.5:
            edges.add((j, i))
    return edges


def directed_triangles(edges):
    out = set()
    for a, b, c in itertools.combinations(range(N_RESP), 3):
        for tri in ((a, b, c), (a, c, b)):
            x, y, z = tri
            if (x, y) in edges and (y, z) in edges and (z, x) in edges:
                out.add(tuple(sorted([(x, y), (y, z), (z, x)])))
    return out


def has_cycle(edges):
    """Any directed cycle over four nodes, length three or four."""
    if directed_triangles(edges):
        return True
    for perm in itertools.permutations(range(N_RESP)):
        if perm[0] != min(perm):
            continue
        a, b, c, d = perm
        if (a, b) in edges and (b, c) in edges and (c, d) in edges and (d, a) in edges:
            return True
    return False


def every_response_loses(edges):
    return all(any((j, i) in edges for j in range(N_RESP) if j != i) for i in range(N_RESP))


def fit_bt_null(data, rubric, prompts, penalty=1e-3):
    """One (b, log nu) per rubric with per-prompt sum-zero scores, fitted on panel A."""
    obs = []                                  # (prompt_index, i, j, order_sign, counts)
    for pi, prompt in enumerate(prompts):
        for pair in PAIRS:
            for order in (0, 1):
                vals = data.get((rubric, prompt, "A", pair, order), [])
                if not vals:
                    continue
                w = sum(1 for v in vals if v == 1.0)
                t = sum(1 for v in vals if v == 0.5)
                l = sum(1 for v in vals if v == 0.0)
                obs.append((pi, pair[0], pair[1], 1.0 if order == 0 else -1.0, w, t, l))
    if not obs:
        return None
    P = len(prompts)
    obs = np.array(obs, dtype=np.float64)

    def unpack(theta):
        free = theta[:P * (N_RESP - 1)].reshape(P, N_RESP - 1)
        scores = np.concatenate([free, -free.sum(1, keepdims=True)], axis=1)
        return scores, theta[-2], theta[-1]

    def nll(theta):
        scores, b, log_nu = unpack(theta)
        nu = np.exp(log_nu)
        pi_idx = obs[:, 0].astype(int)
        z = scores[pi_idx, obs[:, 1].astype(int)] - scores[pi_idx, obs[:, 2].astype(int)] + b * obs[:, 3]
        ew, el = np.exp(z / 2), np.exp(-z / 2)
        denom = ew + 2 * nu + el
        total = -(obs[:, 4] * np.log(ew / denom) + obs[:, 5] * np.log(2 * nu / denom)
                  + obs[:, 6] * np.log(el / denom)).sum()
        return total + penalty * (float((theta[:-2] ** 2).sum()) + b ** 2 + log_nu ** 2)

    theta0 = np.zeros(P * (N_RESP - 1) + 2)
    res = minimize(nll, theta0, method="L-BFGS-B", options={"maxiter": 3000})
    scores, b, log_nu = unpack(res.x)
    return {"scores": scores, "b": float(b), "nu": float(np.exp(log_nu)),
            "nll": float(res.fun), "converged": bool(res.success), "n_obs": len(obs)}


def simulate(fit, data, rubric, prompts, rng):
    """One synthetic A/B pair preserving orders, repeat counts and missingness."""
    scores, b, nu = fit["scores"], fit["b"], fit["nu"]
    sim = defaultdict(list)
    for pi, prompt in enumerate(prompts):
        for panel in ("A", "B"):
            for pair in PAIRS:
                for order in (0, 1):
                    n = len(data.get((rubric, prompt, panel, pair, order), []))
                    if n == 0:
                        continue
                    o = 1.0 if order == 0 else -1.0
                    z = scores[pi, pair[0]] - scores[pi, pair[1]] + b * o
                    w = np.array([np.exp(z / 2), 2 * nu, np.exp(-z / 2)])
                    w = w / w.sum()
                    draws = rng.choice([1.0, 0.5, 0.0], size=n, p=w)
                    sim[(rubric, prompt, panel, pair, order)] = list(draws)
    return sim


def panel_metrics(data, rubric, prompts):
    per_prompt = {}
    for prompt in prompts:
        gA = graph_from(edge_estimates(data, rubric, prompt, "A"))
        gB = graph_from(edge_estimates(data, rubric, prompt, "B"))
        per_prompt[prompt] = (gA, gB)
    complete_both = [p for p, (a, b) in per_prompt.items() if a is not None and b is not None]
    cA = sum(1 for p in prompts if per_prompt[p][0] is not None and has_cycle(per_prompt[p][0]))
    unresolved_A = sum(1 for p in prompts if per_prompt[p][0] is None)
    wA = sum(1 for p in prompts if per_prompt[p][0] is not None
             and every_response_loses(per_prompt[p][0]))
    repeated = 0
    for p in complete_both:
        a, b = per_prompt[p]
        if directed_triangles(a) & directed_triangles(b):
            repeated += 1
    tri = sum(len(directed_triangles(per_prompt[p][0])) for p in prompts
              if per_prompt[p][0] is not None)
    return {"per_prompt": per_prompt, "n_complete_both": len(complete_both),
            "C_A": cA, "R": repeated, "W_A": wA,
            "unresolved_A": unresolved_A,
            "cyclic_triangles_A": tri}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audit-name", default="v1")
    ap.add_argument("--split-file", default="audit_100")
    ap.add_argument("--simulations", type=int, default=2000)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260911)
    args = ap.parse_args()

    base = ROOT / "audit" / args.audit_name
    jpath = base / "judgments" / args.split_file / "judgments.jsonl"
    data = load(jpath)
    prompts = sorted({k[1] for k in data})
    rng = np.random.default_rng(args.seed)

    report = {"protocol": "app:within-rubric-audit", "split_file": args.split_file,
              "n_planned_prompts": len(prompts), "planned_triangles": 4 * len(prompts),
              "judgments_sha256": file_hash(jpath), "simulations": args.simulations,
              "bootstrap_replicates": args.bootstrap, "seed": args.seed,
              "source_sha256": file_hash(__file__), "rubrics": {}}

    raw_p = {}
    for rubric in RUBRICS:
        m = panel_metrics(data, rubric, prompts)
        fit = fit_bt_null(data, rubric, prompts)
        sims = []
        for _ in range(args.simulations):
            sim = simulate(fit, data, rubric, prompts, rng)
            sims.append(panel_metrics(sim, rubric, prompts)["R"])
        sims = np.array(sims, dtype=float)
        R = m["R"]
        p_upper = (1 + int((sims >= R).sum())) / (args.simulations + 1)
        boot = []
        for _ in range(args.bootstrap):
            pick = rng.choice(prompts, size=len(prompts), replace=True)
            mb = panel_metrics(data, rubric, list(pick))
            fb = fit_bt_null(data, rubric, list(pick))
            sb = panel_metrics(simulate(fb, data, rubric, list(pick), rng), rubric, list(pick))["R"]
            boot.append((mb["R"] - sb) / max(len(prompts), 1) * 100.0)
        boot = np.array(boot, dtype=float)
        n = len(prompts)
        report["rubrics"][rubric] = {
            "n_complete_both": m["n_complete_both"],
            "C_A": m["C_A"], "C_A_rate": m["C_A"] / n,
            "C_A_upper_rate_if_all_unresolved_cycled": (m["C_A"] + m["unresolved_A"]) / n,
            "unresolved_A": m["unresolved_A"],
            "R": R, "R_rate": R / n,
            "R_BT_mean": float(sims.mean()), "R_BT_rate": float(sims.mean()) / n,
            "W_A": m["W_A"], "W_A_rate": m["W_A"] / n,
            "cyclic_triangles_A": m["cyclic_triangles_A"],
            "cyclic_triangle_rate_over_planned": m["cyclic_triangles_A"] / (4 * n),
            "delta_R_points": (R - float(sims.mean())) / n * 100.0,
            "delta_R_ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
            "p_upper_tail": p_upper,
            "bt_fit": {"b": fit["b"], "nu": fit["nu"], "converged": fit["converged"],
                       "n_observations": fit["n_obs"]}}
        raw_p[rubric] = p_upper
        print(json.dumps({"rubric": rubric, "n": m["n_complete_both"], "C_A": m["C_A"],
                          "R": R, "R_BT": round(float(sims.mean()), 2), "W_A": m["W_A"],
                          "dR_pts": round((R - float(sims.mean())) / n * 100, 2),
                          "p": round(p_upper, 4)}), flush=True)

    order = sorted(INDIVIDUAL, key=lambda r: raw_p[r])
    holm, running = {}, 0.0
    for rank, rubric in enumerate(order):
        adj = min(1.0, max(running, (len(INDIVIDUAL) - rank) * raw_p[rubric]))
        running = adj
        holm[rubric] = adj
    for rubric in INDIVIDUAL:
        report["rubrics"][rubric]["p_holm"] = holm[rubric]
    report["rubrics"]["joint_control"]["p_holm"] = None
    report["joint_control_note"] = ("unadjusted; reported separately and never pooled with the "
                                    "four individual rubrics")
    report["reading"] = ("R measures repeatability across two panels of the same judge, not "
                         "statistical confirmation. The Hoeffding witness test needs the "
                         "200-draw confirmation stage and is not in this file.")
    out = ROOT / "analysis/audit"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"cycles_{args.split_file}.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"written": str(path),
                      "p_holm": {r: round(holm[r], 4) for r in INDIVIDUAL}}, indent=1), flush=True)


if __name__ == "__main__":
    main()
