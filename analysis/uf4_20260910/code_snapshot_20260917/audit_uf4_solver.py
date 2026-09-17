"""Independent audit of the UF-4 finite-pool solutions. NumPy only, no solver code.

Everything here is recomputed from the stored arrays under the manuscript's own
definitions, so it can disagree with the solver instead of restating it:

  r_k(x,j)  = sum_i pi(x,i) A_k(x,i,j)                              Eq. (6)
  V_k(pi)   = mean_x [ -beta_k logsumexp_j( log mu(x,j) - r/beta_k ) ]   Eq. (8)
  d_k       = V_k(mu) from the reference-as-learner tensor            Eq. (10)
  s_k(pi)   = V_k(pi) - d_k
  F(pi)     = sum_k log s_k(pi)
  D(pi||pi_t) = mean_x KL(pi_x || pi_t,x)
  J(pi)     = F(pi) - D(pi||pi_t)/eta                                Eq. (14)

Eq. (14) is what the Nash stage maximizes, so the decisive check is not whether
lambda_k * s_k = 1 -- that identity is how the solver defines lambda and is
vacuous as a certificate -- but whether any other feasible stored policy attains
a larger J under identical payoffs. The dual certificate is instead recomputed
from lambda alone: pi(lambda) is rebuilt here by the exponential map and its
surplus is compared with 1/lambda.

Antisymmetry and the zero diagonal are construction invariants and are reported
as such, never as evidence that a preference or an optimizer is right.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path("/work/uf4_20260910")
OBJECTIVES = ("instruction_following", "truthfulness", "honesty", "helpfulness")


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def logsumexp(a, axis):
    m = np.max(a, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))).squeeze(axis)


def margins(A, pi):
    """r[k,x,j] = sum_i pi[x,i] A[k,x,i,j]"""
    return np.einsum("xi,kxij->kxj", pi, A)


def game_values(A, pi, mu, beta, per_prompt=False):
    r = margins(A, pi)
    log_w = np.log(mu)[None, :, :] - r / beta[:, None, None]
    v_kx = -beta[:, None] * logsumexp(log_w, axis=-1)
    return v_kx if per_prompt else v_kx.mean(axis=1)


def regularized_opponent(A, pi, mu, beta):
    """nu*_k(x,j) proportional to mu(x,j) exp(-r_k(x,j)/beta_k), Eq. (7)."""
    r = margins(A, pi)
    log_nu = np.log(mu)[None, :, :] - r / beta[:, None, None]
    log_nu = log_nu - logsumexp(log_nu, axis=-1)[..., None]
    return np.exp(log_nu)


def objective_gradient(A, nu):
    """q_k(x,i) = sum_j nu_k(x,j) A_k(x,i,j), Eq. (9)."""
    return np.einsum("kxj,kxij->kxi", nu, A)


def disagreement(A_ref, mu, beta):
    """d_k = V_k(mu) with mu on BOTH sides of the reference game."""
    pool = A_ref.shape[2]
    pi_ref = np.full((A_ref.shape[1], pool), 1.0 / pool)
    return game_values(A_ref, pi_ref, mu, beta)


def kl_to_center(pi, pi_t):
    return float(np.mean(np.sum(pi * (np.log(pi) - np.log(pi_t)), axis=-1)))


def evaluate(A, A_ref, pi, pi_t, mu, beta, eta):
    d = disagreement(A_ref, mu, beta)
    V = game_values(A, pi, mu, beta)
    s = V - d
    D = kl_to_center(pi, pi_t)
    F = float(np.sum(np.log(s))) if bool((s > 0).all()) else float("nan")
    return {"V": V.tolist(), "d": d.tolist(), "surplus": s.tolist(),
            "min_surplus": float(s.min()), "all_surplus_positive": bool((s > 0).all()),
            "F_sum_log_surplus": F, "D_mean_prompt_kl": D,
            "J_objective": F - D / eta}


def exponential_map(pi_t, q, weights, eta):
    """pi(x,i) proportional to pi_t(x,i) exp(eta sum_k w_k q_k(x,i)).

    This is the inner problem's stationarity condition. Applied at q evaluated
    from a candidate pi it is a fixed-point step; its residual measures whether
    the stored policy actually satisfies the condition it is supposed to.
    """
    score = eta * np.einsum("k,kxi->xi", weights, q)
    log_pi = np.log(pi_t) + score
    log_pi = log_pi - logsumexp(log_pi, axis=-1)[:, None]
    return np.exp(log_pi)


def stationarity(A, pi, pi_t, mu, beta, weights, eta):
    """Max over prompts of the spread of log(pi/pi_t) - eta*sum_k w_k q_k."""
    nu = regularized_opponent(A, pi, mu, beta)
    q = objective_gradient(A, nu)
    lhs = np.log(pi) - np.log(pi_t)
    rhs = eta * np.einsum("k,kxi->xi", weights, q)
    gap = lhs - rhs
    spread = gap.max(axis=-1) - gap.min(axis=-1)
    remapped = exponential_map(pi_t, q, weights, eta)
    return {"stationarity_spread_max": float(spread.max()),
            "stationarity_spread_mean": float(spread.mean()),
            "fixed_point_step_linf": float(np.abs(remapped - pi).max()),
            "fixed_point_step_tv_mean": float(0.5 * np.abs(remapped - pi).sum(-1).mean())}


def dual_certificate(A, A_ref, pi_t, mu, beta, weights, eta, iters=400, tol=1e-14):
    """Rebuild pi from lambda alone, then test the dual condition s_k = 1/lambda_k.

    The solver's stored pi is not used. The map is iterated to its own fixed
    point because q depends on pi through the adaptive opponent.
    """
    pi = pi_t.copy()
    for step in range(iters):
        nu = regularized_opponent(A, pi, mu, beta)
        q = objective_gradient(A, nu)
        new = exponential_map(pi_t, q, weights, eta)
        delta = float(np.abs(new - pi).max())
        pi = new
        if delta < tol:
            break
    d = disagreement(A_ref, mu, beta)
    s = game_values(A, pi, mu, beta) - d
    reciprocal = 1.0 / weights
    return {"iterations": step + 1, "final_map_delta": delta,
            "surplus_at_pi_of_lambda": s.tolist(),
            "reciprocal_lambda": reciprocal.tolist(),
            "dual_residual_inf": float(np.abs(s - reciprocal).max()),
            "dual_residual_relative": float(np.abs(s - reciprocal).max() / np.abs(s).min()),
            "pi_of_lambda_vs_stored_note": "compared separately",
            "pi": pi}


def target_distances(pi_a, pi_b, pi_t):
    tv = 0.5 * np.abs(pi_a - pi_b).sum(-1)
    sym_kl = (np.sum(pi_a * (np.log(pi_a) - np.log(pi_b)), -1)
              + np.sum(pi_b * (np.log(pi_b) - np.log(pi_a)), -1))
    g_a = np.log(pi_a) - np.log(pi_t)
    g_b = np.log(pi_b) - np.log(pi_t)
    pool = pi_a.shape[1]
    idx = [(i, j) for i in range(pool) for j in range(i + 1, pool)]
    pair_a = np.stack([g_a[:, i] - g_a[:, j] for i, j in idx], axis=1)
    pair_b = np.stack([g_b[:, i] - g_b[:, j] for i, j in idx], axis=1)
    return {"target_tv_mean": float(tv.mean()), "target_tv_median": float(np.median(tv)),
            "target_tv_p95": float(np.quantile(tv, 0.95)), "target_tv_max": float(tv.max()),
            "symmetric_kl_mean": float(sym_kl.mean()),
            "symmetric_kl_max": float(sym_kl.max()),
            "pair_target_diff_rms": float(np.sqrt(np.mean((pair_a - pair_b) ** 2))),
            "pair_target_rms_a": float(np.sqrt(np.mean(pair_a ** 2))),
            "pair_target_rms_b": float(np.sqrt(np.mean(pair_b ** 2))),
            "pair_target_diff_rms_over_rms_a":
                float(np.sqrt(np.mean((pair_a - pair_b) ** 2)) / np.sqrt(np.mean(pair_a ** 2))),
            "n_prompts": int(pi_a.shape[0])}


def load(target_name, split="train"):
    base = ROOT / "targets" / target_name
    complete = json.loads((base / "complete.json").read_text())
    tensor = base / split / "tensor"
    solver = base / split / "solver"
    solution = json.loads((solver / "solution.json").read_text())
    A = np.load(tensor / "tensor_policy.npz")["A"].astype(np.float64)
    A_ref = np.load(tensor / "tensor_ref.npz")["A"].astype(np.float64)
    pi = np.load(solver / "pi_star.npz")["pi"].astype(np.float64)
    pi_t = np.load(solver / "pi_t.npz")["pi"].astype(np.float64)
    g = np.load(solver / "target_log_ratio.npz")["h"].astype(np.float64)
    return {"name": target_name, "split": split, "A": A, "A_ref": A_ref, "pi": pi,
            "pi_t": pi_t, "g": g,
            "weights": np.asarray(solution["lambda_raw"], dtype=np.float64),
            "beta": float(complete["beta"]), "eta": float(complete["eta"]),
            "aggregation": solution["aggregation"],
            "solver_reported": {"V": solution["V"], "d": solution["d"],
                                "surplus": solution["surplus"],
                                "kkt_residual": solution.get("kkt_residual"),
                                "projected_kkt_residual": solution.get("projected_kkt_residual"),
                                "lambda_at_lower_bound": solution.get("lambda_at_lower_bound"),
                                "lambda_at_upper_bound": solution.get("lambda_at_upper_bound"),
                                "fixed_point_residual": solution.get("fixed_point_residual")},
            "solution_sha256": file_hash(solver / "solution.json"),
            "tensor_policy_sha256": file_hash(tensor / "tensor_policy.npz")}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nash", default="nash_v1")
    ap.add_argument("--utilitarian", default="util_l1matched_v1")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default=str(ROOT / "analysis/solver_audit"))
    args = ap.parse_args()

    nash = load(args.nash, args.split)
    util = load(args.utilitarian, args.split)
    if not np.array_equal(nash["A"], util["A"]) or not np.array_equal(nash["A_ref"], util["A_ref"]):
        raise ValueError("The two solves do not share payoff tensors; the comparison would be invalid")
    if not np.array_equal(nash["pi_t"], util["pi_t"]):
        raise ValueError("The two solves do not share the proximal centre")

    A, A_ref, pi_t = nash["A"], nash["A_ref"], nash["pi_t"]
    K, X, I, J = A.shape
    mu = np.full((X, J), 1.0 / J)
    beta = np.full(K, nash["beta"])
    eta = nash["eta"]
    if util["beta"] != nash["beta"] or util["eta"] != nash["eta"]:
        raise ValueError("beta/eta differ between the two solves")

    report = {"objectives": list(OBJECTIVES), "split": args.split, "K": K, "X": X,
              "pool": I, "beta": nash["beta"], "eta": eta,
              "definitions": {
                  "J": "F(pi) - D(pi||pi_t)/eta, Eq. (14) with u_k = s_k at the optimum",
                  "F": "sum_k log s_k(pi)", "D": "mean_x KL(pi_x || pi_t,x)"},
              "artifacts": {n["name"]: {"solution_sha256": n["solution_sha256"],
                                        "tensor_policy_sha256": n["tensor_policy_sha256"],
                                        "aggregation": n["aggregation"],
                                        "lambda_raw": n["weights"].tolist(),
                                        "lambda_l1": float(n["weights"].sum())}
                            for n in (nash, util)},
              "construction_invariants": {
                  "A_ref_skew_residual": float(np.abs(A_ref + np.swapaxes(A_ref, -1, -2)).max()),
                  "A_ref_diagonal_residual": float(np.abs(np.einsum("kxii->kxi", A_ref)).max()),
                  "reading": "imposed by construction; an invariant, not evidence of correctness"}}

    cross = {}
    for owner in (nash, util):
        for candidate in (nash, util):
            key = "pi_%s_under_%s_payoffs" % (candidate["aggregation"], owner["aggregation"])
            if owner is not nash:
                continue                      # payoffs are identical; evaluate once
            cross[candidate["aggregation"]] = evaluate(
                A, A_ref, candidate["pi"], pi_t, mu, beta, eta)
    report["independent_evaluation"] = cross
    report["J_gap_utilitarian_minus_nash"] = (cross["utilitarian"]["J_objective"]
                                              - cross["nash"]["J_objective"])
    report["F_gap_utilitarian_minus_nash"] = (cross["utilitarian"]["F_sum_log_surplus"]
                                              - cross["nash"]["F_sum_log_surplus"])
    report["D_gap_utilitarian_minus_nash"] = (cross["utilitarian"]["D_mean_prompt_kl"]
                                              - cross["nash"]["D_mean_prompt_kl"])
    report["verdict_J"] = (
        "Nash attains the larger J" if report["J_gap_utilitarian_minus_nash"] < 0 else
        "the utilitarian candidate attains a larger J than the claimed Nash optimizer")

    report["stationarity"] = {
        n["aggregation"]: stationarity(A, n["pi"], pi_t, mu, beta, n["weights"], eta)
        for n in (nash, util)}
    report["feasibility"] = {
        n["aggregation"]: {"simplex_max_abs_deviation": float(np.abs(n["pi"].sum(-1) - 1).max()),
                           "min_probability": float(n["pi"].min()),
                           "target_identity_max_abs":
                               float(np.abs(n["g"] - (np.log(n["pi"]) - np.log(pi_t))).max())}
        for n in (nash, util)}

    cert = dual_certificate(A, A_ref, pi_t, mu, beta, nash["weights"], eta)
    pi_lambda = cert.pop("pi")
    cert["pi_of_lambda_vs_stored_linf"] = float(np.abs(pi_lambda - nash["pi"]).max())
    cert["pi_of_lambda_vs_stored_tv_mean"] = float(
        0.5 * np.abs(pi_lambda - nash["pi"]).sum(-1).mean())
    cert["reading"] = ("pi is rebuilt from lambda by the exponential map with no reference to the "
                       "stored policy, so s_k = 1/lambda_k here is a real dual condition rather "
                       "than the identity that defines lambda.")
    cert["applies_on_this_split"] = args.split == "train"
    if args.split != "train":
        cert["dual_residual_interpretation"] = (
            "lambda was fitted on train and is held fixed here by design, so s_k = 1/lambda_k is "
            "NOT expected to hold on this split. This residual measures the train-to-dev surplus "
            "shift, not a solver error. Read the train split for the dual certificate.")
    else:
        cert["dual_residual_interpretation"] = (
            "lambda is fitted on this split, so this is the dual optimality condition.")
    report["nash_dual_certificate_independent"] = cert
    report["solver_reported"] = {n["aggregation"]: n["solver_reported"] for n in (nash, util)}
    report["distances_nash_vs_utilitarian"] = target_distances(nash["pi"], util["pi"], pi_t)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / ("audit_%s_%s_vs_%s.json" % (args.split, args.nash, args.utilitarian))
    path.write_text(json.dumps(report, indent=2) + "\n")

    print(json.dumps({
        "split": args.split,
        "nash": {k: cross["nash"][k] for k in ("F_sum_log_surplus", "D_mean_prompt_kl", "J_objective", "min_surplus")},
        "utilitarian": {k: cross["utilitarian"][k] for k in ("F_sum_log_surplus", "D_mean_prompt_kl", "J_objective", "min_surplus")},
        "J_gap_util_minus_nash": report["J_gap_utilitarian_minus_nash"],
        "F_gap_util_minus_nash": report["F_gap_utilitarian_minus_nash"],
        "D_gap_util_minus_nash": report["D_gap_utilitarian_minus_nash"],
        "verdict": report["verdict_J"],
        "nash_dual_residual_inf": cert["dual_residual_inf"],
        "nash_stationarity_spread_max": report["stationarity"]["nash"]["stationarity_spread_max"],
        "target_tv_mean": report["distances_nash_vs_utilitarian"]["target_tv_mean"],
        "pair_target_diff_rms_over_rms": report["distances_nash_vs_utilitarian"]["pair_target_diff_rms_over_rms_a"],
        "written": str(path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
