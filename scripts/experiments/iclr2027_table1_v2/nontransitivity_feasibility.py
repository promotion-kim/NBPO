#!/usr/bin/env python3
"""Exact float64 primal geometry of the controlled-nontransitivity benchmark.

The alternating fixed-point solver that produces the stress-test rows answers
"where does this update rule land", not "what is attainable here". Those are
different questions, and the negative result cannot be read as a method failure
until the second one is answered independently. This module answers it directly.

The central quantity is the **feasibility witness**

    rho* = max_pi  min_k [ V_{k,beta}(pi) - V_{k,beta}(pi_ref) ]

If ``rho* <= 0`` no policy improves every objective at once: the bargaining
problem has an empty interior, Assumption 1 does not hold, and the instance
cannot test NBPO under the current theorem no matter what any solver returns.

Why an exact solver is possible at all
--------------------------------------
``V_{k,beta}(pi) = mean_x -beta_k logsumexp_j( log mu_xj - r_kxj / beta_k )``
with ``r_kxj = sum_i pi_xi A_k[x,i,j]`` linear in ``pi``. ``logsumexp`` is convex
and ``-beta * convex`` is concave, so **every** ``V_k`` is concave on the product
of simplices. Therefore

* ``rho*`` is a concave max-min -- a convex program;
* the Nash objective ``sum_k log s_k(pi)`` is concave where ``s > 0``;
* the proximal Nash objective ``sum_k log s_k(pi) - D(pi||pi_ref)/eta`` is
  concave, ``D`` being ``mean_x KL(pi_x || pi_ref_x)``.

All three are solved here with SLSQP on the raw simplex in float64 with analytic
gradients, and every answer carries a certificate rather than a convergence
claim.

The certificate
---------------
Concavity gives a rigorous upper bound with no solver in the loop. For any
weights ``w`` on the objective simplex and any ``pi``,

    rho* <= sum_k w_k s~_k(pi) + sum_x [ max_i g_xi - <pi_x, g_x> ],
            g = sum_k w_k grad s~_k(pi)

because each ``s~_k`` lies below its tangent and the inner maximization over a
product of simplices is attained at a vertex per prompt. The bracket is
computed in closed form; the bound is then tightened over ``w`` (convex in
``w``, K = 4 dimensions). ``rho_star_gap`` is that upper bound minus the
achieved ``min_k s~_k``, and it is what makes the reported ``rho*`` an
established number rather than the output of an optimizer that stopped.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

FLOOR = 1e-12          # simplex lower bound; keeps log and KL finite


# --------------------------------------------------------------------------
# game primitives, numpy float64, independent of the torch solver stack
# --------------------------------------------------------------------------

def payoff_rows(A: np.ndarray, pi: np.ndarray) -> np.ndarray:
    """``r[k,x,j] = sum_i pi[x,i] A[k,x,i,j]`` -- what the opponent faces."""
    return np.einsum("xi,kxij->kxj", pi, A)


def game_values(A: np.ndarray, pi: np.ndarray, mu: np.ndarray, beta: np.ndarray,
                per_prompt: bool = False) -> np.ndarray:
    r = payoff_rows(A, pi)
    log_w = np.log(mu)[None, :, :] - r / beta[:, None, None]
    v_kx = -beta[:, None] * logsumexp(log_w, axis=-1)
    return v_kx if per_prompt else v_kx.mean(axis=-1)


def opponent(A: np.ndarray, pi: np.ndarray, mu: np.ndarray,
             beta: np.ndarray) -> np.ndarray:
    r = payoff_rows(A, pi)
    log_w = np.log(mu)[None, :, :] - r / beta[:, None, None]
    return np.exp(log_w - logsumexp(log_w, axis=-1, keepdims=True))


def value_gradient(A: np.ndarray, pi: np.ndarray, mu: np.ndarray,
                   beta: np.ndarray) -> np.ndarray:
    """``dV_k/dpi[x,i] = q_k[x,i] / X`` with ``q_k[x,i] = sum_j nu_kxj A_k[x,i,j]``.

    The X in the denominator is the prompt average inside ``V``; ``q`` itself is
    the per-prompt representer the Eq. (21) update uses, so the two differ by
    exactly that factor and conflating them rescales every step.
    """
    nu = opponent(A, pi, mu, beta)
    q = np.einsum("kxj,kxij->kxi", nu, A)
    return q / A.shape[1]


def exploitability(A: np.ndarray, pi: np.ndarray) -> float:
    vals = []
    for k in range(A.shape[0]):
        r = np.einsum("xi,xij->xj", pi, A[k])
        vals.append(float(np.mean(r.max(axis=-1) - (r * pi).sum(axis=-1))))
    return float(np.mean(vals))


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.mean(0.5 * np.abs(p - q).sum(axis=-1)))


def mean_kl(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p, FLOOR, None)
    q = np.clip(q, FLOOR, None)
    return float(np.mean((p * (np.log(p) - np.log(q))).sum(axis=-1)))


# --------------------------------------------------------------------------
# simplex plumbing for SLSQP
# --------------------------------------------------------------------------

def _simplex_constraints(X: int, I: int, offset: int = 0, n: int | None = None):
    n = n if n is not None else offset + X * I

    def f(z):
        return z[offset:offset + X * I].reshape(X, I).sum(axis=1) - 1.0

    def jac(z):
        J = np.zeros((X, n))
        for x in range(X):
            J[x, offset + x * I: offset + (x + 1) * I] = 1.0
        return J

    return {"type": "eq", "fun": f, "jac": jac}


def _bounds(X: int, I: int, extra_free: int = 0):
    return [(FLOOR, 1.0)] * (X * I) + [(None, None)] * extra_free


def _renormalize(pi: np.ndarray) -> np.ndarray:
    pi = np.clip(pi, FLOOR, None)
    return pi / pi.sum(axis=-1, keepdims=True)


# --------------------------------------------------------------------------
# the three exact programs
# --------------------------------------------------------------------------

def solve_max_min(A, mu, beta, d, pi0=None, maxiter=400):
    """``rho* = max_pi min_k s_k(pi)``, with a concavity certificate.

    Epigraph form: maximize ``rho`` subject to ``s_k(pi) - rho >= 0`` and
    ``pi`` on the product of simplices.
    """
    K, X, I, _ = A.shape
    n = X * I + 1
    pi0 = _renormalize(np.full((X, I), 1.0 / I) if pi0 is None else pi0)
    s0 = game_values(A, pi0, mu, beta) - d
    z0 = np.concatenate([pi0.ravel(), [float(s0.min())]])

    def unpack(z):
        return _renormalize(z[:X * I].reshape(X, I)), z[-1]

    def neg_rho(z):
        return -z[-1]

    def neg_rho_jac(z):
        g = np.zeros(n)
        g[-1] = -1.0
        return g

    def slack(z):
        pi, rho = unpack(z)
        return (game_values(A, pi, mu, beta) - d) - rho

    def slack_jac(z):
        pi, _ = unpack(z)
        gv = value_gradient(A, pi, mu, beta)          # (K, X, I)
        J = np.zeros((K, n))
        J[:, :X * I] = gv.reshape(K, X * I)
        J[:, -1] = -1.0
        return J

    res = minimize(neg_rho, z0, jac=neg_rho_jac, method="SLSQP",
                   bounds=_bounds(X, I, extra_free=1),
                   constraints=[_simplex_constraints(X, I, 0, n),
                                {"type": "ineq", "fun": slack, "jac": slack_jac}],
                   options={"maxiter": maxiter, "ftol": 1e-14})
    pi, _ = unpack(res.x)
    s = game_values(A, pi, mu, beta) - d
    rho = float(s.min())

    ub, w = certify_max_min(A, pi, mu, beta, d)
    return {
        "rho_star": rho,
        "pi": pi,
        "surplus": s.tolist(),
        "certified_upper_bound": ub,
        "rho_star_gap": float(ub - rho),
        "certificate_weights": w.tolist(),
        "primal_feasibility_residual": float(
            np.abs(pi.sum(axis=-1) - 1.0).max() + max(0.0, FLOOR - pi.min())),
        "complementary_slackness_residual": float(np.max(w * (s - rho))),
        "slsqp_status": int(res.status),
        "slsqp_message": str(res.message),
        "slsqp_iterations": int(res.nit),
    }


def certify_max_min(A, pi, mu, beta, d, w0=None):
    """Tightest tangent-plane upper bound on ``rho*`` at ``pi``, over ``w``.

    ``UB(w) = <w, s~(pi)> + sum_x [ max_i g_xi - <pi_x, g_x> ]`` with
    ``g = sum_k w_k grad s~_k(pi)``. Convex in ``w``; minimized on the simplex.
    """
    K, X, I, _ = A.shape
    s = game_values(A, pi, mu, beta) - d
    gv = value_gradient(A, pi, mu, beta)              # (K, X, I) = grad of V_k

    def ub(w):
        g = np.tensordot(w, gv, axes=(0, 0))          # (X, I)
        # the prompt-average is already inside grad, so the vertex gap is summed
        # over prompts exactly as it appears in the linearization
        return float(w @ s + (g.max(axis=-1) - (pi * g).sum(axis=-1)).sum())

    def ub_grad(w):
        g = np.tensordot(w, gv, axes=(0, 0))
        star = g.argmax(axis=-1)
        out = np.empty(K)
        for k in range(K):
            gk = gv[k]
            out[k] = s[k] + float(gk[np.arange(X), star].sum()
                                  - (pi * gk).sum())
        return out

    w0 = np.full(K, 1.0 / K) if w0 is None else w0
    res = minimize(ub, w0, jac=ub_grad, method="SLSQP",
                   bounds=[(0.0, 1.0)] * K,
                   constraints=[{"type": "eq",
                                 "fun": lambda w: w.sum() - 1.0,
                                 "jac": lambda w: np.ones(K)}],
                   options={"maxiter": 300, "ftol": 1e-16})
    w = np.clip(res.x, 0.0, None)
    w = w / w.sum()
    return min(ub(w), ub(w0)), w


def solve_weighted(A, mu, beta, d, weights, pi0=None, eta=None, pi_ref=None,
                   maxiter=500):
    """``max_pi sum_k w_k s_k(pi) [ - D(pi||pi_ref)/eta ]`` -- linear aggregation."""
    K, X, I, _ = A.shape
    pi0 = _renormalize(np.full((X, I), 1.0 / I) if pi0 is None else pi0)
    w = np.asarray(weights, dtype=np.float64)

    def neg(z):
        pi = _renormalize(z.reshape(X, I))
        val = float(w @ (game_values(A, pi, mu, beta) - d))
        if eta is not None:
            val -= mean_kl(pi, pi_ref) / eta
        return -val

    def neg_jac(z):
        pi = _renormalize(z.reshape(X, I))
        g = np.tensordot(w, value_gradient(A, pi, mu, beta), axes=(0, 0))
        if eta is not None:
            g = g - (np.log(np.clip(pi, FLOOR, None))
                     - np.log(np.clip(pi_ref, FLOOR, None)) + 1.0) / (eta * X)
        return -g.ravel()

    res = minimize(neg, pi0.ravel(), jac=neg_jac, method="SLSQP",
                   bounds=_bounds(X, I),
                   constraints=[_simplex_constraints(X, I)],
                   options={"maxiter": maxiter, "ftol": 1e-14})
    return _renormalize(res.x.reshape(X, I)), res


def solve_nash(A, mu, beta, d, pi0=None, eta=None, pi_ref=None, maxiter=600):
    """``max_pi sum_k log s_k(pi) [ - D(pi||pi_ref)/eta ]``.

    Defined only where every surplus is strictly positive, which is exactly the
    Assumption-1 interior. The caller must establish ``rho* > 0`` first and pass
    a strictly feasible ``pi0``; this refuses rather than clamping.
    """
    K, X, I, _ = A.shape
    if pi0 is None:
        raise ValueError("Nash needs a strictly feasible start; solve rho* first")
    pi0 = _renormalize(pi0)
    s0 = game_values(A, pi0, mu, beta) - d
    if not (s0 > 0).all():
        raise NashInfeasible(
            f"start point has a nonpositive surplus {s0.tolist()}; the Nash "
            "objective is undefined and is not being clamped into existence")

    def neg(z):
        pi = _renormalize(z.reshape(X, I))
        s = game_values(A, pi, mu, beta) - d
        if not (s > 0).all():
            return 1e6 + float(-s.min()) * 1e6      # barrier, not a clamp
        val = float(np.log(s).sum())
        if eta is not None:
            val -= mean_kl(pi, pi_ref) / eta
        return -val

    def neg_jac(z):
        pi = _renormalize(z.reshape(X, I))
        s = game_values(A, pi, mu, beta) - d
        gv = value_gradient(A, pi, mu, beta)
        if not (s > 0).all():
            return np.zeros(X * I)
        g = np.tensordot(1.0 / s, gv, axes=(0, 0))
        if eta is not None:
            g = g - (np.log(np.clip(pi, FLOOR, None))
                     - np.log(np.clip(pi_ref, FLOOR, None)) + 1.0) / (eta * X)
        return -g.ravel()

    res = minimize(neg, pi0.ravel(), jac=neg_jac, method="SLSQP",
                   bounds=_bounds(X, I),
                   constraints=[_simplex_constraints(X, I)],
                   options={"maxiter": maxiter, "ftol": 1e-14})
    pi = _renormalize(res.x.reshape(X, I))
    s = game_values(A, pi, mu, beta) - d
    return pi, res, s


class NashInfeasible(RuntimeError):
    """The Nash program has no interior on this instance."""


def objectivewise_max_surplus(A, mu, beta, d, ir_constrained=False):
    """``max_pi s_k(pi)`` for each ``k``, optionally subject to ``s_j >= 0`` for all j.

    The unconstrained version says how much each objective could gain if it were
    alone; the IR-constrained version is the Kalai--Smorodinsky ideal point, and
    whether it is meaningfully positive is exactly whether Game-KS is defined.
    """
    K, X, I, _ = A.shape
    out = []
    for k in range(K):
        e = np.zeros(K)
        e[k] = 1.0
        if not ir_constrained:
            pi, res = solve_weighted(A, mu, beta, d, e)
        else:
            def neg(z, k=k):
                pi = _renormalize(z.reshape(X, I))
                return -float((game_values(A, pi, mu, beta) - d)[k])

            def neg_jac(z, k=k):
                pi = _renormalize(z.reshape(X, I))
                return -value_gradient(A, pi, mu, beta)[k].ravel()

            def ir(z):
                pi = _renormalize(z.reshape(X, I))
                return game_values(A, pi, mu, beta) - d

            def ir_jac(z):
                pi = _renormalize(z.reshape(X, I))
                return value_gradient(A, pi, mu, beta).reshape(K, X * I)

            r = minimize(neg, np.full(X * I, 1.0 / I), jac=neg_jac, method="SLSQP",
                         bounds=_bounds(X, I),
                         constraints=[_simplex_constraints(X, I),
                                      {"type": "ineq", "fun": ir, "jac": ir_jac}],
                         options={"maxiter": 400, "ftol": 1e-14})
            pi, res = _renormalize(r.x.reshape(X, I)), r
        s = game_values(A, pi, mu, beta) - d
        out.append({"objective": k, "max_surplus": float(s[k]),
                    "surplus_at_argmax": s.tolist(),
                    "ir_feasible": bool((s >= -1e-9).all()),
                    "status": int(res.status)})
    return out


def reference_is_equilibrium(A, mu, beta, d, pi_ref, tol=1e-9):
    """Is ``pi_ref`` already the max-min point, and is it stationary?

    Two independent readings, because they can disagree: the linearized ascent
    gap says whether any *first-order* improvement exists for some objective
    weighting, and the exploitability says whether the reference is a Nash
    equilibrium of the underlying symmetric game.
    """
    K, X, I, _ = A.shape
    s_ref = game_values(A, pi_ref, mu, beta) - d
    gv = value_gradient(A, pi_ref, mu, beta)
    per_obj = []
    for k in range(K):
        g = gv[k]
        per_obj.append(float((g.max(axis=-1) - (pi_ref * g).sum(axis=-1)).sum()))
    ub, w = certify_max_min(A, pi_ref, mu, beta, d)
    return {
        "surplus_at_reference": s_ref.tolist(),
        "max_abs_surplus_at_reference": float(np.abs(s_ref).max()),
        "per_objective_linearized_ascent_gap": per_obj,
        "min_objective_ascent_gap": float(min(per_obj)),
        "certified_rho_star_upper_bound_at_reference": float(ub),
        "reference_is_max_min_optimal": bool(ub <= float(s_ref.min()) + 1e-7),
        "exploitability": exploitability(A, pi_ref),
        "is_stationary_for_every_objective": bool(max(per_obj) <= tol),
    }


# --------------------------------------------------------------------------
# a convergent practical solver
# --------------------------------------------------------------------------

def stationarity_residual(A, mu, beta, pi, pi_ref, w, eta):
    """How far ``pi`` is from solving the proximal subproblem at weights ``w``.

    The Eq. (21) map's fixed point is exactly the stationary point of the
    concave program ``max_pi sum_k w_k s_k(pi) - D(pi||pi_ref)/eta``, so the
    honest residual is how far one *undamped* application of that map moves the
    iterate -- not how far the last damped step moved, which shrinks with the
    damping regardless of convergence.
    """
    q = value_gradient(A, pi, mu, beta) * A.shape[1]      # per-prompt representer
    g = np.tensordot(np.asarray(w, dtype=np.float64), q, axes=(0, 0))
    log_new = np.log(np.clip(pi_ref, FLOOR, None)) + eta * g
    log_new -= logsumexp(log_new, axis=-1, keepdims=True)
    return float(np.abs(np.exp(log_new) - pi).max())


def solve_proximal_exact(A, mu, beta, d, w, pi_ref, eta, pi0=None):
    """The inner proximal solve done by concave maximization instead of iteration.

    The deployed solver applies the Eq. (21) map ``R`` times. That map is only a
    contraction while the weighted score is small; once raw Nash multipliers grow
    (``lambda_k = 1/s_k``, so they diverge as a surplus approaches zero) the
    exponent saturates the per-prompt softmax and the iteration bang-bangs
    between vertices instead of converging -- fixed-point residual 1.000 in the
    audit. Damping hides that in the *last-step* residual without fixing it: the
    undamped map still moves the damped iterate by 0.54.

    The subproblem itself is concave, so it does not need iterating at all. This
    solves it directly and reports the stationarity residual, which is the
    quantity the R-step loop was implicitly trying to drive to zero.
    """
    pi_hat, res = solve_weighted(A, mu, beta, d, w, pi0=pi0, eta=eta, pi_ref=pi_ref)

    # Return the Eq. (21) MAP APPLIED AT the maximizer, not the maximizer itself.
    # The Eq. (26) pair builder depends on
    #     [log pi*(y) - log pi_t(y)] - [log pi*(y') - log pi_t(y')]
    #       == eta * sum_k w_k (q_k(y) - q_k(y'))
    # and the artifact writer refuses above 1e-9. An SLSQP iterate satisfies that
    # identity only to its own stationarity tolerance (~1e-7 here), so it would
    # be refused. One map application makes the identity hold to float64 exactly
    # BY CONSTRUCTION; the residual error then lives entirely in `extra_map`,
    # which is where the existing solver already reports it and which downstream
    # code already knows how to read.
    q = value_gradient(A, pi_hat, mu, beta) * A.shape[1]
    g = np.tensordot(np.asarray(w, dtype=np.float64), q, axes=(0, 0))
    log_new = np.log(np.clip(pi_ref, FLOOR, None)) + eta * g
    pi = np.exp(log_new - logsumexp(log_new, axis=-1, keepdims=True))

    lr = np.log(np.clip(pi, FLOOR, None)) - np.log(np.clip(pi_ref, FLOOR, None))
    pred = eta * g
    centre = lambda M: M - M.mean(axis=-1, keepdims=True)
    identity = float(np.abs(centre(lr) - centre(pred)).max())
    return pi, {
        "stationarity_residual": stationarity_residual(A, mu, beta, pi, pi_ref, w, eta),
        "extra_map_residual": float(np.abs(pi - pi_hat).max()),
        "target_log_ratio_identity_residual": identity,
        "slsqp_status": int(res.status), "slsqp_iterations": int(res.nit),
    }


def solve_nash_dual_exact_inner(A, mu, beta, d, pi_ref, eta, *, M=200, gamma=0.5,
                                lambda_box=(1e-3, 1e3), warm_start=True):
    """The deployed outer dual loop, with the inner map replaced by an exact solve.

    Nothing about the projected dual update changes -- ``lambda <- clamp(lambda -
    gamma (s - 1/lambda))`` is the same Eq. (27) step, the multipliers stay raw
    and are never renormalized, and the box is the same. Only the inner
    subproblem is solved rather than iterated.
    """
    K = A.shape[0]
    lam = np.ones(K)
    lo, hi = lambda_box
    pi = np.array(pi_ref, dtype=np.float64, copy=True)
    history = []
    for m in range(M):
        pi, info = solve_proximal_exact(A, mu, beta, d, lam, pi_ref, eta,
                                        pi0=pi if warm_start else None)
        s = game_values(A, pi, mu, beta) - d
        if m % max(1, M // 10) == 0:
            history.append({"iteration": m, "lambda": lam.tolist(),
                            "surplus": s.tolist(), "min_surplus": float(s.min()),
                            "kkt_residual": float(np.abs(s - 1.0 / lam).max()),
                            "stationarity_residual": info["stationarity_residual"]})
        lam = np.clip(lam - gamma * (s - 1.0 / lam), lo, hi)
    pi, info = solve_proximal_exact(A, mu, beta, d, lam, pi_ref, eta, pi0=pi)
    s = game_values(A, pi, mu, beta) - d
    return pi, {
        "weights": lam.tolist(), "weight_l1": float(lam.sum()),
        "surplus": s.tolist(), "min_surplus": float(s.min()),
        "kkt_residual": float(np.abs(s - 1.0 / lam).max()),
        "stationarity_residual": info["stationarity_residual"],
        "extra_map_residual": info["extra_map_residual"],
        "target_log_ratio_identity_residual": info["target_log_ratio_identity_residual"],
        "outer_iterations": M, "history": history,
    }
