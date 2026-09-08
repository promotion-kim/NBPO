"""One finite-pool solver for every matched Table-1 row.

The matched comparison in the manuscript turns on two axes that this module
makes independently selectable:

* **representation** -- how objective ``k`` is valued
  (``mnpo_scripts.nbpo_representations``: ``adaptive_game`` /
  ``fixed_reference`` / ``bt_reward``);
* **aggregation** -- how the ``K`` surpluses are combined into one policy
  update (``nash`` / ``utilitarian`` / ``kalai_smorodinsky``, plus the two
  legacy max-min controls).

Every aggregation solves inside the *same* KL-proximal family

    pi_w = argmax_pi { sum_k w_k s_k(pi) - (1/eta) D(pi || pi_t) }        (Eq. 17)

whose finite-pool solution is closed-form (Eq. (21))

    pi_w(y|x)  proportional to  pi_t(y|x) * exp( eta * sum_k w_k q_k(y|x) ),

so the aggregations differ **only in the weight vector ``w``** and share eta,
the response pool, the tokenization, the target scale and the neural budget.
That is what makes ``NBPO vs Game-utilitarian`` and ``NBPO vs Game-KS`` a
mechanism comparison rather than a step-size comparison.

Consequently the pairwise realization target is *identical in form* for all of
them, and is verified numerically in
``tests/test_nbpo_generic_solver.py``:

    [log pi_w(y|x) - log pi_t(y|x)] - [log pi_w(y'|x) - log pi_t(y'|x)]
        = eta * sum_k w_k ( q_k(y|x) - q_k(y'|x) ).

``eta`` appears once, in the exponent; the artifact stores the UNSCALED weight
vector and the unscaled ``q``, and the trainer applies ``eta`` exactly once.

Kalai--Smorodinsky
------------------
``kalai_smorodinsky`` is the real bargaining rule, not a renamed max-min:

1. **ideal point** ``u_k`` -- the largest surplus objective ``k`` can reach in
   this family while every objective stays individually rational
   (``s_j >= 0`` for all ``j``), computed by dual ascent on the IR multipliers;
2. **Stage 1** ``rho* = max_pi min_k s_k(pi)/u_k`` -- the *ideal-normalized*
   egalitarian problem.  Normalizing by ``u_k`` is exactly what distinguishes
   KS from raw surplus max-min, and it makes the rule invariant to independent
   positive rescaling of the objectives;
3. **Stage 2** -- a lexicographic Pareto refinement: among the Stage-1 optima,
   maximize ``sum_k s_k/u_k``, implemented as a small utilitarian tilt ``tau``
   on the normalized surpluses whose Stage-1 loss is measured and reported;
4. **Stage 3** -- among those, minimize ``D(pi || pi_t)``.  The proximal term
   is *strictly* convex, so the Stage-1/Stage-2 optimum in this family is
   already unique and Stage 3 is satisfied by construction; the achieved
   ``D(pi || pi_t)`` is reported so the claim is checkable.

Deliberate, recorded deviation from a literal reading of the three-stage
program: the stages are solved **inside the eta-proximal family**, not over the
raw simplex.  Over the raw simplex the Stage-3 argmin-KL point sits on a face
of the simplex (zero probability on every non-optimal response), so its
log-ratio target ``log pi* - log pi_t`` is unbounded and no neural realization
of it exists; it would also compare KS against Nash at a completely different
step size.  The spec-literal quantities (a near-unregularized ideal point,
``rho*`` and the individual-rationality violation) are still computed and
written into the solver artifact by ``ks_unregularized_diagnostics`` so the
approximation is measured rather than hidden.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Union

import torch

from mnpo_scripts.nbpo_core import (
    as_float64,
    kl_divergence,
    opponent_entropy,
    opponent_ess,
    uniform_policy,
    validate_distribution,
)
from mnpo_scripts.nbpo_representations import ObjectiveRepresentation
from mnpo_scripts.nbpo_solver import (
    box_active_coordinates,
    projected_kkt_residual,
    proximal_divergence,
    resolve_gamma_schedule,
)

AGGREGATIONS = ("nash", "utilitarian", "kalai_smorodinsky",
                "absolute_maxmin", "surplus_maxmin")

# Aggregations whose weights are a point on (a rescaling of) the simplex, so
# they must be compared at a common L1 norm rather than at whatever norm the
# rule happens to produce.
SIMPLEX_WEIGHT_AGGREGATIONS = ("utilitarian", "kalai_smorodinsky",
                               "absolute_maxmin", "surplus_maxmin")


def exp_update(pi_t: torch.Tensor, q: torch.Tensor, w: torch.Tensor,
               eta: float) -> torch.Tensor:
    """Eq. (21): ``pi(y|x) ~ pi_t(y|x) exp(eta sum_k w_k q_k(y|x))``, in log space.

    Identical to ``nbpo_core.weighted_policy_update`` but accepts ``w_k = 0``
    (a KS or max-min weight vector may put zero mass on an objective, which the
    strictly-positive Nash multipliers never do).  ``eta`` multiplies the score
    exactly once and ``w`` is used raw -- it is never renormalized here.
    """
    pi_t = validate_distribution(pi_t, "pi_t", require_full_support=True)
    q = as_float64(q)
    w = as_float64(w)
    if not torch.isfinite(q).all():
        raise ValueError("q contains non-finite values")
    if w.shape != (q.shape[0],) or not torch.isfinite(w).all() or bool((w < 0).any()):
        raise ValueError("w must be a finite nonnegative vector of length K")
    if not (eta > 0):
        raise ValueError("eta must be strictly positive")
    log_new = torch.log(pi_t) + eta * torch.einsum("k,kxi->xi", w, q)
    log_new = log_new - torch.logsumexp(log_new, dim=-1, keepdim=True)
    return validate_distribution(torch.exp(log_new), "pi_new")


@dataclass
class ProximalSolve:
    """Result of ``pi_w = argmax sum_k w_k s_k(pi) - D(pi||pi_t)/eta``."""

    pi: torch.Tensor
    nu_update: torch.Tensor
    q_update: torch.Tensor
    nu_final_policy: torch.Tensor
    q_final_policy: torch.Tensor
    fixed_point_residual: float
    extra_map_residual: float
    iterations: int
    update_source_pi: torch.Tensor
    update_source_kind: str
    update_source_iteration: int


def solve_proximal(rep: ObjectiveRepresentation, pi_t: torch.Tensor,
                   w: torch.Tensor, eta: float, R: int, *,
                   pi_init: Optional[torch.Tensor] = None,
                   damping: float = 0.0) -> ProximalSolve:
    """Fixed-point solve of Eq. (18) at fixed weights ``w``.

    For a policy-adaptive representation this alternates ``R`` times between
    rebuilding the opponent at the current iterate and applying Eq. (21)
    *centered at the proximal centre* ``pi_t``.

    For a representation whose ``q`` does not depend on the policy
    (``fixed_reference``, ``bt_reward``) the Eq. (21) map is a *constant* map,
    so one application already lands on its fixed point and ``R`` changes
    nothing.  Read ``extra_map_residual`` for that claim: it applies the map
    once more to the returned policy and is exactly ``0`` at every ``R``.
    ``fixed_point_residual`` is the *last-iteration change*, so at ``R = 1`` it
    measures the (nonzero) move away from the proximal centre and only becomes
    ``0`` from ``R >= 2``; it is reported as-is rather than special-cased.
    """
    if R < 1:
        raise ValueError("R must be at least 1")
    if not (0.0 <= damping < 1.0):
        raise ValueError("damping must lie in [0, 1)")
    pi_t = validate_distribution(pi_t, "pi_t", require_full_support=True)
    pi_r = pi_t if pi_init is None else validate_distribution(pi_init, "pi_init")
    residual = float("inf")
    nu = q = None
    source_pi = None
    source_iteration = 0
    for r in range(R):
        nu, q = rep.opponent_and_gradient(pi_r)
        source_pi = pi_r.clone()
        source_iteration = r
        pi_next = exp_update(pi_t, q, w, eta)
        if damping > 0.0:
            pi_next = (1.0 - damping) * pi_next + damping * pi_r
        residual = (pi_next - pi_r).abs().max().item()
        pi_r = pi_next
    if source_iteration > 0:
        source_kind = "fixed_point_iterate"
    elif pi_init is None:
        source_kind = "proximal_centre"
    else:
        source_kind = "warm_start_iterate"
    nu_final, q_final = rep.opponent_and_gradient(pi_r)
    pi_extra = exp_update(pi_t, q_final, w, eta)
    extra = (pi_extra - pi_r).abs().max().item()
    return ProximalSolve(
        pi=pi_r, nu_update=nu, q_update=q, nu_final_policy=nu_final,
        q_final_policy=q_final, fixed_point_residual=residual,
        extra_map_residual=extra, iterations=R, update_source_pi=source_pi,
        update_source_kind=source_kind, update_source_iteration=source_iteration)


_WORKER_STATE: dict = {}


def _worker_init(A, mu, log_pi_t, beta, eta, X, I, floor, maxiter, ftol):
    """Ship the static problem to each worker ONCE, at pool creation.

    The first version passed one payload per prompt on every call. With 60-100
    dual evaluations and thousands of prompts that is hundreds of thousands of
    array pickles, and it cost more than the solves it was parallelizing --
    measured: 42.7 s on 32 workers against 41.1 s serial at 1000 prompts. The
    per-prompt tensors never change within a solve, so they belong in worker
    state and only the dual weights travel per call.
    """
    # Forked children inherit a RESTRICTED CPU affinity mask -- measured here as
    # 2 CPUs each against the parent's 192, because the numeric runtime pins
    # threads in the parent before the fork. The pool then achieves full
    # concurrency (32 chunks genuinely overlap) while every child runs ~30x
    # slower per prompt, so the speedup is exactly cancelled and the parallel
    # path looks useless. Restoring the mask is the whole fix.
    import os
    try:
        os.sched_setaffinity(0, range(os.cpu_count() or 1))
    except (AttributeError, OSError):
        pass
    _WORKER_STATE.update(A=A, mu=mu, log_pi_t=log_pi_t, beta=beta, eta=eta,
                         X=X, I=I, floor=floor, maxiter=maxiter, ftol=ftol)


def _solve_prompt_chunk(task):
    """One contiguous block of prompts, solved with the worker's resident tensors.

    The scipy import is at module scope, not in this function: a per-call import
    cost of about 0.16 s does not amortize over a chunk of a few prompts and was
    measured to swallow the whole parallel gain.
    """
    import numpy as np
    from scipy.optimize import minimize
    lo, hi, w_np, start_block = task
    g = _WORKER_STATE
    A, mu, log_pi_t = g["A"], g["mu"], g["log_pi_t"]
    beta, eta, X, I = g["beta"], g["eta"], g["X"], g["I"]
    floor, maxiter, ftol = g["floor"], g["maxiter"], g["ftol"]
    ones = np.ones(I)
    bounds = [(floor, 1.0)] * I
    out = np.empty((hi - lo, I))

    for n, x in enumerate(range(lo, hi)):
        Ax, lmu, lpt = A[:, x], np.log(mu[x]), log_pi_t[x]

        def negx(v, Ax=Ax, lmu=lmu, lpt=lpt):
            p = np.clip(v, floor, None)
            p = p / p.sum()
            r = np.einsum("i,kij->kj", p, Ax)
            lw = lmu[None, :] - r / beta[:, None]
            m = lw.max(axis=-1, keepdims=True)
            v_k = -beta * (m[:, 0] + np.log(np.exp(lw - m).sum(axis=-1)))
            kl = float((p * (np.log(p) - lpt)).sum())
            return -(float(w_np @ v_k) / X - kl / (eta * X))

        def negx_jac(v, Ax=Ax, lmu=lmu, lpt=lpt):
            p = np.clip(v, floor, None)
            p = p / p.sum()
            r = np.einsum("i,kij->kj", p, Ax)
            lw = lmu[None, :] - r / beta[:, None]
            lw -= lw.max(axis=-1, keepdims=True)
            nu = np.exp(lw)
            nu /= nu.sum(axis=-1, keepdims=True)
            q = np.einsum("kj,kij->ki", nu, Ax)
            gg = (w_np @ q) / X
            gg -= (np.log(p) - lpt + 1.0) / (eta * X)
            return -gg

        res = minimize(negx, start_block[n], jac=negx_jac, method="SLSQP",
                       bounds=bounds,
                       constraints=[{"type": "eq",
                                     "fun": lambda v: np.array([v.sum() - 1.0]),
                                     "jac": lambda v: ones[None, :]}],
                       options={"maxiter": int(maxiter), "ftol": float(ftol)})
        v = np.clip(res.x, floor, None)
        out[n] = v / v.sum()
    return lo, out


_EXECUTORS: dict = {}


def _executor(workers: int, key, init_args):
    """A pool per (worker count, problem), reused across dual evaluations."""
    import atexit
    from concurrent.futures import ProcessPoolExecutor
    cached = _EXECUTORS.get(workers)
    if cached is not None and cached[0] == key:
        return cached[1]
    if cached is not None:
        cached[1].shutdown(wait=False)
    ex = ProcessPoolExecutor(max_workers=workers, initializer=_worker_init,
                             initargs=init_args)
    _EXECUTORS[workers] = (key, ex)
    atexit.register(ex.shutdown, wait=False)
    return ex


def _solve_prompts_parallel(A, mu, log_pi_t, start, beta, w_np, eta, X, I,
                            floor, maxiter, ftol, workers):
    import hashlib
    import numpy as np
    key = hashlib.sha256(np.ascontiguousarray(A).tobytes()).hexdigest()[:32]
    ex = _executor(int(workers), key,
                   (A, mu, log_pi_t, beta, eta, X, I, floor, maxiter, ftol))
    n_chunks = min(X, int(workers))
    edges = np.linspace(0, X, n_chunks + 1).astype(int)
    tasks = [(int(edges[i]), int(edges[i + 1]), w_np, start[edges[i]:edges[i + 1]])
             for i in range(n_chunks) if edges[i + 1] > edges[i]]
    out = np.empty((X, I))
    for lo, block in ex.map(_solve_prompt_chunk, tasks):
        out[lo:lo + block.shape[0]] = block
    return out


def solve_proximal_exact(rep: ObjectiveRepresentation, pi_t: torch.Tensor,
                         w: torch.Tensor, eta: float, *,
                         pi_init: Optional[torch.Tensor] = None,
                         maxiter: int = 400, ftol: float = 1e-16,
                         workers: int = 1) -> ProximalSolve:
    """Solve the Eq. (18) subproblem by maximization instead of iteration.

    ``solve_proximal`` applies the Eq. (21) map ``R`` times. That map contracts
    only while ``eta * sum_k w_k q_k`` stays small; once raw Nash multipliers
    grow -- and they must, since ``lambda_k = 1/s_k`` diverges as a surplus
    approaches zero -- the exponent saturates the per-prompt softmax and the
    iteration bang-bangs between vertices instead of converging. The audit in
    `scripts/experiments/iclr2027_table1_v2` measures exactly that: a fixed-point
    residual of 1.000, the largest a simplex iterate can have, and a policy
    ``TV = 0.64`` away from the true proximal solution.

    The subproblem does not need iterating, and it does not need a large solver
    either. ``V_{k,beta}(pi) = mean_x v_{k,x}(pi_x)`` and each ``v_{k,x}`` depends
    on ``pi`` **only through that prompt's row**, while the proximal term is a sum
    of per-prompt KLs. So

        J_w(pi) = sum_x [ (1/X) sum_k w_k v_{k,x}(pi_x) - KL(pi_x||pi_t,x)/(eta X) ]

    **separates completely across prompts**, and the whole thing is X independent
    concave programs over the I-simplex rather than one program over X*I
    variables. That is what makes this usable at 7000 prompts: cost is linear in
    X, each subproblem has a handful of variables, and they are embarrassingly
    parallel. ``logsumexp`` is convex and ``-beta * convex`` is concave, so each
    piece is concave and its maximizer is unique.

    The return is **the Eq. (21) map applied at the maximizer**, not the
    maximizer, so that

        [log pi*(y) - log pi_t(y)] - [log pi*(y') - log pi_t(y')]
            == eta * sum_k w_k (q_k(y) - q_k(y'))

    holds to float64 exactly -- the identity the Eq. (26) pair builder depends on
    and the artifact writer refuses above 1e-9. An optimizer iterate satisfies it
    only to its own tolerance and would be refused; the residual error instead
    lands in ``extra_map_residual``, where the existing solver already reports it.

    For a representation whose ``q`` does not depend on the policy
    (``fixed_reference``, ``bt_reward``) one application of the map is already the
    exact maximizer, so this delegates rather than invoking an optimizer that has
    nothing to do.
    """
    import numpy as np
    from scipy.optimize import minimize

    pi_t = validate_distribution(pi_t, "pi_t", require_full_support=True)
    w = as_float64(w)
    if not (eta > 0):
        raise ValueError("eta must be strictly positive")
    if not rep.policy_adaptive:
        return solve_proximal(rep, pi_t, w, eta, R=1, pi_init=pi_init)

    X, I, K = rep.X, rep.I, rep.K
    floor = 1e-12
    A = rep.A.numpy()                       # (K, X, I, J)
    beta = rep.beta.numpy()
    mu = rep.mu.numpy()
    w_np = w.numpy()
    pi_t_np = pi_t.numpy()
    log_pi_t = np.log(np.clip(pi_t_np, floor, None))
    start = (pi_t_np if pi_init is None
             else validate_distribution(pi_init, "pi_init").numpy())

    ones = np.ones(I)
    cons = [{"type": "eq", "fun": lambda v: np.array([v.sum() - 1.0]),
             "jac": lambda v: ones[None, :]}]
    bounds = [(floor, 1.0)] * I
    out = np.empty((X, I))

    if workers and workers > 1 and X >= 2 * workers:
        # Infrastructure-level parallelism only: the same per-prompt program, the
        # same tolerance, evaluated on more cores. The prompts are independent by
        # construction, which is the whole reason this solve scales.
        out = _solve_prompts_parallel(A, mu, log_pi_t, start, beta, w_np, eta, X, I,
                                      floor, maxiter, ftol, workers)
        pi_hat = torch.from_numpy(out)
        nu, q = rep.opponent_and_gradient(pi_hat)
        pi_star = exp_update(pi_t, q, w, eta)
        nu_final, q_final = rep.opponent_and_gradient(pi_star)
        pi_extra = exp_update(pi_t, q_final, w, eta)
        return ProximalSolve(
            pi=pi_star, nu_update=nu, q_update=q,
            nu_final_policy=nu_final, q_final_policy=q_final,
            fixed_point_residual=float((pi_star - pi_hat).abs().max()),
            extra_map_residual=float((pi_extra - pi_star).abs().max()),
            iterations=X, update_source_pi=pi_hat.clone(),
            update_source_kind="exact_proximal_maximizer", update_source_iteration=0)

    for x in range(X):
        Ax = A[:, x]                        # (K, I, J)
        mux = mu[x]
        lmu = np.log(mux)
        lpt = log_pi_t[x]

        def negx(v, Ax=Ax, lmu=lmu, lpt=lpt):
            p = np.clip(v, floor, None)
            p = p / p.sum()
            r = np.einsum("i,kij->kj", p, Ax)
            lw = lmu[None, :] - r / beta[:, None]
            m = lw.max(axis=-1, keepdims=True)
            v_k = -beta * (m[:, 0] + np.log(np.exp(lw - m).sum(axis=-1)))
            kl = float((p * (np.log(p) - lpt)).sum())
            return -(float(w_np @ v_k) / X - kl / (eta * X))

        def negx_jac(v, Ax=Ax, lmu=lmu, lpt=lpt):
            p = np.clip(v, floor, None)
            p = p / p.sum()
            r = np.einsum("i,kij->kj", p, Ax)
            lw = lmu[None, :] - r / beta[:, None]
            lw -= lw.max(axis=-1, keepdims=True)
            nu = np.exp(lw)
            nu /= nu.sum(axis=-1, keepdims=True)
            q = np.einsum("kj,kij->ki", nu, Ax)          # (K, I)
            g = (w_np @ q) / X
            g -= (np.log(p) - lpt + 1.0) / (eta * X)
            return -g

        res = minimize(negx, start[x], jac=negx_jac, method="SLSQP",
                       bounds=bounds, constraints=cons,
                       options={"maxiter": int(maxiter), "ftol": float(ftol)})
        v = np.clip(res.x, floor, None)
        out[x] = v / v.sum()

    pi_hat = torch.from_numpy(out)
    nu, q = rep.opponent_and_gradient(pi_hat)
    pi_star = exp_update(pi_t, q, w, eta)
    nu_final, q_final = rep.opponent_and_gradient(pi_star)
    pi_extra = exp_update(pi_t, q_final, w, eta)
    return ProximalSolve(
        pi=pi_star, nu_update=nu, q_update=q,
        nu_final_policy=nu_final, q_final_policy=q_final,
        fixed_point_residual=float((pi_star - pi_hat).abs().max()),
        extra_map_residual=float((pi_extra - pi_star).abs().max()),
        iterations=X, update_source_pi=pi_hat.clone(),
        update_source_kind="exact_proximal_maximizer", update_source_iteration=0)


def _converged_solve(rep, pi_t, w, eta, *, R=400, damping=0.5, tol=1e-12):
    """Inner solve run to (near) fixed-point convergence, for the KS sub-problems."""
    sol = solve_proximal(rep, pi_t, w, eta, R, damping=damping)
    return sol


# --------------------------------------------------------------------------
# Kalai--Smorodinsky
# --------------------------------------------------------------------------

@dataclass
class KSResult:
    weights: torch.Tensor                 # (K,) simplex weights realizing the KS point
    u_ideal: torch.Tensor                 # (K,) individually rational ideal surpluses
    rho_star: float
    surplus: torch.Tensor                 # (K,) at the returned policy
    normalized_surplus: torch.Tensor      # (K,) s_k / u_k
    pi: torch.Tensor
    solve: ProximalSolve
    ir_violation: float                   # max_k max(0, -s_k); 0 iff individually rational
    ideal_feasible: bool
    stage1_residual: float                # rho_star - min_k s_k/u_k at the returned policy
    stage2_gain: float                    # sum_k s_k/u_k improvement from the Pareto tilt
    stage3_kl: float                      # D(pi || pi_t)
    tau: float
    history: List[dict] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)


class KSUndefinedError(ValueError):
    """Raised when the Kalai--Smorodinsky problem has no ideal point above ``d``.

    Carries ``diagnostics`` so the caller can *record* the degeneracy (which
    objective, what its best reachable surplus was, whether individual
    rationality was feasible at all) instead of clamping a nonpositive surplus
    and reporting a bargaining solution that does not exist.  A nonpositive
    ideal surplus also means the Nash program is infeasible on the same pool --
    it is a property of the stage, not of the KS rule.
    """

    def __init__(self, message: str, diagnostics: dict):
        super().__init__(message)
        self.diagnostics = diagnostics


def _proximal(rep, pi_t, w, eta, R, *, pi_init=None, damping=0.0,
              inner_solver: str = "fixed_point"):
    """One inner proximal solve, honouring the selected inner solver.

    Every sub-solve inside Kalai--Smorodinsky routes through here. Before this
    existed, `solve_finite_pool(..., inner_solver="exact")` repaired the Nash
    branch while KS silently kept iterating the Eq. (21) map internally, so the
    Game-KS row inherited exactly the divergence the repair removes: on the
    controlled benchmark its inner residual reached 1.000 and its solution
    violated individual rationality, which a bargaining solution must not do.
    """
    if inner_solver == "exact":
        return solve_proximal_exact(rep, pi_t, w, eta, pi_init=pi_init)
    return solve_proximal(rep, pi_t, w, eta, R, pi_init=pi_init, damping=damping)


def _proximal_vertices(rep, pi_t, eta, R, damping, weight_l1,
                       inner_solver="fixed_point"):
    """``s(pi)`` at each weight-simplex vertex ``weight_l1 * e_k``: the attainable ideal."""
    K = rep.K
    out = []
    for k in range(K):
        e_k = torch.zeros(K, dtype=torch.float64)
        e_k[k] = weight_l1
        out.append(rep.surplus(_proximal(rep, pi_t, e_k, eta, R, damping=damping,
                                         inner_solver=inner_solver).pi))
    return out


def _ir_constrained_max(rep, pi_t, eta, k: int, R: int, *, iters: int, step: float,
                        inner_solver: str = "fixed_point",
                        damping: float, weight_l1: float, scale: float,
                        seed_points):
    """``max_pi s_k(pi)`` in the proximal family subject to ``s_j(pi) >= 0`` for all j.

    Two quantities are produced and both are reported:

    * the **vertex** value -- ``s_k`` at the weight vector ``weight_l1 * e_k``,
      i.e. the most objective ``k`` can gain when the whole matched weight
      budget is spent on it.  This is the ideal point of the attainable set at
      this step size, ignoring individual rationality;
    * the **individually rational** value -- the best ``s_k`` over every weight
      vector visited, which is the spec's ``u_k_ideal``.  The search is a dual
      ascent on the IR multipliers ``m_j >= 0`` (inner weights ``e_k + m``,
      rescaled to the common L1 norm ``weight_l1``), started from ``m = 1``
      (which is the equal-weight point tilted toward ``k``) so that a feasible
      region reachable near the centre of the simplex is not missed by a search
      that only ever leaves from the vertex.  ``seed_points`` adds explicitly
      supplied weight vectors (the uniform one) to the same feasibility scan.

    The dual step is divided by ``scale`` -- the largest attainable ideal
    surplus -- because surpluses on a centered-preference pool are O(1e-2) and
    an unscaled step of order one would move the multipliers by nothing.

    When no individually rational point is found the fact is returned as
    ``ir_feasible = False`` together with the vertex value, so the artifact says
    the ideal point was taken on the unconstrained attainable set.  It is never
    silently substituted.
    """
    K = rep.K
    e_k = torch.zeros(K, dtype=torch.float64)
    e_k[k] = weight_l1
    u_vertex = float(rep.surplus(_proximal(rep, pi_t, e_k, eta, R, damping=damping,
                                           inner_solver=inner_solver).pi)[k])

    best_ir = None
    worst_violation = float("inf")

    def _scan(s):
        nonlocal best_ir, worst_violation
        worst_violation = min(worst_violation, float(torch.clamp(-s, min=0.0).max()))
        if bool((s >= -1e-12).all()) and (best_ir is None or float(s[k]) > best_ir):
            best_ir = float(s[k])

    for s_seed in seed_points:
        _scan(s_seed)

    m = torch.ones(K, dtype=torch.float64)
    pi_warm = None
    eff_step = step / max(scale, 1e-12)
    for _ in range(iters):
        w = torch.zeros(K, dtype=torch.float64)
        w[k] = 1.0
        w = w + m
        w_eff = w * (weight_l1 / float(w.sum()))
        sol = _proximal(rep, pi_t, w_eff, eta, R, pi_init=pi_warm, damping=damping,
                        inner_solver=inner_solver)
        pi_warm = sol.pi
        s = rep.surplus(sol.pi)
        _scan(s)
        m = torch.clamp(m - eff_step * s, min=0.0)
    if best_ir is not None:
        return best_ir, True, "individually_rational_proximal_max", u_vertex, worst_violation
    return u_vertex, False, "unconstrained_proximal_vertex", u_vertex, worst_violation


def solve_kalai_smorodinsky(
    rep: ObjectiveRepresentation,
    pi_t: torch.Tensor,
    eta: float,
    R: int = 1,
    *,
    weight_l1: float = 1.0,
    ideal_iters: int = 200,
    ideal_step: float = 1.0,
    stage1_iters: int = 2000,
    adversary_step: float = 1.0,
    tau: float = 1e-3,
    tolerance: float = 1e-4,
    ks_degeneracy_rel: float = 1e-6,
    damping: float = 0.0,
    log_every: int = 0,
    inner_solver: str = "fixed_point",
) -> KSResult:
    """The Kalai--Smorodinsky bargaining solution on the frozen finite pool.

    See the module docstring for the exact three stages and for the recorded
    deviation (the stages are solved inside the eta-proximal family).
    """
    K = rep.K
    pi_t = validate_distribution(pi_t, "pi_t", require_full_support=True)

    # ---- ideal point ------------------------------------------------------
    # Vertices first: they give both the unconstrained ideal and the scale the
    # IR dual step must be measured in.
    vertex_surpluses = _proximal_vertices(rep, pi_t, eta, R, damping, weight_l1,
                                          inner_solver=inner_solver)
    uniform_w = torch.full((K,), weight_l1 / K, dtype=torch.float64)
    seed_points = list(vertex_surpluses) + [
        rep.surplus(_proximal(rep, pi_t, uniform_w, eta, R, damping=damping,
                              inner_solver=inner_solver).pi)]
    scale = max(float(vertex_surpluses[k][k].abs()) for k in range(K))

    u = torch.zeros(K, dtype=torch.float64)
    u_vertex = torch.zeros(K, dtype=torch.float64)
    ideal_feasible = True
    ideal_definitions, ir_gaps = [], []
    for k in range(K):
        u_k, feas, definition, vertex_k, gap_k = _ir_constrained_max(
            rep, pi_t, eta, k, R, iters=ideal_iters, step=ideal_step,
            damping=damping, weight_l1=weight_l1, scale=scale,
            seed_points=seed_points,
            inner_solver=inner_solver)
        u[k] = u_k
        u_vertex[k] = vertex_k
        ideal_definitions.append(definition)
        ir_gaps.append(gap_k)
        ideal_feasible = ideal_feasible and feas
    # A merely POSITIVE ideal is not enough: if the individually rational set
    # collapses onto the disagreement point (an exactly zero-sum pool, say) then
    # u_k is positive only by float noise, and dividing by it turns rho* into
    # garbage that would still look like a solved bargaining problem. Judge u_k
    # against the attainable range, not against zero.
    ideal_scale = float(u_vertex.abs().max())
    degenerate = u <= max(ideal_scale, 1e-300) * ks_degeneracy_rel
    if bool(degenerate.any()):
        bad = [int(i) for i in degenerate.nonzero().flatten()]
        diagnostics = {
            "u_ideal": [float(v) for v in u],
            "u_vertex": [float(v) for v in u_vertex],
            "nonpositive_objectives": bad,
            "ideal_point_ir_feasible": ideal_feasible,
            "ideal_point_definitions": ideal_definitions,
            "best_ir_violation_seen": ir_gaps,
            "attainable_ideal_scale": ideal_scale,
            "degeneracy_relative_floor": float(ks_degeneracy_rel),
            "eta": float(eta),
            "weight_l1": float(weight_l1),
        }
        raise KSUndefinedError(
            "Kalai-Smorodinsky is undefined on this pool: the individually rational ideal "
            f"surplus is nonpositive or numerically degenerate for objective(s) {bad} "
            f"(u = {[float(v) for v in u]}, attainable scale {ideal_scale:.3e}, relative "
            f"floor {ks_degeneracy_rel:g}). No policy in the eta-proximal family lifts "
            "those objectives meaningfully above the disagreement point, so the bargaining "
            "set above d is empty or a single point -- and the Nash program is infeasible on "
            "the same pool. Reported, not clamped.", diagnostics)

    # ---- Stage 1: rho* = max_pi min_k s_k(pi)/u_k -------------------------
    # Saddle point of a concave-in-pi, linear-in-w problem: the adversary runs
    # Hedge on the ideal-NORMALIZED surpluses (this normalization is what makes
    # the rule Kalai-Smorodinsky rather than raw surplus max-min), and the
    # AVERAGED weights are the ones that converge.
    w = torch.full((K,), 1.0 / K, dtype=torch.float64)
    w_sum = torch.zeros(K, dtype=torch.float64)
    pi_warm = None
    history: List[dict] = []
    for m_it in range(stage1_iters):
        w_eff = w * (weight_l1 / float(w.sum()))
        sol = _proximal(rep, pi_t, w_eff, eta, R, pi_init=pi_warm, damping=damping,
                        inner_solver=inner_solver)
        pi_warm = sol.pi
        s = rep.surplus(sol.pi)
        v = s / u
        w_sum += w
        if log_every and (m_it % log_every == 0):
            history.append({"stage": 1, "iteration": int(m_it),
                            "weights": [float(a) for a in w],
                            "surplus": [float(a) for a in s],
                            "normalized_surplus": [float(a) for a in v],
                            "min_normalized": float(v.min())})
        step = adversary_step / ((m_it + 1) ** 0.5)
        log_w = torch.log(w) - step * v
        w = torch.exp(log_w - torch.logsumexp(log_w, dim=0))
    w_bar = w_sum / stage1_iters
    w_bar = w_bar / w_bar.sum()
    sol1 = _proximal(rep, pi_t, w_bar * weight_l1, eta, R, pi_init=pi_warm,
                     inner_solver=inner_solver,
                          damping=damping)
    s1 = rep.surplus(sol1.pi)
    rho_star = float((s1 / u).min())

    # ---- Stage 2: Pareto refinement on the Stage-1 optimal face -----------
    # Lexicographic: among Stage-1 optima maximize sum_k s_k/u_k. Implemented as
    # a small utilitarian tilt on the NORMALIZED surpluses; the Stage-1 loss it
    # costs is measured and must stay within `tolerance`.
    w_tilt = w_bar + tau * (1.0 / u) / float((1.0 / u).sum())
    w_tilt = w_tilt / w_tilt.sum()
    sol2 = _proximal(rep, pi_t, w_tilt * weight_l1, eta, R, pi_init=sol1.pi,
                     inner_solver=inner_solver,
                          damping=damping)
    s2 = rep.surplus(sol2.pi)
    rho2 = float((s2 / u).min())
    stage2_gain = float((s2 / u).sum() - (s1 / u).sum())
    if (rho_star - rho2) <= tolerance and stage2_gain > 0.0:
        final_w, final_sol, final_s = w_tilt, sol2, s2
    else:
        final_w, final_sol, final_s = w_bar, sol1, s1
        stage2_gain = 0.0
    stage1_residual = rho_star - float((final_s / u).min())

    ir_violation = float(torch.clamp(-final_s, min=0.0).max())
    stage3_kl = proximal_divergence(final_sol.pi, pi_t)
    # How nearly the rule actually equalized the ideal-normalized surpluses. This
    # is KS's defining property, and it is not the same quantity as the Stage-1
    # residual: a solve can sit exactly on its own rho* estimate while the
    # normalized surpluses are still spread out, if the saddle point was under-
    # converged. Reported so the two cannot be confused.
    normalized_spread = float((final_s / u).max() - (final_s / u).min())
    return KSResult(
        weights=final_w * weight_l1,
        u_ideal=u,
        rho_star=rho_star,
        surplus=final_s,
        normalized_surplus=final_s / u,
        pi=final_sol.pi,
        solve=final_sol,
        ir_violation=ir_violation,
        ideal_feasible=ideal_feasible,
        stage1_residual=stage1_residual,
        stage2_gain=stage2_gain,
        stage3_kl=stage3_kl,
        tau=float(tau),
        history=history,
        diagnostics={
            "normalized_surplus_spread": normalized_spread,
            "stage1_residual_sign_note": (
                "rho_star minus the achieved min normalized surplus. NEGATIVE means the "
                "returned point beats the Stage-1 estimate, which happens when the "
                "Stage-1 saddle-point average was conservative and the Stage-2 tilt "
                "improved on it -- not a constraint violation. Positive values above "
                "`tolerance` are the ones that matter."),
            "stage1_weights": [float(a) for a in w_bar],
            "stage2_weights": [float(a) for a in w_tilt],
            "stage2_applied": bool(stage2_gain > 0.0),
            "tolerance": float(tolerance),
            "weight_l1": float(weight_l1),
            "stage1_iterations": int(stage1_iters),
            "ideal_iterations": int(ideal_iters),
            "stage3_selection": ("unique by strict convexity of D(pi||pi_t) in the "
                                 "eta-proximal family; achieved value reported"),
            "u_vertex": [float(v) for v in u_vertex],
            "ideal_point_definitions": ideal_definitions,
            "best_ir_violation_seen": ir_gaps,
        },
    )


def ks_unregularized_diagnostics(rep: ObjectiveRepresentation, pi_t: torch.Tensor,
                                 eta: float, R: int, *, eta_scale: float = 1e3,
                                 **kwargs) -> dict:
    """The spec-literal KS quantities, measured at a nearly unregularized step.

    Reported so the deviation recorded in the module docstring is quantified
    rather than asserted: the same three stages are run with the proximal term
    weakened by ``eta_scale``, which approaches the raw-simplex bargaining
    problem.  These numbers are diagnostics only -- they never produce a
    training target, because the raw-simplex Stage-3 point has an unbounded
    log-ratio.
    """
    try:
        res = solve_kalai_smorodinsky(rep, pi_t, eta * eta_scale, R, **kwargs)
    except KSUndefinedError as exc:
        return {"status": "undefined", "reason": str(exc),
                "eta_scale": float(eta_scale), **exc.diagnostics}
    return {
        "status": "ok",
        "eta_scale": float(eta_scale),
        "eta_used": float(eta * eta_scale),
        "u_ideal": [float(v) for v in res.u_ideal],
        "rho_star": res.rho_star,
        "surplus": [float(v) for v in res.surplus],
        "normalized_surplus": [float(v) for v in res.normalized_surplus],
        "individual_rationality_violation": res.ir_violation,
        "ideal_point_ir_feasible": res.ideal_feasible,
        "stage1_feasibility_residual": res.stage1_residual,
        "stage2_pareto_gain": res.stage2_gain,
        "kl_to_proximal_centre": res.stage3_kl,
    }


def matched_weight_l1(nash_solution: "FinitePoolSolution") -> float:
    """The common L1 weight norm every matched control must be solved at.

    NBPO's multipliers are raw (``lambda_k ~ 1/s_k``), so their L1 norm is of
    order ``sum_k 1/s_k`` -- hundreds, not one.  A control solved with weights on
    the simplex therefore takes a step hundreds of times smaller and would lose
    the comparison on step size alone, which is exactly the "undisclosed unequal
    budget" the protocol forbids.

    The matched protocol is: solve NBPO first, keep its lambda **raw** (Eq. (21)
    consumes it unchanged, satisfying the non-negotiable that lambda is never
    normalized), and pass this number as ``weight_l1`` to every other
    aggregation so all rows differ only in the *direction* of the weight vector.
    """
    return float(nash_solution.weights.sum())


# --------------------------------------------------------------------------
# Unified entry point
# --------------------------------------------------------------------------

@dataclass
class FinitePoolSolution:
    """Everything the generic finite-pool target artifact needs (Section 3)."""

    representation: str
    aggregation: str
    weights: torch.Tensor          # (K,) RAW aggregation/KKT weights used in Eq. (21)
    V: torch.Tensor                # (K,)
    d: torch.Tensor                # (K,)
    surplus: torch.Tensor          # (K,) raw, never clamped
    pi_t: torch.Tensor             # (X, I) proximal centre
    pi: torch.Tensor               # (X, I) solved finite-pool policy pi_star
    q_update: torch.Tensor         # (K, X, I) response-level objective scores
    nu_update: torch.Tensor        # (K, X, J)
    nu_final_policy: torch.Tensor  # (K, X, J)
    target_log_ratio: torch.Tensor # (X, I) log pi_star - log pi_t
    eta: float
    R: int
    fixed_point_residual: float
    extra_map_residual: float
    kkt_residual: Optional[float] = None
    projected_kkt_residual: Optional[float] = None
    control_residual: Optional[float] = None
    outer_iterations_used: Optional[int] = None
    dual_converged: Optional[bool] = None
    inner_iterations_last: Optional[int] = None
    gamma_ref: Optional[float] = None
    lambda_at_lower_bound: List[int] = field(default_factory=list)
    lambda_at_upper_bound: List[int] = field(default_factory=list)
    update_source_pi: Optional[torch.Tensor] = None
    update_source_kind: str = "proximal_centre"
    update_source_iteration: int = 0
    opponent_entropy: Optional[torch.Tensor] = None
    opponent_ess: Optional[torch.Tensor] = None
    proximal_kl: float = float("nan")
    ks: Optional[dict] = None
    history: List[dict] = field(default_factory=list)
    config: dict = field(default_factory=dict)

    def target_log_ratio_check(self) -> float:
        """max |(log pi* - log pi_t) - eta sum_k w_k q_k + const| over the pool.

        Zero (to numerical tolerance) iff the stored weights and ``q`` reproduce
        the solved policy, which is the identity the pair builder relies on.
        """
        score = self.eta * torch.einsum("k,kxi->xi", self.weights, self.q_update)
        diff = self.target_log_ratio - score
        return float((diff - diff.mean(dim=-1, keepdim=True)).abs().max())


def _solve_nash_dual_by_root(rep, pi_t, d, lam0, eta, R, inner, damping, lo, hi,
                             *, tol, max_calls, history, log_every):
    """Solve the Nash dual as a ROOT problem instead of a subgradient descent.

    The dual stationarity condition of the Nash aggregation is exactly

        s_k(pi(lambda)) = 1 / lambda_k      for every k strictly inside the box,

    and ``pi(lambda)`` is a well-defined function once the inner subproblem is
    solved rather than iterated. So this is a K-dimensional root problem -- four
    dimensions here -- and a projected subgradient with a constant step is a poor
    way to attack it: on the controlled benchmark it stalls near a residual of
    1e-2 after 3000 outer iterations, nowhere near the 1e-6 gate.

    Solved in log-lambda coordinates so positivity is structural rather than
    enforced by clipping, warm-started across evaluations so the inner solves stay
    cheap, and reported with the actual number of inner solves used.
    """
    import numpy as np
    from scipy.optimize import root

    state = {"pi": None, "calls": 0}

    def residual(u):
        lam = np.clip(np.exp(u), lo, hi)
        sol = inner(rep, pi_t, torch.from_numpy(lam), eta, R,
                    pi_init=state["pi"], damping_=damping)
        state["pi"] = sol.pi
        state["calls"] += 1
        s = (rep.game_values(sol.pi) - d).numpy()
        if log_every and state["calls"] % max(1, log_every) == 0:
            history.append({"iteration": state["calls"], "lambda_raw": lam.tolist(),
                            "surplus": s.tolist(), "min_surplus": float(s.min()),
                            "kkt_residual": float(np.abs(s - 1.0 / lam).max())})
        return s - 1.0 / lam

    u0 = np.log(np.clip(as_float64(lam0).numpy(), lo, hi))
    res = root(residual, u0, method="hybr",
               options={"xtol": 1e-14, "maxfev": int(max_calls)})
    lam = torch.from_numpy(np.clip(np.exp(res.x), lo, hi))
    # Convergence is decided by a FRESHLY MEASURED residual at the returned
    # multipliers, not by the optimizer's own success flag. `hybr` reports
    # failure whenever it stops making progress, which it does as soon as the
    # residual reaches the inner solve's own noise floor -- exactly where the
    # answer is correct. Trusting `res.success` would mark good solves as failed
    # and, worse, could mark a bad one as good if the flag ever disagreed.
    final_res = float(np.abs(residual(np.log(lam.numpy()))).max())

    # One deterministic restart when the first solve stops short. `lambda = 1/s`
    # at the current point is the natural better start -- it is the stationarity
    # condition itself -- and re-solving from there costs a few dozen inner
    # solves against the tens of thousands a subgradient would need. If it still
    # misses, the miss is reported; the tolerance is not moved to accommodate it.
    if final_res >= tol:
        s_here = (rep.game_values(state["pi"]) - d).numpy() if state["pi"] is not None else None
        if s_here is not None and (s_here > 0).all():
            res2 = root(residual, np.log(np.clip(1.0 / s_here, lo, hi)),
                        method="hybr", options={"xtol": 1e-14, "maxfev": int(max_calls)})
            lam2 = torch.from_numpy(np.clip(np.exp(res2.x), lo, hi))
            r2 = float(np.abs(residual(np.log(lam2.numpy()))).max())
            if r2 < final_res:
                lam, final_res = lam2, r2
    return lam, state["pi"], state["calls"], bool(final_res < tol), final_res




def solve_finite_pool(
    rep: ObjectiveRepresentation,
    aggregation: str,
    *,
    eta: float,
    pi_t: Optional[torch.Tensor] = None,
    R: int = 1,
    M: int = 1,
    gamma: Union[float, Sequence[float]] = 0.5,
    lambda_box: Sequence[float] = (1e-3, 1e3),
    lambda_init: Optional[torch.Tensor] = None,
    warm_start_policy: bool = True,
    damping: float = 0.0,
    adversary_step: float = 1.0,
    weight_l1: Optional[float] = None,
    log_every: int = 0,
    ks_kwargs: Optional[dict] = None,
    inner_solver: str = "fixed_point",
    dual_tol: Optional[float] = None,
    dual_solver: str = "subgradient",
    inner_workers: int = 1,
) -> FinitePoolSolution:
    """Solve one outer stage for any (representation, aggregation) pair.

    ``inner_solver`` selects how the Eq. (18) subproblem is solved at fixed
    weights. ``"fixed_point"`` -- the default, and what every existing artifact
    was produced with -- applies the Eq. (21) map ``R`` times. ``"exact"``
    maximizes the concave subproblem directly, which the controlled-
    nontransitivity audit showed is necessary once raw multipliers grow: at
    ``alpha = 1`` the ``R``-step map has a fixed-point residual of 1.000 and
    lands at min surplus -0.185, while the exact inner solve reaches +0.011
    against an attainable ``rho* = +0.015``. The default is unchanged so no
    existing result moves silently.
    """
    if inner_solver not in ("fixed_point", "exact"):
        raise ValueError("inner_solver must be 'fixed_point' or 'exact'")
    if dual_tol is not None and not (dual_tol > 0):
        raise ValueError("dual_tol must be strictly positive when given")
    if dual_solver not in ("subgradient", "root"):
        raise ValueError("dual_solver must be 'subgradient' or 'root'")

    def _inner(rep_, pi_t_, w_, eta_, R_, *, pi_init=None, damping_=0.0):
        if inner_solver == "exact":
            return solve_proximal_exact(rep_, pi_t_, w_, eta_, pi_init=pi_init,
                                        workers=inner_workers)
        return solve_proximal(rep_, pi_t_, w_, eta_, R_, pi_init=pi_init,
                              damping=damping_)

    if aggregation not in AGGREGATIONS:
        raise ValueError(f"aggregation must be one of {AGGREGATIONS}, got {aggregation!r}")
    K, X, I = rep.K, rep.X, rep.I
    if pi_t is None:
        pi_t = uniform_policy(X, I)
    pi_t = validate_distribution(pi_t, "pi_t", require_full_support=True)
    d = rep.disagreement
    lo, hi = float(lambda_box[0]), float(lambda_box[1])
    gamma_sched = resolve_gamma_schedule(gamma, M)
    history: List[dict] = []
    ks_block = None
    control_residual = None
    outer_used = None
    dual_converged = None
    kkt = proj_res = gamma_ref = None
    lower_active: List[int] = []
    upper_active: List[int] = []

    if aggregation == "kalai_smorodinsky":
        kw = dict(ks_kwargs or {})
        kw.setdefault("weight_l1", 1.0 if weight_l1 is None else float(weight_l1))
        kw.setdefault("damping", damping)
        kw.setdefault("adversary_step", adversary_step)
        kw.setdefault("log_every", log_every)
        kw.setdefault("inner_solver", inner_solver)
        ks = solve_kalai_smorodinsky(rep, pi_t, eta, R, **kw)
        w = ks.weights
        final = ks.solve
        history = ks.history
        ks_block = {
            "u_ideal": [float(v) for v in ks.u_ideal],
            "rho_star": ks.rho_star,
            "normalized_surplus": [float(v) for v in ks.normalized_surplus],
            "individual_rationality_violation": ks.ir_violation,
            "ideal_point_ir_feasible": ks.ideal_feasible,
            "stage1_feasibility_residual": ks.stage1_residual,
            "stage2_pareto_gain": ks.stage2_gain,
            "stage3_kl": ks.stage3_kl,
            "tau": ks.tau,
            **ks.diagnostics,
        }
    elif aggregation == "utilitarian":
        w = torch.full((K,), 1.0 / K, dtype=torch.float64)
        if weight_l1 is not None:
            w = w * (float(weight_l1) / float(w.sum()))
        final = _inner(rep, pi_t, w, eta, R, damping_=damping)
    elif aggregation == "nash":
        lam = (torch.ones(K, dtype=torch.float64) if lambda_init is None
               else as_float64(lambda_init).clone())
        if lam.shape != (K,) or bool((lam <= 0).any()):
            raise ValueError("lambda_init must be a strictly positive vector of length K")
        pi_warm = None
        outer_used = 0
        if dual_solver == "root":
            lam, pi_warm, outer_used, dual_converged, _root_res = _solve_nash_dual_by_root(
                rep, pi_t, d, lam, eta, R, _inner, damping, lo, hi,
                tol=(dual_tol if dual_tol is not None else 1e-10), max_calls=M,
                history=history, log_every=log_every)
            w = lam
            final = _inner(rep, pi_t, w, eta, R, pi_init=pi_warm, damping_=damping)
            s_fin = rep.game_values(final.pi) - d
            gamma_ref = float(gamma_sched[0])
            kkt = float((s_fin - 1.0 / w).abs().max())
            proj_res = projected_kkt_residual(w, s_fin, gamma_ref, lo, hi)
            lower_active, upper_active = box_active_coordinates(w, lo, hi)
            _nash_done = True
        else:
            _nash_done = False
        for m_it in (range(M) if not _nash_done else range(0)):
            sol = _inner(rep, pi_t, lam, eta, R, pi_init=pi_warm, damping_=damping)
            if warm_start_policy:
                pi_warm = sol.pi
            s = rep.game_values(sol.pi) - d
            outer_used = m_it + 1
            if log_every and (m_it % log_every == 0):
                history.append({"iteration": int(m_it),
                                "lambda_raw": [float(v) for v in lam],
                                "surplus": [float(v) for v in s],
                                "min_surplus": float(s.min()),
                                "kkt_residual": float((s - 1.0 / lam).abs().max())})
            # Stop on the DUAL residual, not on an iteration count. The natural-map
            # residual is the right test because it vanishes at a projected
            # stationary point even when a box bound is active, where
            # ||s - 1/lambda|| is not expected to.
            if dual_tol is not None:
                r_now = projected_kkt_residual(lam, s, float(gamma_sched[m_it]), lo, hi)
                if r_now < dual_tol:
                    dual_converged = True
                    break
            lam = torch.clamp(lam - gamma_sched[m_it] * (s - 1.0 / lam), min=lo, max=hi)
        if not _nash_done:
            if dual_tol is not None and dual_converged is None:
                dual_converged = False
            w = lam
            final = _inner(rep, pi_t, w, eta, R, pi_init=pi_warm, damping_=damping)
            s_fin = rep.game_values(final.pi) - d
            gamma_ref = float(gamma_sched[min(outer_used, M) - 1])
            kkt = float((s_fin - 1.0 / w).abs().max())
            proj_res = projected_kkt_residual(w, s_fin, gamma_ref, lo, hi)
            lower_active, upper_active = box_active_coordinates(w, lo, hi)
    else:  # absolute_maxmin / surplus_maxmin
        l1 = 1.0 if weight_l1 is None else float(weight_l1)
        wv = torch.full((K,), 1.0 / K, dtype=torch.float64)
        w_sum = torch.zeros(K, dtype=torch.float64)
        pi_sum = torch.zeros_like(pi_t)
        pi_warm = None
        for m_it in range(M):
            sol = _inner(rep, pi_t, wv * l1, eta, R, pi_init=pi_warm, damping_=damping)
            if warm_start_policy:
                pi_warm = sol.pi
            V = rep.game_values(sol.pi)
            s = V - d
            v = V if aggregation == "absolute_maxmin" else s
            w_sum += wv
            pi_sum += sol.pi
            step = adversary_step / ((m_it + 1) ** 0.5)
            log_w = torch.log(wv) - step * v
            wv = torch.exp(log_w - torch.logsumexp(log_w, dim=0))
        w_bar = w_sum / M
        pi_bar = pi_sum / M
        pi_bar = pi_bar / pi_bar.sum(dim=-1, keepdim=True)
        w = w_bar * l1
        final = _inner(rep, pi_t, w, eta, R, pi_init=pi_bar, damping_=damping)
        V_bar = rep.game_values(pi_bar)
        v_bar = V_bar if aggregation == "absolute_maxmin" else V_bar - d
        V_br = rep.game_values(final.pi)
        v_br = V_br if aggregation == "absolute_maxmin" else V_br - d
        control_residual = (float((w_bar * v_br).sum()) - proximal_divergence(final.pi, pi_t) / eta
                            - (float(v_bar.min()) - proximal_divergence(pi_bar, pi_t) / eta))

    V = rep.game_values(final.pi)
    s = V - d
    log_ratio = torch.log(final.pi) - torch.log(pi_t)
    return FinitePoolSolution(
        representation=rep.type,
        aggregation=aggregation,
        weights=w,
        V=V, d=d, surplus=s,
        pi_t=pi_t, pi=final.pi,
        q_update=final.q_update,
        nu_update=final.nu_update,
        nu_final_policy=final.nu_final_policy,
        target_log_ratio=log_ratio,
        eta=float(eta), R=int(R),
        fixed_point_residual=final.fixed_point_residual,
        extra_map_residual=final.extra_map_residual,
        kkt_residual=kkt,
        projected_kkt_residual=proj_res,
        control_residual=control_residual,
        outer_iterations_used=outer_used,
        dual_converged=dual_converged,
        inner_iterations_last=getattr(final, "iterations", None),
        gamma_ref=gamma_ref,
        lambda_at_lower_bound=lower_active,
        lambda_at_upper_bound=upper_active,
        update_source_pi=final.update_source_pi,
        update_source_kind=final.update_source_kind,
        update_source_iteration=final.update_source_iteration,
        opponent_entropy=opponent_entropy(final.nu_final_policy),
        opponent_ess=opponent_ess(final.nu_final_policy),
        proximal_kl=proximal_divergence(final.pi, pi_t),
        ks=ks_block,
        history=history,
        config={
            "representation": rep.info().__dict__,
            "aggregation": aggregation,
            "inner_solver": inner_solver,
            "dual_tol": dual_tol,
            "dual_solver": dual_solver,
            "inner_workers": inner_workers,
            # Top-level `beta` is what the Eq. (26) pair builder reads. It is the
            # opponent temperature, so it exists only for a representation that
            # HAS an adaptive opponent; fixed_reference and bt_reward record None
            # rather than a number they do not have, and the pair rows then carry
            # a null opponent_beta instead of a false one.
            "beta": rep.info().beta,
            "eta": float(eta),
            "eta_applications": 1,
            "eta_applied_in": "exp_update (Eq. 21) exponent; the trainer applies it "
                              "to the pairwise target exactly once",
            "R": int(R),
            "R_is_approximation": bool(R == 1 and rep.policy_adaptive),
            "M": int(M),
            "gamma": [float(g) for g in gamma_sched],
            "lambda_box": [lo, hi],
            "warm_start_policy": bool(warm_start_policy),
            "damping": float(damping),
            "adversary_step": float(adversary_step),
            "weight_l1": (None if weight_l1 is None else float(weight_l1)),
            "weights_are_raw": True,
            "weights_normalized_for_training": False,
        },
    )
