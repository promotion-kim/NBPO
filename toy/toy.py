
"""
Toy comparison for SPPO, INPO, MNPO, and RONPO on a heterogeneous matrix game.

This script creates a small discrete preference-optimization problem with multiple
conflicting preference oracles, then compares four idealized algorithms:

1. SPPO-avg:
   - Uses a fixed averaged oracle.
   - Uses the current policy as the opponent.
   - Performs a practical SPPO-style centered multiplicative update:
       pi_{t+1}(y) ∝ pi_t(y) exp(alpha * (P_bar(y ≻ pi_t) - 1/2)).
   - Since the -1/2 term is a common constant in tabular normalization, it does
     not change the exact normalized distribution, but it is kept in the code
     to reflect SPPO's centered target.

2. INPO-avg:
   - Uses a fixed averaged oracle.
   - Uses the current policy as the opponent.
   - Adds KL anchoring toward the reference policy.

3. MNPO-hist-avg:
   - Uses a fixed averaged oracle.
   - Uses a population of historical policies as opponents.
   - Uses the geometric mean of recent policies as the multiplicative prior.

4. RONPO:
   - Treats heterogeneous objectives and opponent actions as an adversary.
   - Maintains sigma(k, a), a distribution over objective k and adversarial action a.
   - Optimizes worst-case robustness.

Metrics
-------
- worst_case_win_rate(pi) = min_{k,a} E_{y~pi}[P_k(y beats a)]
- robust_gap(pi) = robust_optimum_value - worst_case_win_rate(pi)
- avg_ref_win_rate(pi) = average_k E_{y~pi,a~uniform}[P_k(y beats a)]

The robust optimum is computed by a small linear program when scipy is available.
If scipy is not installed, a multiplicative-weights fallback approximation is used.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


EPS = 1e-12


def normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    total = x.sum()
    if total <= 0:
        return np.ones_like(x) / len(x)
    return x / total


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - np.max(logits)
    return normalize(np.exp(logits))


def entropy(p: np.ndarray) -> float:
    p = np.clip(p, EPS, 1.0)
    return float(-np.sum(p * np.log(p)))


def set_pair(P: np.ndarray, i: int, j: int, p: float) -> None:
    """Set P[i,j]=p and P[j,i]=1-p."""
    if i == j:
        P[i, j] = 0.5
    else:
        P[i, j] = p
        P[j, i] = 1.0 - p


def make_conflicting_preferences(
    n_actions: int = 8,
    n_objectives: int = 3,
    seed: int = 0,
    specialist_strength: float = 0.82,
    decoy_good: float = 0.78,
    decoy_bad: float = 0.06,
    background_strength: float = 0.56,
) -> np.ndarray:
    """
    Construct heterogeneous pairwise preference matrices P[k, i, j].

    P[k, i, j] is the probability that action i beats action j under objective k.

    The construction intentionally includes:
    - objective-specific specialist actions: action k is strong for objective k.
    - a decoy action: good for objectives 0 and 1 but terrible for objective 2.
      This makes average-oracle methods attractive but weak in worst-case metrics.
    """
    rng = np.random.default_rng(seed)
    P = np.full((n_objectives, n_actions, n_actions), 0.5, dtype=np.float64)

    # Mild random anti-symmetric background tournament around 0.5.
    for k in range(n_objectives):
        for i in range(n_actions):
            for j in range(i + 1, n_actions):
                p = background_strength if rng.random() < 0.5 else 1.0 - background_strength
                set_pair(P[k], i, j, p)

    # Objective-specific specialists.
    specialists = list(range(min(n_objectives, n_actions)))
    for k, champ in enumerate(specialists):
        for a in range(n_actions):
            if a != champ:
                set_pair(P[k], champ, a, specialist_strength)

    # Decoy: attractive under the average of objectives 0 and 1, but bad under objective 2.
    if n_actions >= n_objectives + 1 and n_objectives >= 3:
        decoy = n_objectives
        for a in range(n_actions):
            if a == decoy:
                continue
            set_pair(P[0], decoy, a, decoy_good)
            set_pair(P[1], decoy, a, decoy_good)
            set_pair(P[2], decoy, a, decoy_bad)

    # Enforce exact anti-symmetry and self-tie.
    for k in range(n_objectives):
        np.fill_diagonal(P[k], 0.5)
        for i in range(n_actions):
            for j in range(i + 1, n_actions):
                P[k, j, i] = 1.0 - P[k, i, j]

    return P


def robust_value(pi: np.ndarray, P: np.ndarray) -> float:
    """
    Hard robust value:
        min_{objective k, opponent action a} E_{y~pi} P_k(y beats a).
    """
    values = np.einsum("y,kya->ka", pi, P)
    return float(values.min())


def avg_ref_win_rate(pi: np.ndarray, P: np.ndarray, ref: np.ndarray | None = None) -> float:
    """
    Average win rate against a fixed reference opponent distribution.
    """
    n_actions = P.shape[1]
    if ref is None:
        ref = np.ones(n_actions) / n_actions
    vals = np.einsum("y,a,kya->k", pi, ref, P)
    return float(vals.mean())


def per_objective_ref_win_rates(pi: np.ndarray, P: np.ndarray, ref: np.ndarray | None = None) -> np.ndarray:
    n_actions = P.shape[1]
    if ref is None:
        ref = np.ones(n_actions) / n_actions
    return np.einsum("y,a,kya->k", pi, ref, P)


def approximate_robust_optimum_by_mw(
    P: np.ndarray,
    n_steps: int = 20000,
    alpha_pi: float = 0.3,
    alpha_sigma: float = 0.3,
) -> Tuple[np.ndarray, float]:
    """
    Fallback approximation to:
        max_pi min_{k,a} E_pi[P_k(y,a)].
    """
    n_objectives, n_actions, _ = P.shape
    pi = np.ones(n_actions) / n_actions
    sigma = np.ones((n_objectives, n_actions)) / (n_objectives * n_actions)

    for _ in range(n_steps):
        r = np.einsum("ka,kya->y", sigma, P)
        pi = softmax(np.log(np.clip(pi, EPS, 1.0)) + alpha_pi * r)

        c = np.einsum("y,kya->ka", pi, P)
        sigma = np.exp(np.log(np.clip(sigma, EPS, 1.0)) - alpha_sigma * c)
        sigma = sigma / sigma.sum()

    return pi, robust_value(pi, P)


def solve_robust_optimum(P: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Solve:
        max_{pi in simplex} v
        s.t. sum_y pi_y P[k,y,a] >= v for all k,a.

    Uses scipy when available. Falls back to a MW approximation otherwise.
    """
    try:
        from scipy.optimize import linprog

        n_objectives, n_actions, _ = P.shape
        n_vars = n_actions + 1  # pi plus v
        v_idx = n_actions

        c = np.zeros(n_vars)
        c[v_idx] = -1.0  # minimize -v

        A_ub = []
        b_ub = []
        for k in range(n_objectives):
            for a in range(n_actions):
                row = np.zeros(n_vars)
                row[:n_actions] = -P[k, :, a]
                row[v_idx] = 1.0
                A_ub.append(row)
                b_ub.append(0.0)

        A_eq = np.zeros((1, n_vars))
        A_eq[0, :n_actions] = 1.0
        b_eq = np.array([1.0])

        bounds = [(0.0, 1.0)] * n_actions + [(0.0, 1.0)]
        res = linprog(
            c,
            A_ub=np.array(A_ub),
            b_ub=np.array(b_ub),
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
        )
        if not res.success:
            raise RuntimeError(res.message)

        pi_star = normalize(res.x[:n_actions])
        v_star = float(res.x[v_idx])
        return pi_star, v_star

    except Exception:
        return approximate_robust_optimum_by_mw(P)


@dataclass
class RunHistory:
    name: str
    policies: List[np.ndarray]
    worst_case: List[float]
    robust_gap: List[float]
    avg_ref: List[float]
    entropy: List[float]


def record_metrics(
    name: str,
    policies: List[np.ndarray],
    P: np.ndarray,
    robust_opt_value: float,
    ref: np.ndarray,
) -> RunHistory:
    worst = [robust_value(pi, P) for pi in policies]
    gap = [robust_opt_value - w for w in worst]
    avg = [avg_ref_win_rate(pi, P, ref=ref) for pi in policies]
    ent = [entropy(pi) for pi in policies]
    return RunHistory(name=name, policies=policies, worst_case=worst, robust_gap=gap, avg_ref=avg, entropy=ent)


def run_sppo(
    P: np.ndarray,
    n_iter: int,
    alpha: float,
    ref: np.ndarray,
    objective_weights: np.ndarray | None = None,
) -> List[np.ndarray]:
    """
    Idealized SPPO-style baseline with an averaged oracle.

    SPPO's practical target is:
        log(pi_{t+1}(y)/pi_t(y)) ≈ alpha * (P_bar(y ≻ pi_t) - 1/2).

    In a tabular exact update, the common -1/2 shift vanishes after normalization,
    but we keep it in the logits to reflect the centered target.
    """
    n_objectives, _, _ = P.shape
    if objective_weights is None:
        objective_weights = np.ones(n_objectives) / n_objectives

    P_bar = np.einsum("k,kij->ij", objective_weights, P)
    pi = ref.copy()
    policies = [pi.copy()]

    for _ in range(n_iter):
        r = P_bar @ pi
        logits = np.log(np.clip(pi, EPS, 1.0)) + alpha * (r - 0.5)
        pi = softmax(logits)
        policies.append(pi.copy())

    return policies


def run_inpo(
    P: np.ndarray,
    n_iter: int,
    alpha: float,
    tau: float,
    ref: np.ndarray,
    objective_weights: np.ndarray | None = None,
) -> List[np.ndarray]:
    """
    Idealized INPO-style baseline:
      - Uses a fixed scalarized oracle P_bar = sum_k w_k P_k.
      - Uses current policy as opponent.
      - KL-anchors policy toward reference ref.
    """
    n_objectives, _, _ = P.shape
    if objective_weights is None:
        objective_weights = np.ones(n_objectives) / n_objectives

    P_bar = np.einsum("k,kij->ij", objective_weights, P)
    pi = ref.copy()
    policies = [pi.copy()]

    for _ in range(n_iter):
        r = P_bar @ pi
        logits = (
            (1.0 - alpha * tau) * np.log(np.clip(pi, EPS, 1.0))
            + alpha * tau * np.log(np.clip(ref, EPS, 1.0))
            + alpha * r
        )
        pi = softmax(logits)
        policies.append(pi.copy())

    return policies


def run_mnpo(
    P: np.ndarray,
    n_iter: int,
    alpha: float,
    ref: np.ndarray,
    history_len: int = 4,
    objective_weights: np.ndarray | None = None,
) -> List[np.ndarray]:
    """
    Idealized TD-MNPO-style baseline:
      - Uses a fixed scalarized oracle P_bar.
      - Uses a population of historical policies as opponents.
      - Uses the geometric mean of recent policies as the multiplicative prior.
    """
    n_objectives, _, _ = P.shape
    if objective_weights is None:
        objective_weights = np.ones(n_objectives) / n_objectives

    P_bar = np.einsum("k,kij->ij", objective_weights, P)
    pi = ref.copy()
    policies = [pi.copy()]

    for _ in range(n_iter):
        recent = policies[-history_len:]
        opponents = [ref] + recent
        weights = np.ones(len(opponents)) / len(opponents)

        # Arithmetic opponent mixture for sampling.
        nu = normalize(sum(w * p for w, p in zip(weights, opponents)))

        # Geometric mixture prior.
        log_prior = sum(w * np.log(np.clip(p, EPS, 1.0)) for w, p in zip(weights, opponents))

        r = P_bar @ nu
        pi = softmax(log_prior + alpha * r)
        policies.append(pi.copy())

    return policies


def run_ronpo(
    P: np.ndarray,
    n_iter: int,
    alpha_pi: float,
    alpha_sigma: float,
    tau: float,
    kappa: float,
    ref: np.ndarray,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    RONPO:
      - Maintains sigma over (objective k, adversarial action a).
      - Policy update:
          pi_{t+1}(y) ∝ pi_t(y)^(1-alpha*tau) ref(y)^(alpha*tau) exp(alpha*r_t(y))
        where r_t(y)=E_{(k,a)~sigma_t} P_k(y,a).
      - Adversary update:
          sigma_{t+1}(k,a) ∝ sigma_t(k,a)^(1-alpha*kappa)
                                  sigma_0(k,a)^(alpha*kappa)
                                  exp(-alpha_sigma*c_t(k,a))
        where c_t(k,a)=E_{y~pi_{t+1}} P_k(y,a).
    """
    n_objectives, n_actions, _ = P.shape
    pi = ref.copy()
    sigma = np.ones((n_objectives, n_actions)) / (n_objectives * n_actions)
    sigma0 = sigma.copy()

    policies = [pi.copy()]
    sigmas = [sigma.copy()]

    for _ in range(n_iter):
        r = np.einsum("ka,kya->y", sigma, P)
        logits_pi = (
            (1.0 - alpha_pi * tau) * np.log(np.clip(pi, EPS, 1.0))
            + alpha_pi * tau * np.log(np.clip(ref, EPS, 1.0))
            + alpha_pi * r
        )
        pi = softmax(logits_pi)

        c = np.einsum("y,kya->ka", pi, P)
        logits_sigma = (
            (1.0 - alpha_sigma * kappa) * np.log(np.clip(sigma, EPS, 1.0))
            + alpha_sigma * kappa * np.log(np.clip(sigma0, EPS, 1.0))
            - alpha_sigma * c
        )
        sigma = np.exp(logits_sigma - np.max(logits_sigma))
        sigma = sigma / sigma.sum()

        policies.append(pi.copy())
        sigmas.append(sigma.copy())

    return policies, sigmas


def plot_metric_curves(histories: Dict[str, RunHistory], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    for name, hist in histories.items():
        plt.plot(hist.robust_gap, label=name)
    plt.xlabel("Iteration")
    plt.ylabel("Robust gap: robust optimum - worst-case win rate")
    plt.title("Robust Gap, Lower Is Better")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "toy_robust_gap.png", dpi=200)
    plt.show()

    plt.figure(figsize=(8, 5))
    for name, hist in histories.items():
        plt.plot(hist.worst_case, label=name)
    plt.xlabel("Iteration")
    plt.ylabel("Worst-case win rate")
    plt.title("Worst-case Win Rate, Higher Is Better")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "toy_worst_case_win_rate.png", dpi=200)
    plt.show()

    plt.figure(figsize=(8, 5))
    for name, hist in histories.items():
        plt.plot(hist.avg_ref, label=name)
    plt.xlabel("Iteration")
    plt.ylabel("Average win rate against uniform reference")
    plt.title("Average Reference Win Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "toy_avg_ref_win_rate.png", dpi=200)
    plt.show()


def plot_final_policies(histories: Dict[str, RunHistory], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    names = list(histories.keys())
    n_actions = len(next(iter(histories.values())).policies[-1])
    x = np.arange(n_actions)
    width = 0.8 / len(names)

    plt.figure(figsize=(9, 5))
    for idx, name in enumerate(names):
        pi = histories[name].policies[-1]
        plt.bar(x + idx * width, pi, width=width, label=name)
    plt.xticks(x + width * (len(names) - 1) / 2, [f"a{i}" for i in range(n_actions)])
    plt.xlabel("Action")
    plt.ylabel("Final policy probability")
    plt.title("Final Policy Distributions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "toy_final_policies.png", dpi=200)
    plt.show()


def plot_ronpo_adversary(sigmas: List[np.ndarray], outdir: Path) -> None:
    """
    Optional diagnostic: show the final RONPO adversary mass over objectives.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    final_sigma = sigmas[-1]
    objective_mass = final_sigma.sum(axis=1)

    plt.figure(figsize=(7, 4))
    plt.bar(np.arange(len(objective_mass)), objective_mass)
    plt.xlabel("Objective index k")
    plt.ylabel("Final adversary mass")
    plt.title("RONPO Final Adversary Mass over Objectives")
    plt.tight_layout()
    plt.savefig(outdir / "toy_ronpo_final_adversary_objective_mass.png", dpi=200)
    plt.show()


def print_summary(
    histories: Dict[str, RunHistory],
    P: np.ndarray,
    robust_opt_value: float,
    ref: np.ndarray,
    robust_pi_star: np.ndarray,
) -> None:
    print("\n=== Robust optimum ===")
    print(f"robust optimum value: {robust_opt_value:.4f}")
    print(f"robust optimum policy: {np.round(robust_pi_star, 4)}")

    print("\n=== Final metrics ===")
    for name, hist in histories.items():
        pi = hist.policies[-1]
        per_obj = per_objective_ref_win_rates(pi, P, ref=ref)
        print(f"\n{name}")
        print(f"  final policy: {np.round(pi, 4)}")
        print(f"  worst-case win rate: {hist.worst_case[-1]:.4f}")
        print(f"  robust gap: {hist.robust_gap[-1]:.4f}")
        print(f"  avg ref win rate: {hist.avg_ref[-1]:.4f}")
        print(f"  per-objective ref win rates: {np.round(per_obj, 4)}")
        print(f"  entropy: {hist.entropy[-1]:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_actions", type=int, default=8)
    parser.add_argument("--n_objectives", type=int, default=3)
    parser.add_argument("--n_iter", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--alpha_sppo", type=float, default=1.2)
    parser.add_argument("--alpha_inpo", type=float, default=1.2)
    parser.add_argument("--alpha_mnpo", type=float, default=1.2)
    parser.add_argument("--alpha_ronpo_pi", type=float, default=1.2)
    parser.add_argument("--alpha_ronpo_sigma", type=float, default=1.5)

    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--kappa", type=float, default=0.05)
    parser.add_argument("--history_len", type=int, default=4)
    parser.add_argument("--outdir", type=str, default="toy_ronpo_outputs")
    args = parser.parse_args()

    P = make_conflicting_preferences(
        n_actions=args.n_actions,
        n_objectives=args.n_objectives,
        seed=args.seed,
    )
    ref = np.ones(args.n_actions) / args.n_actions

    robust_pi_star, robust_opt_value = solve_robust_optimum(P)

    sppo_policies = run_sppo(
        P=P,
        n_iter=args.n_iter,
        alpha=args.alpha_sppo,
        ref=ref,
    )
    inpo_policies = run_inpo(
        P=P,
        n_iter=args.n_iter,
        alpha=args.alpha_inpo,
        tau=args.tau,
        ref=ref,
    )
    mnpo_policies = run_mnpo(
        P=P,
        n_iter=args.n_iter,
        alpha=args.alpha_mnpo,
        ref=ref,
        history_len=args.history_len,
    )
    ronpo_policies, ronpo_sigmas = run_ronpo(
        P=P,
        n_iter=args.n_iter,
        alpha_pi=args.alpha_ronpo_pi,
        alpha_sigma=args.alpha_ronpo_sigma,
        tau=args.tau,
        kappa=args.kappa,
        ref=ref,
    )

    histories = {
        "SPPO-avg": record_metrics("SPPO-avg", sppo_policies, P, robust_opt_value, ref),
        "INPO-avg": record_metrics("INPO-avg", inpo_policies, P, robust_opt_value, ref),
        "MNPO-hist-avg": record_metrics("MNPO-hist-avg", mnpo_policies, P, robust_opt_value, ref),
        "RONPO": record_metrics("RONPO", ronpo_policies, P, robust_opt_value, ref),
    }

    outdir = Path(args.outdir)
    print_summary(histories, P, robust_opt_value, ref, robust_pi_star)
    plot_metric_curves(histories, outdir)
    plot_final_policies(histories, outdir)
    plot_ronpo_adversary(ronpo_sigmas, outdir)

    print(f"\nSaved figures to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
