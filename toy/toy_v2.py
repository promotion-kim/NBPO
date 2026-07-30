"""
Priority toy experiments for RONPO.

This script extends the original toy.py comparison with the first experiments that
are most useful before moving to LLM-scale experiments:

1. single
   - Reproduce the original decoy game comparison.
   - Adds an idealized heterogeneous MNPO baseline.

2. decoy_sweep
   - Sweep the hidden bad-objective severity of the decoy action.
   - Tests whether averaged-oracle methods collapse to a response that is good on
     average but weak for one objective.

3. kappa_sweep
   - Sweep the RONPO adversary KL parameter kappa.
   - Produces the average-vs-minimum trade-off curve.

4. stochastic_batch
   - Compare exact RONPO to a partition-free two-query stochastic policy update.
   - The stochastic update fits tabular log-probabilities from pairwise relative
     labels z_y - z_y'.

5. random_tournament
   - Run random heterogeneous antisymmetric tournaments across several seeds and
     correlation levels.
   - This is a cherry-picking defense: the decoy example is not the only setting.

The original toy.py already contained SPPO-avg, INPO-avg, MNPO-hist-avg, exact
RONPO, and the hard robust metrics.  This file keeps those pieces and adds the
priority ablations.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EPS = 1e-12


# -----------------------------------------------------------------------------
# Basic utilities
# -----------------------------------------------------------------------------


def normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    total = float(x.sum())
    if total <= 0 or not np.isfinite(total):
        return np.ones_like(x, dtype=np.float64) / len(x)
    return x / total


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - np.max(logits)
    return normalize(np.exp(logits))


def entropy(p: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0)
    return float(-np.sum(p * np.log(p)))


def kl(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0)
    q = np.clip(np.asarray(q, dtype=np.float64), EPS, 1.0)
    return float(np.sum(p * (np.log(p) - np.log(q))))


def set_pair(P: np.ndarray, i: int, j: int, p: float) -> None:
    """Set P[i,j]=p and P[j,i]=1-p."""
    p = float(np.clip(p, EPS, 1.0 - EPS))
    if i == j:
        P[i, j] = 0.5
    else:
        P[i, j] = p
        P[j, i] = 1.0 - p


# -----------------------------------------------------------------------------
# Preference game construction
# -----------------------------------------------------------------------------


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

    Design:
    - action k is a specialist for objective k;
    - action d = n_objectives is a decoy: good for objectives 0 and 1, bad for
      objective 2.  This makes averaged-oracle methods attractive but weak in
      worst-case metrics.
    """
    if n_actions < n_objectives + 1:
        raise ValueError("n_actions must be at least n_objectives + 1 for the decoy construction.")
    if n_objectives < 3:
        raise ValueError("decoy construction expects at least 3 objectives.")

    rng = np.random.default_rng(seed)
    P = np.full((n_objectives, n_actions, n_actions), 0.5, dtype=np.float64)

    # Mild random anti-symmetric background tournament around 0.5.
    for k in range(n_objectives):
        for i in range(n_actions):
            for j in range(i + 1, n_actions):
                p = background_strength if rng.random() < 0.5 else 1.0 - background_strength
                set_pair(P[k], i, j, p)

    # Objective-specific specialists.
    for k in range(n_objectives):
        champ = k
        for a in range(n_actions):
            if a != champ:
                set_pair(P[k], champ, a, specialist_strength)

    # Decoy: attractive under objectives 0 and 1, terrible under objective 2.
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


def make_random_antisymmetric_preferences(
    n_actions: int,
    n_objectives: int,
    seed: int,
    beta: float = 4.0,
    rho: float = 0.0,
) -> np.ndarray:
    """
    Random heterogeneous tournament.

    Let M_k = rho M_0 + sqrt(1-rho^2) E_k.  Each M_k is anti-symmetric, and
    P_k(i,j)=sigmoid(beta M_k(i,j)).

    rho=1 gives homogeneous objectives; rho=0 gives independent objectives.
    """
    rng = np.random.default_rng(seed)

    def random_skew() -> np.ndarray:
        M = rng.normal(size=(n_actions, n_actions))
        M = M - M.T
        np.fill_diagonal(M, 0.0)
        # Keep scale reasonably stable as n varies.
        M = M / (np.std(M[np.triu_indices(n_actions, 1)]) + EPS)
        return M

    M0 = random_skew()
    P = np.empty((n_objectives, n_actions, n_actions), dtype=np.float64)
    rho = float(np.clip(rho, 0.0, 1.0))
    for k in range(n_objectives):
        Ek = random_skew()
        Mk = rho * M0 + math.sqrt(max(0.0, 1.0 - rho * rho)) * Ek
        P[k] = 1.0 / (1.0 + np.exp(-beta * Mk))
        np.fill_diagonal(P[k], 0.5)
        for i in range(n_actions):
            for j in range(i + 1, n_actions):
                P[k, j, i] = 1.0 - P[k, i, j]
    return P


# -----------------------------------------------------------------------------
# Metrics and robust optimum
# -----------------------------------------------------------------------------


def robust_value(pi: np.ndarray, P: np.ndarray) -> float:
    """Hard minimum over objective k and opponent action a."""
    values = np.einsum("y,kya->ka", pi, P)
    return float(values.min())


def per_objective_worst_values(pi: np.ndarray, P: np.ndarray) -> np.ndarray:
    """For each objective k, min_a E_y P_k(y beats a)."""
    values = np.einsum("y,kya->ka", pi, P)
    return values.min(axis=1)


def avg_ref_win_rate(pi: np.ndarray, P: np.ndarray, ref: np.ndarray | None = None) -> float:
    """Average win rate against a fixed reference opponent distribution."""
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
    """Fallback approximation for max_pi min_{k,a} E_pi[P_k(y,a)]."""
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
    """
    try:
        from scipy.optimize import linprog

        n_objectives, n_actions, _ = P.shape
        n_vars = n_actions + 1
        v_idx = n_actions

        c = np.zeros(n_vars)
        c[v_idx] = -1.0

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
        return pi_star, float(res.x[v_idx])
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
    kl_to_ref: List[float]


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
    kls = [kl(pi, ref) for pi in policies]
    return RunHistory(name=name, policies=policies, worst_case=worst, robust_gap=gap,
                      avg_ref=avg, entropy=ent, kl_to_ref=kls)


def final_row(name: str, hist: RunHistory, P: np.ndarray, ref: np.ndarray,
              robust_opt_value: float, decoy_index: int | None = None) -> Dict[str, float | str]:
    pi = hist.policies[-1]
    out: Dict[str, float | str] = {
        "method": name,
        "worst_case": hist.worst_case[-1],
        "robust_gap": hist.robust_gap[-1],
        "avg_ref": hist.avg_ref[-1],
        "entropy": hist.entropy[-1],
        "kl_to_ref": hist.kl_to_ref[-1],
        "robust_opt_value": robust_opt_value,
    }
    if decoy_index is not None and 0 <= decoy_index < len(pi):
        out["decoy_mass"] = float(pi[decoy_index])
    return out


# -----------------------------------------------------------------------------
# Algorithms
# -----------------------------------------------------------------------------


def run_sppo(
    P: np.ndarray,
    n_iter: int,
    alpha: float,
    ref: np.ndarray,
    objective_weights: np.ndarray | None = None,
) -> List[np.ndarray]:
    """Idealized SPPO-style update with an averaged oracle."""
    n_objectives, _, _ = P.shape
    if objective_weights is None:
        objective_weights = np.ones(n_objectives) / n_objectives
    P_bar = np.einsum("k,kij->ij", objective_weights, P)
    pi = ref.copy()
    policies = [pi.copy()]
    for _ in range(n_iter):
        r = P_bar @ pi
        pi = softmax(np.log(np.clip(pi, EPS, 1.0)) + alpha * (r - 0.5))
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
    """Idealized INPO-style update with an averaged oracle and KL anchor."""
    n_objectives, _, _ = P.shape
    if objective_weights is None:
        objective_weights = np.ones(n_objectives) / n_objectives
    P_bar = np.einsum("k,kij->ij", objective_weights, P)
    pi = ref.copy()
    policies = [pi.copy()]
    for _ in range(n_iter):
        r = P_bar @ pi
        logits = ((1.0 - alpha * tau) * np.log(np.clip(pi, EPS, 1.0))
                  + alpha * tau * np.log(np.clip(ref, EPS, 1.0))
                  + alpha * r)
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
    """Idealized TD-MNPO-style update with a scalarized oracle."""
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
        nu = normalize(sum(w * p for w, p in zip(weights, opponents)))
        log_prior = sum(w * np.log(np.clip(p, EPS, 1.0)) for w, p in zip(weights, opponents))
        r = P_bar @ nu
        pi = softmax(log_prior + alpha * r)
        policies.append(pi.copy())
    return policies


def run_ht_mnpo(
    P: np.ndarray,
    n_iter: int,
    alpha: float,
    tau: float,
    ref: np.ndarray,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Idealized heterogeneous MNPO baseline.

    There are K players.  Player k updates against the mixture of the other
    players using its own oracle P_k.  The deployed policy is the arithmetic
    mixture of the K player policies.  This is intentionally simple: it captures
    the general-sum heterogeneous-player effect that RONPO is designed to avoid.
    """
    n_objectives, n_actions, _ = P.shape
    players = np.tile(ref[None, :], (n_objectives, 1))
    deployed = [normalize(players.mean(axis=0))]
    player_snapshots = [players.copy()]

    for _ in range(n_iter):
        new_players = np.zeros_like(players)
        for k in range(n_objectives):
            opp_idx = [j for j in range(n_objectives) if j != k]
            if len(opp_idx) == 0:
                opp_mix = players[k]
                log_prior = np.log(np.clip(players[k], EPS, 1.0))
            else:
                opp_mix = normalize(players[opp_idx].mean(axis=0))
                log_prior = np.mean(np.log(np.clip(players[opp_idx], EPS, 1.0)), axis=0)
            r = P[k] @ opp_mix
            logits = ((1.0 - alpha * tau) * log_prior
                      + alpha * tau * np.log(np.clip(ref, EPS, 1.0))
                      + alpha * r)
            new_players[k] = softmax(logits)
        players = new_players
        deployed.append(normalize(players.mean(axis=0)))
        player_snapshots.append(players.copy())
    return deployed, player_snapshots


def run_ronpo(
    P: np.ndarray,
    n_iter: int,
    alpha_pi: float,
    alpha_sigma: float,
    tau: float,
    kappa: float,
    ref: np.ndarray,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Exact full-information RONPO update."""
    n_objectives, n_actions, _ = P.shape
    pi = ref.copy()
    sigma = np.ones((n_objectives, n_actions)) / (n_objectives * n_actions)
    sigma0 = sigma.copy()
    policies = [pi.copy()]
    sigmas = [sigma.copy()]

    for _ in range(n_iter):
        r = np.einsum("ka,kya->y", sigma, P)
        logits_pi = ((1.0 - alpha_pi * tau) * np.log(np.clip(pi, EPS, 1.0))
                     + alpha_pi * tau * np.log(np.clip(ref, EPS, 1.0))
                     + alpha_pi * r)
        pi = softmax(logits_pi)

        c = np.einsum("y,kya->ka", pi, P)
        logits_sigma = ((1.0 - alpha_sigma * kappa) * np.log(np.clip(sigma, EPS, 1.0))
                        + alpha_sigma * kappa * np.log(np.clip(sigma0, EPS, 1.0))
                        - alpha_sigma * c)
        sigma = np.exp(logits_sigma - np.max(logits_sigma))
        sigma = sigma / sigma.sum()
        policies.append(pi.copy())
        sigmas.append(sigma.copy())
    return policies, sigmas


def run_ronpo_fixed_sigma(
    P: np.ndarray,
    n_iter: int,
    alpha_pi: float,
    tau: float,
    ref: np.ndarray,
    sigma: np.ndarray | None = None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    RONPO policy update with the adversary frozen at sigma0.

    This ablation separates "uses multiple objectives" from "adapts adversarially
    to the weakest objective-response atom".
    """
    n_objectives, n_actions, _ = P.shape
    pi = ref.copy()
    if sigma is None:
        sigma = np.ones((n_objectives, n_actions)) / (n_objectives * n_actions)
    sigma = normalize(sigma.reshape(-1)).reshape(n_objectives, n_actions)

    policies = [pi.copy()]
    sigmas = [sigma.copy()]
    for _ in range(n_iter):
        r = np.einsum("ka,kya->y", sigma, P)
        logits_pi = ((1.0 - alpha_pi * tau) * np.log(np.clip(pi, EPS, 1.0))
                     + alpha_pi * tau * np.log(np.clip(ref, EPS, 1.0))
                     + alpha_pi * r)
        pi = softmax(logits_pi)
        policies.append(pi.copy())
        sigmas.append(sigma.copy())
    return policies, sigmas


def fit_log_policy_from_pairwise_differences(
    n_actions: int,
    pairs_i: np.ndarray,
    pairs_j: np.ndarray,
    targets: np.ndarray,
    ridge: float = 1e-4,
) -> np.ndarray:
    """
    Solve min_q sum_l ((q_i - q_j) - target_l)^2 + ridge ||q||^2.

    q is identifiable only up to an additive constant without ridge.  Softmax(q)
    is invariant to that constant.  The small ridge just selects a stable solution.
    """
    B = len(targets)
    A = np.zeros((B, n_actions), dtype=np.float64)
    A[np.arange(B), pairs_i] = 1.0
    A[np.arange(B), pairs_j] = -1.0
    lhs = A.T @ A + ridge * np.eye(n_actions)
    rhs = A.T @ targets
    q = np.linalg.solve(lhs, rhs)
    return q


def run_ronpo_pairwise_stochastic(
    P: np.ndarray,
    n_iter: int,
    batch_size: int,
    alpha_pi: float,
    alpha_sigma: float,
    tau: float,
    kappa: float,
    ref: np.ndarray,
    seed: int = 0,
    ridge: float = 1e-4,
    exact_sigma_update: bool = True,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Stochastic partition-free RONPO policy update.

    For each sampled pair y,y'~pi_t and adversarial atom (k,a)~sigma_t, query
    two binary labels z_y~Ber(P_k(y,a)) and z_y'~Ber(P_k(y',a)).  Fit q=log pi
    from pairwise equations:

        q_y - q_y' ~= (1-alpha*tau)(log pi_t(y)-log pi_t(y'))
                     + alpha*tau(log ref(y)-log ref(y'))
                     + alpha(z_y-z_y').

    This is the tabular least-squares analogue of the practical RONPO loss.
    """
    rng = np.random.default_rng(seed)
    n_objectives, n_actions, _ = P.shape
    pi = ref.copy()
    sigma = np.ones((n_objectives, n_actions)) / (n_objectives * n_actions)
    sigma0 = sigma.copy()
    policies = [pi.copy()]
    sigmas = [sigma.copy()]

    for _ in range(n_iter):
        y = rng.choice(n_actions, size=batch_size, p=pi)
        yp = rng.choice(n_actions, size=batch_size, p=pi)
        flat_atoms = rng.choice(n_objectives * n_actions, size=batch_size, p=sigma.ravel())
        ks = flat_atoms // n_actions
        actions = flat_atoms % n_actions

        prob_y = P[ks, y, actions]
        prob_yp = P[ks, yp, actions]
        z_y = (rng.random(batch_size) < prob_y).astype(np.float64)
        z_yp = (rng.random(batch_size) < prob_yp).astype(np.float64)

        log_pi = np.log(np.clip(pi, EPS, 1.0))
        log_ref = np.log(np.clip(ref, EPS, 1.0))
        targets = ((1.0 - alpha_pi * tau) * (log_pi[y] - log_pi[yp])
                   + alpha_pi * tau * (log_ref[y] - log_ref[yp])
                   + alpha_pi * (z_y - z_yp))
        q = fit_log_policy_from_pairwise_differences(n_actions, y, yp, targets, ridge=ridge)
        pi = softmax(q)

        if exact_sigma_update:
            c = np.einsum("y,kya->ka", pi, P)
        else:
            y_sigma = rng.choice(n_actions, size=batch_size, p=pi)
            # Average over sampled y for every atom (k,a).
            c = P[:, y_sigma, :].mean(axis=1)
        logits_sigma = ((1.0 - alpha_sigma * kappa) * np.log(np.clip(sigma, EPS, 1.0))
                        + alpha_sigma * kappa * np.log(np.clip(sigma0, EPS, 1.0))
                        - alpha_sigma * c)
        sigma = np.exp(logits_sigma - np.max(logits_sigma))
        sigma = sigma / sigma.sum()
        policies.append(pi.copy())
        sigmas.append(sigma.copy())
    return policies, sigmas


# -----------------------------------------------------------------------------
# Plotting and output
# -----------------------------------------------------------------------------


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_metric_curves(histories: Dict[str, RunHistory], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    for metric, ylabel, title, fname in [
        ("robust_gap", "Gap to LP optimum", "Gap to Robust LP Optimum", "gap_curves.png"),
        ("worst_case", "Minimum win rate", "Worst Objective-Action Win Rate", "worst_case_curves.png"),
        ("avg_ref", "Average win rate against uniform reference", "Average Reference Win Rate", "avg_ref_curves.png"),
        ("entropy", "Policy entropy", "Policy Entropy", "entropy_curves.png"),
        ("kl_to_ref", "KL(policy || reference)", "KL to Reference", "kl_curves.png"),
    ]:
        plt.figure(figsize=(8, 5))
        for name, hist in histories.items():
            plt.plot(getattr(hist, metric), label=name)
        plt.xlabel("Iteration")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / fname, dpi=200)
        plt.close()


def plot_final_policies(histories: Dict[str, RunHistory], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    names = list(histories.keys())
    n_actions = len(next(iter(histories.values())).policies[-1])
    x = np.arange(n_actions)
    width = 0.8 / max(len(names), 1)

    plt.figure(figsize=(10, 5))
    for idx, name in enumerate(names):
        pi = histories[name].policies[-1]
        plt.bar(x + idx * width, pi, width=width, label=name)
    plt.xticks(x + width * (len(names) - 1) / 2, [f"a{i}" for i in range(n_actions)])
    plt.xlabel("Action")
    plt.ylabel("Final policy probability")
    plt.title("Final Policy Distributions")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outdir / "final_policies.png", dpi=200)
    plt.close()


def plot_ronpo_adversary(sigmas: List[np.ndarray], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    final_sigma = sigmas[-1]
    objective_mass = final_sigma.sum(axis=1)

    plt.figure(figsize=(7, 4))
    plt.bar(np.arange(len(objective_mass)), objective_mass)
    plt.xlabel("Objective index k")
    plt.ylabel("Final adversary mass")
    plt.title("RONPO Final Adversary Mass over Objectives")
    plt.tight_layout()
    plt.savefig(outdir / "ronpo_final_adversary_objective_mass.png", dpi=200)
    plt.close()


def summarize(histories: Dict[str, RunHistory], P: np.ndarray, robust_opt_value: float,
              ref: np.ndarray, robust_pi_star: np.ndarray, decoy_index: int | None) -> List[Dict[str, object]]:
    print("\n=== Robust LP optimum ===")
    print(f"value: {robust_opt_value:.6f}")
    print(f"policy: {np.round(robust_pi_star, 4)}")
    rows: List[Dict[str, object]] = []
    print("\n=== Final metrics ===")
    for name, hist in histories.items():
        pi = hist.policies[-1]
        per_obj_ref = per_objective_ref_win_rates(pi, P, ref=ref)
        per_obj_min = per_objective_worst_values(pi, P)
        row = final_row(name, hist, P, ref, robust_opt_value, decoy_index)
        rows.append(row)
        print(f"\n{name}")
        print(f"  final policy: {np.round(pi, 4)}")
        print(f"  worst-case:   {hist.worst_case[-1]:.6f}")
        print(f"  gap:          {hist.robust_gap[-1]:.6f}")
        print(f"  avg-ref:      {hist.avg_ref[-1]:.6f}")
        print(f"  entropy:      {hist.entropy[-1]:.6f}")
        print(f"  KL to ref:    {hist.kl_to_ref[-1]:.6f}")
        if decoy_index is not None:
            print(f"  decoy mass:   {pi[decoy_index]:.6f}")
        print(f"  per-objective avg-ref: {np.round(per_obj_ref, 4)}")
        print(f"  per-objective min:     {np.round(per_obj_min, 4)}")
    return rows


# -----------------------------------------------------------------------------
# Experiment runners
# -----------------------------------------------------------------------------


def run_all_methods_on_game(P: np.ndarray, args: argparse.Namespace, ref: np.ndarray) -> Tuple[Dict[str, RunHistory], List[np.ndarray]]:
    robust_pi_star, robust_opt_value = solve_robust_optimum(P)

    sppo = run_sppo(P, args.n_iter, args.alpha_sppo, ref)
    inpo = run_inpo(P, args.n_iter, args.alpha_inpo, args.tau, ref)
    mnpo = run_mnpo(P, args.n_iter, args.alpha_mnpo, ref, history_len=args.history_len)
    ht_mnpo, _ = run_ht_mnpo(P, args.n_iter, args.alpha_mnpo, args.tau, ref)
    ronpo, sigmas = run_ronpo(P, args.n_iter, args.alpha_ronpo_pi, args.alpha_ronpo_sigma,
                              args.tau, args.kappa, ref)

    histories = {
        "SPPO-avg": record_metrics("SPPO-avg", sppo, P, robust_opt_value, ref),
        "INPO-avg": record_metrics("INPO-avg", inpo, P, robust_opt_value, ref),
        "MNPO-hist-avg": record_metrics("MNPO-hist-avg", mnpo, P, robust_opt_value, ref),
        "HT-MNPO-mix": record_metrics("HT-MNPO-mix", ht_mnpo, P, robust_opt_value, ref),
        "RONPO": record_metrics("RONPO", ronpo, P, robust_opt_value, ref),
    }
    return histories, sigmas


def experiment_single(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir) / "single"
    P = make_conflicting_preferences(
        n_actions=args.n_actions,
        n_objectives=args.n_objectives,
        seed=args.seed,
        specialist_strength=args.specialist_strength,
        decoy_good=args.decoy_good,
        decoy_bad=args.decoy_bad,
        background_strength=args.background_strength,
    )
    ref = np.ones(args.n_actions) / args.n_actions
    robust_pi_star, robust_opt_value = solve_robust_optimum(P)
    histories, sigmas = run_all_methods_on_game(P, args, ref)
    decoy_index = args.n_objectives
    rows = summarize(histories, P, robust_opt_value, ref, robust_pi_star, decoy_index)
    write_csv(outdir / "summary.csv", rows)
    plot_metric_curves(histories, outdir)
    plot_final_policies(histories, outdir)
    plot_ronpo_adversary(sigmas, outdir)
    print(f"\nSaved single-run outputs to {outdir.resolve()}")


def aggregate_mean_se(rows: List[Dict[str, object]], group_keys: List[str], value_keys: List[str]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[object, ...], List[Dict[str, object]]] = {}
    for r in rows:
        key = tuple(r[k] for k in group_keys)
        groups.setdefault(key, []).append(r)
    out = []
    for key, rs in groups.items():
        row: Dict[str, object] = {k: v for k, v in zip(group_keys, key)}
        n = len(rs)
        row["n"] = n
        for vk in value_keys:
            vals = np.array([float(r[vk]) for r in rs], dtype=np.float64)
            row[f"{vk}_mean"] = float(vals.mean())
            row[f"{vk}_se"] = float(vals.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
        out.append(row)
    return out


def experiment_decoy_sweep(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir) / "decoy_sweep"
    outdir.mkdir(parents=True, exist_ok=True)
    values = [float(x) for x in args.decoy_bad_values.split(",")]
    rows: List[Dict[str, object]] = []
    for b in values:
        for seed in range(args.num_seeds):
            P = make_conflicting_preferences(
                n_actions=args.n_actions,
                n_objectives=args.n_objectives,
                seed=args.seed + seed,
                specialist_strength=args.specialist_strength,
                decoy_good=args.decoy_good,
                decoy_bad=b,
                background_strength=args.background_strength,
            )
            ref = np.ones(args.n_actions) / args.n_actions
            robust_pi_star, robust_opt_value = solve_robust_optimum(P)
            histories, _ = run_all_methods_on_game(P, args, ref)
            decoy_index = args.n_objectives
            for name, hist in histories.items():
                pi = hist.policies[-1]
                rows.append({
                    "decoy_bad": b,
                    "seed": seed,
                    "method": name,
                    "worst_case": hist.worst_case[-1],
                    "robust_gap": hist.robust_gap[-1],
                    "avg_ref": hist.avg_ref[-1],
                    "decoy_mass": float(pi[decoy_index]),
                    "entropy": hist.entropy[-1],
                    "kl_to_ref": hist.kl_to_ref[-1],
                    "robust_opt_value": robust_opt_value,
                })
    write_csv(outdir / "decoy_sweep_raw.csv", rows)
    agg = aggregate_mean_se(rows, ["decoy_bad", "method"],
                            ["worst_case", "robust_gap", "avg_ref", "decoy_mass", "entropy", "kl_to_ref"])
    write_csv(outdir / "decoy_sweep_agg.csv", agg)

    methods = sorted(set(r["method"] for r in rows))
    for metric, ylabel, fname in [
        ("worst_case", "Minimum win rate", "decoy_sweep_worst_case.png"),
        ("robust_gap", "Gap to LP optimum", "decoy_sweep_gap.png"),
        ("avg_ref", "Average reference win rate", "decoy_sweep_avg_ref.png"),
        ("decoy_mass", "Final decoy mass", "decoy_sweep_decoy_mass.png"),
    ]:
        plt.figure(figsize=(8, 5))
        for method in methods:
            xs, ys, es = [], [], []
            for b in values:
                found = [a for a in agg if a["method"] == method and float(a["decoy_bad"]) == b]
                if found:
                    a = found[0]
                    xs.append(b)
                    ys.append(float(a[f"{metric}_mean"]))
                    es.append(float(a[f"{metric}_se"]))
            plt.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=method)
        plt.xlabel("Decoy badness under hidden objective (lower = more dangerous decoy)")
        plt.ylabel(ylabel)
        plt.title(ylabel + " vs Decoy Badness")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(outdir / fname, dpi=200)
        plt.close()
    print(f"Saved decoy sweep outputs to {outdir.resolve()}")


def experiment_kappa_sweep(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir) / "kappa_sweep"
    outdir.mkdir(parents=True, exist_ok=True)
    kappas = [float(x) for x in args.kappa_values.split(",")]
    rows: List[Dict[str, object]] = []

    for seed in range(args.num_seeds):
        P = make_conflicting_preferences(
            n_actions=args.n_actions,
            n_objectives=args.n_objectives,
            seed=args.seed + seed,
            specialist_strength=args.specialist_strength,
            decoy_good=args.decoy_good,
            decoy_bad=args.decoy_bad,
            background_strength=args.background_strength,
        )
        ref = np.ones(args.n_actions) / args.n_actions
        robust_pi_star, robust_opt_value = solve_robust_optimum(P)
        decoy_index = args.n_objectives

        # Fixed baselines, recorded once per seed with method labels.
        base_histories, _ = run_all_methods_on_game(P, args, ref)
        for name in ["SPPO-avg", "INPO-avg", "MNPO-hist-avg", "HT-MNPO-mix"]:
            hist = base_histories[name]
            pi = hist.policies[-1]
            rows.append({
                "kappa": np.nan,
                "seed": seed,
                "method": name,
                "worst_case": hist.worst_case[-1],
                "robust_gap": hist.robust_gap[-1],
                "avg_ref": hist.avg_ref[-1],
                "decoy_mass": float(pi[decoy_index]),
                "entropy": hist.entropy[-1],
                "kl_to_ref": hist.kl_to_ref[-1],
                "adversary_entropy": np.nan,
                "robust_opt_value": robust_opt_value,
            })

        for kappa in kappas:
            policies, sigmas = run_ronpo(P, args.n_iter, args.alpha_ronpo_pi, args.alpha_ronpo_sigma,
                                         args.tau, kappa, ref)
            hist = record_metrics(f"RONPO-kappa={kappa:g}", policies, P, robust_opt_value, ref)
            pi = policies[-1]
            rows.append({
                "kappa": kappa,
                "seed": seed,
                "method": "RONPO",
                "worst_case": hist.worst_case[-1],
                "robust_gap": hist.robust_gap[-1],
                "avg_ref": hist.avg_ref[-1],
                "decoy_mass": float(pi[decoy_index]),
                "entropy": hist.entropy[-1],
                "kl_to_ref": hist.kl_to_ref[-1],
                "adversary_entropy": entropy(sigmas[-1].ravel()),
                "robust_opt_value": robust_opt_value,
            })
    write_csv(outdir / "kappa_sweep_raw.csv", rows)

    ronpo_rows = [r for r in rows if r["method"] == "RONPO"]
    agg = aggregate_mean_se(ronpo_rows, ["kappa", "method"],
                            ["worst_case", "robust_gap", "avg_ref", "decoy_mass", "entropy", "kl_to_ref", "adversary_entropy"])
    write_csv(outdir / "kappa_sweep_ronpo_agg.csv", agg)

    # kappa curves for RONPO.
    for metric, ylabel, fname in [
        ("worst_case", "Minimum win rate", "kappa_sweep_worst_case.png"),
        ("avg_ref", "Average reference win rate", "kappa_sweep_avg_ref.png"),
        ("robust_gap", "Gap to LP optimum", "kappa_sweep_gap.png"),
        ("adversary_entropy", "Adversary entropy", "kappa_sweep_adv_entropy.png"),
    ]:
        xs, ys, es = [], [], []
        for kappa in kappas:
            found = [a for a in agg if float(a["kappa"]) == kappa]
            if found:
                a = found[0]
                xs.append(kappa)
                ys.append(float(a[f"{metric}_mean"]))
                es.append(float(a[f"{metric}_se"]))
        plt.figure(figsize=(8, 5))
        plt.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label="RONPO")
        plt.xscale("log")
        plt.xlabel("kappa: adversary KL regularization")
        plt.ylabel(ylabel)
        plt.title(ylabel + " vs kappa")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / fname, dpi=200)
        plt.close()

    # Trade-off scatter: average vs minimum for RONPO kappa values and baseline means.
    plt.figure(figsize=(7, 6))
    for kappa in kappas:
        vals = [r for r in ronpo_rows if float(r["kappa"]) == kappa]
        x = np.mean([float(r["avg_ref"]) for r in vals])
        y = np.mean([float(r["worst_case"]) for r in vals])
        plt.scatter([x], [y])
        plt.text(x, y, f"κ={kappa:g}", fontsize=8)
    for method in ["SPPO-avg", "INPO-avg", "MNPO-hist-avg", "HT-MNPO-mix"]:
        vals = [r for r in rows if r["method"] == method]
        x = np.mean([float(r["avg_ref"]) for r in vals])
        y = np.mean([float(r["worst_case"]) for r in vals])
        plt.scatter([x], [y], marker="x")
        plt.text(x, y, method, fontsize=8)
    plt.xlabel("Average reference win rate")
    plt.ylabel("Minimum win rate")
    plt.title("Average-vs-Minimum Trade-off")
    plt.tight_layout()
    plt.savefig(outdir / "kappa_tradeoff_avg_vs_min.png", dpi=200)
    plt.close()
    print(f"Saved kappa sweep outputs to {outdir.resolve()}")


def experiment_adversary_ablation(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir) / "adversary_ablation"
    outdir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []

    for seed in range(args.num_seeds):
        P = make_conflicting_preferences(
            n_actions=args.n_actions,
            n_objectives=args.n_objectives,
            seed=args.seed + seed,
            specialist_strength=args.specialist_strength,
            decoy_good=args.decoy_good,
            decoy_bad=args.decoy_bad,
            background_strength=args.background_strength,
        )
        ref = np.ones(args.n_actions) / args.n_actions
        _, robust_opt_value = solve_robust_optimum(P)
        decoy_index = args.n_objectives

        exact_policies, exact_sigmas = run_ronpo(
            P, args.n_iter, args.alpha_ronpo_pi, args.alpha_ronpo_sigma,
            args.tau, args.kappa, ref,
        )
        fixed_policies, fixed_sigmas = run_ronpo_fixed_sigma(
            P, args.n_iter, args.alpha_ronpo_pi, args.tau, ref,
        )

        runs = [
            ("RONPO", exact_policies, exact_sigmas),
            ("RONPO-fixed-sigma", fixed_policies, fixed_sigmas),
        ]
        for method, policies, sigmas in runs:
            hist = record_metrics(method, policies, P, robust_opt_value, ref)
            pi = policies[-1]
            rows.append({
                "seed": seed,
                "method": method,
                "worst_case": hist.worst_case[-1],
                "robust_gap": hist.robust_gap[-1],
                "avg_ref": hist.avg_ref[-1],
                "decoy_mass": float(pi[decoy_index]),
                "entropy": hist.entropy[-1],
                "kl_to_ref": hist.kl_to_ref[-1],
                "adversary_entropy": entropy(sigmas[-1].ravel()),
                "objective2_adversary_mass": float(sigmas[-1][2].sum()) if sigmas[-1].shape[0] > 2 else np.nan,
                "robust_opt_value": robust_opt_value,
            })

    write_csv(outdir / "adversary_ablation_raw.csv", rows)
    agg = aggregate_mean_se(rows, ["method"],
                            ["worst_case", "robust_gap", "avg_ref", "decoy_mass",
                             "entropy", "kl_to_ref", "adversary_entropy", "objective2_adversary_mass"])
    write_csv(outdir / "adversary_ablation_agg.csv", agg)

    methods = ["RONPO-fixed-sigma", "RONPO"]
    for metric, ylabel, fname in [
        ("worst_case", "Minimum win rate", "adversary_ablation_worst_case.png"),
        ("robust_gap", "Gap to LP optimum", "adversary_ablation_gap.png"),
        ("avg_ref", "Average reference win rate", "adversary_ablation_avg_ref.png"),
        ("decoy_mass", "Final decoy mass", "adversary_ablation_decoy_mass.png"),
        ("objective2_adversary_mass", "Objective-2 adversary mass", "adversary_ablation_objective2_mass.png"),
    ]:
        xs = np.arange(len(methods))
        ys, es = [], []
        for method in methods:
            found = [a for a in agg if a["method"] == method]
            if found:
                ys.append(float(found[0][f"{metric}_mean"]))
                es.append(float(found[0][f"{metric}_se"]))
            else:
                ys.append(np.nan)
                es.append(0.0)
        plt.figure(figsize=(7, 5))
        plt.bar(xs, ys, yerr=es, capsize=4)
        plt.xticks(xs, methods, rotation=15, ha="right")
        plt.ylabel(ylabel)
        plt.title(ylabel + ": fixed vs adaptive adversary")
        plt.tight_layout()
        plt.savefig(outdir / fname, dpi=200)
        plt.close()
    print(f"Saved adversary ablation outputs to {outdir.resolve()}")


def experiment_stochastic_batch(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir) / "stochastic_batch"
    outdir.mkdir(parents=True, exist_ok=True)
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    rows: List[Dict[str, object]] = []

    for seed in range(args.num_seeds):
        P = make_conflicting_preferences(
            n_actions=args.n_actions,
            n_objectives=args.n_objectives,
            seed=args.seed + seed,
            specialist_strength=args.specialist_strength,
            decoy_good=args.decoy_good,
            decoy_bad=args.decoy_bad,
            background_strength=args.background_strength,
        )
        ref = np.ones(args.n_actions) / args.n_actions
        _, robust_opt_value = solve_robust_optimum(P)
        decoy_index = args.n_objectives

        exact_policies, _ = run_ronpo(P, args.n_iter, args.alpha_ronpo_pi, args.alpha_ronpo_sigma,
                                      args.tau, args.kappa, ref)
        exact_hist = record_metrics("RONPO-exact", exact_policies, P, robust_opt_value, ref)
        rows.append({
            "batch_size": "exact",
            "seed": seed,
            "method": "RONPO-exact",
            "worst_case": exact_hist.worst_case[-1],
            "robust_gap": exact_hist.robust_gap[-1],
            "avg_ref": exact_hist.avg_ref[-1],
            "decoy_mass": float(exact_policies[-1][decoy_index]),
            "entropy": exact_hist.entropy[-1],
            "kl_to_ref": exact_hist.kl_to_ref[-1],
        })

        for B in batch_sizes:
            policies, sigmas = run_ronpo_pairwise_stochastic(
                P, args.n_iter, B, args.alpha_ronpo_pi, args.alpha_ronpo_sigma,
                args.tau, args.kappa, ref, seed=args.seed + 1000 * seed + B,
                ridge=args.ls_ridge,
                exact_sigma_update=not args.stochastic_sigma,
            )
            hist = record_metrics(f"RONPO-stoch-B={B}", policies, P, robust_opt_value, ref)
            rows.append({
                "batch_size": B,
                "seed": seed,
                "method": "RONPO-stochastic",
                "worst_case": hist.worst_case[-1],
                "robust_gap": hist.robust_gap[-1],
                "avg_ref": hist.avg_ref[-1],
                "decoy_mass": float(policies[-1][decoy_index]),
                "entropy": hist.entropy[-1],
                "kl_to_ref": hist.kl_to_ref[-1],
            })
    write_csv(outdir / "stochastic_batch_raw.csv", rows)

    stoch_rows = [r for r in rows if r["method"] == "RONPO-stochastic"]
    agg = aggregate_mean_se(stoch_rows, ["batch_size", "method"],
                            ["worst_case", "robust_gap", "avg_ref", "decoy_mass", "entropy", "kl_to_ref"])
    write_csv(outdir / "stochastic_batch_agg.csv", agg)

    exact_vals = [r for r in rows if r["method"] == "RONPO-exact"]
    exact_mean = {metric: np.mean([float(r[metric]) for r in exact_vals])
                  for metric in ["worst_case", "robust_gap", "avg_ref", "decoy_mass"]}

    for metric, ylabel, fname in [
        ("worst_case", "Minimum win rate", "stochastic_batch_worst_case.png"),
        ("robust_gap", "Gap to LP optimum", "stochastic_batch_gap.png"),
        ("avg_ref", "Average reference win rate", "stochastic_batch_avg_ref.png"),
        ("decoy_mass", "Final decoy mass", "stochastic_batch_decoy_mass.png"),
    ]:
        xs, ys, es = [], [], []
        for B in batch_sizes:
            found = [a for a in agg if int(a["batch_size"]) == B]
            if found:
                a = found[0]
                xs.append(B)
                ys.append(float(a[f"{metric}_mean"]))
                es.append(float(a[f"{metric}_se"]))
        plt.figure(figsize=(8, 5))
        plt.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label="stochastic two-query policy update")
        plt.axhline(exact_mean[metric], linestyle="--", label="exact RONPO")
        plt.xscale("log")
        plt.xlabel("Pairwise samples per iteration")
        plt.ylabel(ylabel)
        plt.title(ylabel + " vs stochastic batch size")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(outdir / fname, dpi=200)
        plt.close()
    print(f"Saved stochastic batch outputs to {outdir.resolve()}")


def experiment_random_tournament(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir) / "random_tournament"
    outdir.mkdir(parents=True, exist_ok=True)
    rhos = [float(x) for x in args.rho_values.split(",")]
    rows: List[Dict[str, object]] = []
    for rho in rhos:
        for seed in range(args.num_seeds):
            P = make_random_antisymmetric_preferences(
                n_actions=args.random_n_actions,
                n_objectives=args.random_n_objectives,
                seed=args.seed + 10000 * seed + int(1000 * rho),
                beta=args.random_beta,
                rho=rho,
            )
            ref = np.ones(args.random_n_actions) / args.random_n_actions
            _, robust_opt_value = solve_robust_optimum(P)
            # Reuse same args but n_actions differs only for ref/P shapes.
            histories, _ = run_all_methods_on_game(P, args, ref)
            for name, hist in histories.items():
                rows.append({
                    "rho": rho,
                    "seed": seed,
                    "method": name,
                    "worst_case": hist.worst_case[-1],
                    "robust_gap": hist.robust_gap[-1],
                    "avg_ref": hist.avg_ref[-1],
                    "entropy": hist.entropy[-1],
                    "kl_to_ref": hist.kl_to_ref[-1],
                    "robust_opt_value": robust_opt_value,
                })
    write_csv(outdir / "random_tournament_raw.csv", rows)
    agg = aggregate_mean_se(rows, ["rho", "method"],
                            ["worst_case", "robust_gap", "avg_ref", "entropy", "kl_to_ref"])
    write_csv(outdir / "random_tournament_agg.csv", agg)

    methods = sorted(set(r["method"] for r in rows))
    for metric, ylabel, fname in [
        ("worst_case", "Minimum win rate", "random_tournament_worst_case.png"),
        ("robust_gap", "Gap to LP optimum", "random_tournament_gap.png"),
        ("avg_ref", "Average reference win rate", "random_tournament_avg_ref.png"),
    ]:
        plt.figure(figsize=(8, 5))
        for method in methods:
            xs, ys, es = [], [], []
            for rho in rhos:
                found = [a for a in agg if a["method"] == method and float(a["rho"]) == rho]
                if found:
                    a = found[0]
                    xs.append(rho)
                    ys.append(float(a[f"{metric}_mean"]))
                    es.append(float(a[f"{metric}_se"]))
            plt.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=method)
        plt.xlabel("Objective correlation rho (1 = homogeneous, 0 = highly heterogeneous)")
        plt.ylabel(ylabel)
        plt.title(ylabel + " under random heterogeneous tournaments")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(outdir / fname, dpi=200)
        plt.close()
    print(f"Saved random tournament outputs to {outdir.resolve()}")


def experiment_all_priority(args: argparse.Namespace) -> None:
    experiment_single(args)
    experiment_decoy_sweep(args)
    experiment_kappa_sweep(args)
    experiment_adversary_ablation(args)
    experiment_stochastic_batch(args)
    experiment_random_tournament(args)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Priority toy experiments for RONPO.")
    parser.add_argument("--experiment", type=str, default="single",
                        choices=["single", "decoy_sweep", "kappa_sweep", "stochastic_batch",
                                 "adversary_ablation", "random_tournament", "all_priority"])

    # Base decoy game.
    parser.add_argument("--n_actions", type=int, default=8)
    parser.add_argument("--n_objectives", type=int, default=3)
    parser.add_argument("--n_iter", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_seeds", type=int, default=10)
    parser.add_argument("--specialist_strength", type=float, default=0.82)
    parser.add_argument("--decoy_good", type=float, default=0.78)
    parser.add_argument("--decoy_bad", type=float, default=0.06)
    parser.add_argument("--background_strength", type=float, default=0.56)

    # Algorithm hyperparameters.
    parser.add_argument("--alpha_sppo", type=float, default=1.2)
    parser.add_argument("--alpha_inpo", type=float, default=1.2)
    parser.add_argument("--alpha_mnpo", type=float, default=1.2)
    parser.add_argument("--alpha_ronpo_pi", type=float, default=1.2)
    parser.add_argument("--alpha_ronpo_sigma", type=float, default=1.5)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--kappa", type=float, default=0.05)
    parser.add_argument("--history_len", type=int, default=4)

    # Sweeps.
    parser.add_argument("--decoy_bad_values", type=str,
                        default="0.02,0.04,0.06,0.10,0.20,0.30,0.40")
    parser.add_argument("--kappa_values", type=str,
                        default="0.005,0.01,0.02,0.05,0.1,0.2,0.5,1.0")

    # Stochastic partition-free update.
    parser.add_argument("--batch_sizes", type=str, default="8,16,32,64,128,256,512")
    parser.add_argument("--ls_ridge", type=float, default=1e-4)
    parser.add_argument("--stochastic_sigma", action="store_true",
                        help="Also estimate the adversary update c(k,a) from samples. Default uses exact c to isolate policy-loss noise.")

    # Random tournaments.
    parser.add_argument("--rho_values", type=str, default="0.0,0.25,0.5,0.75,1.0")
    parser.add_argument("--random_n_actions", type=int, default=12)
    parser.add_argument("--random_n_objectives", type=int, default=5)
    parser.add_argument("--random_beta", type=float, default=3.0)

    parser.add_argument("--outdir", type=str, default="toy_ronpo_priority_outputs")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.alpha_ronpo_pi * args.tau >= 1.0:
        print("Warning: alpha_ronpo_pi * tau >= 1.  Mirror-descent mixing exponent is non-positive.")
    if args.alpha_ronpo_sigma * args.kappa >= 1.0:
        print("Warning: alpha_ronpo_sigma * kappa >= 1.  Adversary mixing exponent is non-positive.")

    if args.experiment == "single":
        experiment_single(args)
    elif args.experiment == "decoy_sweep":
        experiment_decoy_sweep(args)
    elif args.experiment == "kappa_sweep":
        experiment_kappa_sweep(args)
    elif args.experiment == "adversary_ablation":
        experiment_adversary_ablation(args)
    elif args.experiment == "stochastic_batch":
        experiment_stochastic_batch(args)
    elif args.experiment == "random_tournament":
        experiment_random_tournament(args)
    elif args.experiment == "all_priority":
        experiment_all_priority(args)
    else:
        raise ValueError(args.experiment)


if __name__ == "__main__":
    main()
