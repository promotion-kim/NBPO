#!/usr/bin/env python3
"""Deterministic finite-game checks for review R06/R12 (no LLM experiment).

Run: python verify_direct_solver_counterexamples.py [output.json]
Dependencies: numpy, scipy. All numbers are recomputed rather than hard-coded.
R12 fixes eta*lambda=4 and studies the weighted inner solve, not the solved
outer Nash dual. The direct optimizer includes the adaptive opponent exactly.
"""
import json
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize, root
from scipy.special import logsumexp, softmax

A = np.array([[0., .5, -.5], [-.5, 0., .5], [.5, -.5, 0.]])
beta = .25

def value(p, anchor):
    return float(-beta * logsumexp(np.log(anchor) - A.T @ p / beta))

def covariance(p):
    return np.diag(p) - np.outer(p, p)

q6 = np.array([.8, .1, .1])
p6 = np.array([.25, .43, .32])
r06 = {
    'reference': q6.tolist(), 'policy': p6.tolist(),
    'reference_value': value(q6, q6), 'policy_value': value(p6, q6),
    'surplus': value(p6, q6) - value(q6, q6),
    'direct_margin': float(p6 @ A @ q6),
    'direct_win_probability': float(.5 + p6 @ A @ q6),
}

q = np.array([.34, .33, .33])
eta_lambda = 4.

def opponent(p):
    return softmax(np.log(q) - A.T @ p / beta)

def legacy_map(p):
    return softmax(np.log(q) + eta_lambda * A @ opponent(p))

def lift(z):
    return np.r_[z, 1. - np.sum(z)]

fixed = root(lambda z: (legacy_map(lift(z)) - lift(z))[:2], q[:2], tol=1e-12)
p_fixed = lift(fixed.x)
# d nu / dp = -(diag(nu)-nu nu^T) A^T / beta. Here -A^T=A.
nu = opponent(p_fixed)
jac = (-eta_lambda / beta * covariance(legacy_map(p_fixed))
       @ A @ covariance(nu) @ A.T)
basis = np.array([[1., 0.], [0., 1.], [-1., -1.]])
jac_tangent = jac[:2] @ basis
p_legacy = q.copy()
for _ in range(100):
    p_legacy = legacy_map(p_legacy)

def loss(p):
    return float(np.sum(p * np.log(p / q)) - eta_lambda * value(p, q))

def gradient(p):
    return np.log(p / q) + 1. - eta_lambda * A @ opponent(p)

solved = minimize(loss, q, jac=gradient, method='SLSQP',
    bounds=[(1e-12, 1.)] * 3,
    constraints=[{'type': 'eq', 'fun': lambda p: np.sum(p)-1.,
                  'jac': lambda p: np.ones_like(p)}],
    options={'ftol': 1e-14, 'maxiter': 1000})
p_direct = solved.x
g_direct = gradient(p_direct)
# Positive-p simplex KKT requires all gradient coordinates to agree.
inner_kkt = float(np.max(np.abs(g_direct - np.mean(g_direct))))
r12 = {
    'reference': q.tolist(), 'eta_times_lambda': eta_lambda,
    'fixed_point_root_success': bool(fixed.success),
    'fixed_point': p_fixed.tolist(),
    'fixed_point_residual_linf': float(np.max(np.abs(legacy_map(p_fixed)-p_fixed))),
    'tangent_jacobian': jac_tangent.tolist(),
    'tangent_eigenvalues': np.linalg.eigvals(jac_tangent).real.tolist(),
    'reference_value': value(q,q), 'fixed_point_value': value(p_fixed,q),
    'legacy_iterations': 100, 'legacy_final_policy': p_legacy.tolist(),
    'legacy_fixed_point_error_l2': float(np.linalg.norm(p_legacy-p_fixed)),
    'legacy_map_residual_l2': float(np.linalg.norm(legacy_map(p_legacy)-p_legacy)),
    'direct_solver': 'scipy.optimize.minimize(SLSQP), analytic gradient',
    'direct_success': bool(solved.success), 'direct_message': str(solved.message),
    'direct_iterations': int(solved.nit), 'direct_policy': p_direct.tolist(),
    'direct_fixed_point_error_l2': float(np.linalg.norm(p_direct-p_fixed)),
    'direct_simplex_error': float(abs(np.sum(p_direct)-1.)),
    'direct_inner_kkt_linf': inner_kkt,
    'scaled_inner_objective_reference': -loss(q),
    'scaled_inner_objective_direct': -loss(p_direct),
    'scaled_inner_objective_legacy': -loss(p_legacy),
}
result = {'kind': 'deterministic finite-game audit; not LLM evidence',
          'payoff_matrix': A.tolist(), 'beta': beta, 'R06': r06, 'R12': r12}
assert r06['surplus'] > 0 and r06['direct_win_probability'] < .5
assert fixed.success and r12['fixed_point_residual_linf'] < 1e-10
assert min(abs(v) for v in r12['tangent_eigenvalues']) > 1
assert solved.success and inner_kkt < 1e-6
assert r12['direct_fixed_point_error_l2'] < 1e-6
out = Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).with_suffix('.json')
out.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
