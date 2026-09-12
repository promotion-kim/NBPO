"""Does MOPO's constraint mechanism engage at our horizon, or stay vacuous?

With t0 = infinity the bound b stays anchored at the reference policy's own
value, so the constraints bind only if raising the primary objective pushes a
secondary BELOW the untrained reference. The dual said lambda = 0, i.e. it does
not. The paper's mechanism is a ratchet: every t0 steps b is raised to beta
times what the policy itself just achieved, so the constraints tighten as the
policy improves.

This simulates the ratchet with array arithmetic only -- no policy forward pass,
because b = beta * G(rho) depends on the teacher values and rho, not on the
network -- and reports whether lambda leaves zero and what it costs the primary
objective. If lambda stays zero through every stage, MOPO reduces to
single-objective preference optimisation on this panel and that is the result to
report rather than a number to tune until it looks different.
"""
import json
import sys

import numpy as np

sys.path.insert(0, "/work/uf4_20260910/code")
from solve_mopo_targets import (candidate_values, best_lower_bound, rho_of_lambda,
                                calibrate_epsilon, solve_dual)

A = np.load("/work/uf4_20260910/targets/nash_v1/dev/tensor/tensor_policy.npz")["A"]
W = candidate_values(A)
pbar, qbar = W[0], np.stack([W[1], W[2], W[3]])
grid = np.array([1e-6, 1e6])
tau, beta, stages = 0.08, 0.9995, 6
EPS_MODE = __import__("os").environ.get("EPS_MODE", "calibrated")

eps, cal = (0.0, {}) if EPS_MODE == "zero" else calibrate_epsilon(qbar, beta, grid)
G_ref = qbar.mean(axis=(1, 2))
print(json.dumps({"epsilon": eps, "G_reference": G_ref.tolist()}))

rho = np.ones_like(pbar)                       # stage 0: the reference policy itself
b = beta * G_ref
rows = []
for s in range(stages):
    lam, rho_new, lb, J, F, _ = solve_dual(pbar, qbar, b, tau, eps, iters=120,
                                           step=0.5, grid=grid)
    G_new = np.array([float(np.mean(rho_new * qbar[k])) for k in range(3)])
    rows.append({"stage": s, "b": b.round(6).tolist(), "lambda": lam.round(4).tolist(),
                 "slack": (lb - b).round(5).tolist(),
                 "primary_F": round(F, 5),
                 "G_achieved": G_new.round(6).tolist(),
                 "rho_max": round(float(rho_new.max()), 3),
                 "rho_min": round(float(rho_new.min()), 6)})
    print(json.dumps(rows[-1]), flush=True)
    rho = rho_new
    b = beta * G_new                           # the ratchet
print(json.dumps({"lambda_ever_positive": any(max(r["lambda"]) > 1e-8 for r in rows),
                  "primary_F_first_to_last": [rows[0]["primary_F"], rows[-1]["primary_F"]],
                  "b_first_to_last": [rows[0]["b"], rows[-1]["b"]]}, indent=1))
