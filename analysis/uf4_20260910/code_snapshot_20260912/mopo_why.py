"""Why MOPO's constraints never bind here: measure it, do not assert it.

The claim to check is that on this candidate pool raising the primary objective
also raises the other three, so a constraint of the form "keep the secondaries
at beta times what you just achieved" is satisfied for free. The quantity that
decides it is the within-prompt rank association between the primary and each
secondary across the eight candidates.
"""
import json
import sys

import numpy as np

sys.path.insert(0, "/work/uf4_20260910/code")
from solve_mopo_targets import candidate_values, rho_of_lambda

out = {}
for split in ("train", "dev"):
    A = np.load("/work/uf4_20260910/targets/nash_v1/%s/tensor/tensor_policy.npz" % split)["A"]
    W = candidate_values(A)            # (4, X, 8)
    names = ["instruction_following", "truthfulness", "honesty", "helpfulness"]
    X = W.shape[1]

    def spearman_within(a, b):
        ra = np.argsort(np.argsort(a, axis=1), axis=1).astype(float)
        rb = np.argsort(np.argsort(b, axis=1), axis=1).astype(float)
        ra -= ra.mean(axis=1, keepdims=True)
        rb -= rb.mean(axis=1, keepdims=True)
        num = (ra * rb).sum(axis=1)
        den = np.sqrt((ra ** 2).sum(axis=1) * (rb ** 2).sum(axis=1))
        return num / np.maximum(den, 1e-12)

    rows = {}
    for k in range(1, 4):
        r = spearman_within(W[0], W[k])
        rows[names[k]] = {"mean_within_prompt_spearman_vs_primary": float(r.mean()),
                          "fraction_of_prompts_positive": float((r > 0).mean()),
                          "fraction_below_-0.5": float((r < -0.5).mean())}
    # The decisive test: does the argmax of the primary also clear each
    # secondary's mean?  That is exactly what the ratcheted bound asks.
    top = W[0].argmax(axis=1)
    clears = {}
    for k in range(1, 4):
        v_top = W[k][np.arange(X), top]
        clears[names[k]] = {"mean_secondary_at_primary_argmax": float(v_top.mean()),
                            "pool_mean": float(W[k].mean()),
                            "fraction_of_prompts_at_or_above_pool_mean":
                                float((v_top >= W[k].mean(axis=1)).mean())}
    # And under the actual rho at lambda = 0.
    rho = rho_of_lambda(W[0], np.stack([W[1], W[2], W[3]]), np.zeros(3), 0.08)
    g = {names[k]: float(np.mean(rho * W[k])) for k in range(1, 4)}
    g["primary"] = float(np.mean(rho * W[0]))
    ref = {names[k]: float(W[k].mean()) for k in range(1, 4)}
    ref["primary"] = float(W[0].mean())
    out[split] = {"n_prompts": int(X), "within_prompt_association": rows,
                  "primary_argmax_check": clears,
                  "value_under_rho_lambda0": g, "value_under_reference": ref}
print(json.dumps(out, indent=1))
