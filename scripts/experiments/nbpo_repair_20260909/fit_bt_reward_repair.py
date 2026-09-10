"""Fit the scalar Bradley-Terry reward table the BT-RM control needs.

The BT-RM row asks what happens when each objective is first collapsed into a
scalar utility -- the standard RLHF move -- and only then bargained over. It is
fit to *exactly the comparisons the game tensors are built from*: the same
frozen three-seed ensemble, the same calibrated probabilities, averaged over
seeds the same way, with no new judgments and no model training.

Observed graph, per (objective k, prompt x)
    P_LC[i, j] = P(learner_i > comparator_j)          from scores["policy"]
    P_RC[m, j] = P(reference_learner_m > comparator_j) from scores["reference"]

Two bipartite blocks joined at the comparator pool, so the 24 scores are
identified up to one additive constant.  That constant is fixed by centring on
the comparator pool.  ``r_learner = s_L`` and ``r_reference = s_R``: the
disagreement point is then the BT value of an *independent* draw from the
reference policy, which is the same construction ``adaptive_game`` and
``fixed_reference`` measure their ``d_k`` with.

The per-prompt residual is written out.  It is the honest diagnostic of how much
of the pairwise structure a scalar reward cannot represent, and it is a
measurement, not a gate: nothing here selects on it.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from scripts.experiments.nbpo_repair_20260909.common import file_hash, write_json


def _nll_and_grad(theta, P_LC, P_RC, I, J, M):
    """Soft-label BT cross-entropy over [s_L, s_C, s_R] and its exact gradient."""
    sL, sC, sR = theta[:I], theta[I:I+J], theta[I+J:]
    total, grad = 0.0, np.zeros_like(theta)
    for S, P, lo, hi in ((sL, P_LC, 0, I), (sR, P_RC, I+J, I+J+M)):
        d = S[:, None] - sC[None, :]
        # log(1+exp(-d)) without overflow, then the soft-label cross-entropy
        soft = np.logaddexp(0.0, -d)
        total += float((P*soft + (1.0-P)*(soft+d)).sum())
        e = 1.0/(1.0+np.exp(-d)) - P
        grad[lo:hi] += e.sum(axis=1)
        grad[I:I+J] -= e.sum(axis=0)
    return total, grad


def fit_prompt(P_LC, P_RC):
    I, J = P_LC.shape
    M = P_RC.shape[0]
    theta = np.zeros(I + J + M)
    out = minimize(_nll_and_grad, theta, args=(P_LC, P_RC, I, J, M), jac=True,
                   method="L-BFGS-B", options={"maxiter": 500, "ftol": 1e-14, "gtol": 1e-10})
    theta = out.x.copy()
    theta -= theta[I:I+J].mean()          # centre on the comparator pool
    sL, sC, sR = theta[:I], theta[I:I+J], theta[I+J:]
    resid = 0.0
    for S, P in ((sL, P_LC), (sR, P_RC)):
        q = 1.0/(1.0+np.exp(-(S[:, None] - sC[None, :])))
        resid = max(resid, float(np.abs(q - P).max()))
    return sL, sC, sR, resid, float(np.max(np.abs(out.jac))), bool(out.success)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--out-name", required=True)
    args = ap.parse_args()
    root = args.root
    out = root / "teacher_bt" / args.out_name
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()

    scores, manifests = {}, []
    for shard in range(args.shards):
        folder = root / "scores" / f"shard{shard}"
        manifest = json.loads((folder / "manifest.json").read_text())
        if file_hash(folder / "probabilities.npz") != manifest["probabilities_sha256"]:
            raise ValueError("Score tensor hash mismatch")
        if manifest["objectives"] != ["helpfulness", "harmlessness"] or manifest["seeds"] != [41, 42, 43]:
            raise ValueError("Unexpected teacher objective order or ensemble members")
        if manifest.get("calibration_applied_before_seed_average") is not True:
            raise ValueError("Frozen calibration must precede ensemble averaging")
        arrays = np.load(folder / "probabilities.npz")
        # Average the three calibrated members once, exactly as the game tensor does.
        pol, ref = arrays["policy"].mean(0), arrays["reference"].mean(0)
        for index, pid in enumerate(manifest["prompt_ids"]):
            if pid in scores:
                raise ValueError("Duplicate scored prompt")
            scores[pid] = (pol[:, index], ref[:, index])
        manifests.append({"shard": shard, "probabilities_sha256": manifest["probabilities_sha256"],
                          "n_prompts": len(manifest["prompt_ids"])})

    splits = {split: [json.loads(line)["prompt_id"] for line in
                      (root / "splits" / f"{split}.jsonl").read_text().splitlines() if line]
              for split in ("train", "dev", "test")}
    diagnostics = {}
    for split, pids in splits.items():
        K = 2
        X = len(pids)
        r_l = np.zeros((K, X, 8)); r_c = np.zeros((K, X, 8)); r_r = np.zeros((K, X, 8))
        resid = np.zeros((K, X)); gnorm = np.zeros((K, X)); ok = np.ones((K, X), dtype=bool)
        for x, pid in enumerate(pids):
            P_LC_all, P_RC_all = scores[pid]
            for k in range(K):
                sL, sC, sR, res, g, success = fit_prompt(P_LC_all[k], P_RC_all[k])
                r_l[k, x], r_c[k, x], r_r[k, x] = sL, sC, sR
                resid[k, x], gnorm[k, x], ok[k, x] = res, g, success
        np.savez_compressed(out / f"bt_reward_table_{split}.npz",
                            r_learner=r_l, r_reference=r_r, r_comparator=r_c,
                            residual_max=resid, gradient_inf=gnorm)
        diagnostics[split] = {
            "n_prompts": X, "converged_fraction": float(ok.mean()),
            "max_gradient_inf": float(gnorm.max()),
            "residual_max_over_all": float(resid.max()),
            "residual_max_mean": float(resid.mean()),
            "residual_max_p90": float(np.quantile(resid, 0.9)),
            "prompt_ids_sha256": file_hash(root / "splits" / f"{split}.jsonl")}
        print(json.dumps({"split": split, **diagnostics[split]}), flush=True)

    write_json(out / "complete.json", {
        "score_manifests": manifests, "splits": diagnostics,
        "r_learner_role": "BT score of each of the 8 learner-pool occurrences",
        "r_reference_role": "BT score of each of the 8 independent reference-learner occurrences",
        "centering": "scores centred on the 8-response comparator pool per (objective, prompt)",
        "residual_meaning": ("max |sigmoid(s_a - s_b) - P_ab| over the observed comparisons; "
                             "a measurement of what a scalar reward cannot represent, never a gate"),
        "seconds": time.monotonic()-start, "source_sha256": file_hash(__file__)})


if __name__ == "__main__":
    main()
