"""Check that the all-candidate regression equals the uniform all-pair one.

Three checks, all on CPU:

1. Random tensors. For n candidates and arbitrary u, compare
   (1/C(n,2)) sum_{i<j}(u_i-u_j)^2 against (2/(n-1)) sum_i (u_i-mean u)^2 in
   value and in gradient, in float64 and float32.

2. Real solver output. Build u at theta = pi_t from the frozen NBPO target,
   u_i = -log(p*_i / p_{t,i}), on the 200 frozen training prompts, and compare
   the two forms on those numbers rather than on synthetic ones. The value at
   theta = pi_t is also the initial objective the pilot starts from.

3. The pair target column. The sampled-pair loss regresses on
   log(p*_a/p_{t,a}) - log(p*_b/p_{t,b}); verify that identity against the
   canonical target column actually stored in the training dataset, so the two
   arms are shown to share one target rather than asserted to.

Nothing here trains, and the identity is only claimed for the uniform unordered
pair measure the pilot uses.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/work/uf4_20260910")
DIAG = ROOT / "analysis/diag_20260914"
SALT = "uf4-diag-20260914|"
TARGET = "nash_v1"


def key(text):
    return hashlib.sha256((SALT + text).encode()).hexdigest()


def pair_form(u):
    n = u.shape[-1]
    idx = torch.triu_indices(n, n, offset=1)
    diffs = u[..., idx[0]] - u[..., idx[1]]
    return (diffs ** 2).mean(dim=-1)


def centered_form(u):
    n = u.shape[-1]
    return (2.0 / (n - 1)) * ((u - u.mean(dim=-1, keepdim=True)) ** 2).sum(dim=-1)


def random_check(dtype, n=8, batch=64, seed=20260914):
    g = torch.Generator().manual_seed(seed)
    u = torch.randn(batch, n, generator=g, dtype=dtype, requires_grad=True)
    a = pair_form(u).sum()
    ga, = torch.autograd.grad(a, u, retain_graph=True)
    b = centered_form(u).sum()
    gb, = torch.autograd.grad(b, u)
    return {"dtype": str(dtype), "value_abs_diff": float((a - b).abs()),
            "value_rel_diff": float((a - b).abs() / a.abs().clamp_min(1e-30)),
            "grad_max_abs_diff": float((ga - gb).abs().max()),
            "grad_rms": float(ga.pow(2).mean().sqrt())}


def main():
    DIAG.mkdir(parents=True, exist_ok=True)
    out = {"random_checks": [random_check(torch.float64), random_check(torch.float32)]}

    meta = json.loads((ROOT / "targets" / TARGET / "train/tensor/meta.json").read_text())
    ids = list(meta["prompt_ids"])
    pi_star = np.load(ROOT / "targets" / TARGET / "train/solver/pi_star.npz")["pi"]
    pi_t = np.load(ROOT / "targets" / TARGET / "train/solver/pi_t.npz")["pi"]
    order = sorted(range(len(ids)), key=lambda k: key(ids[k]))[:200]
    panel = [ids[k] for k in order]
    (DIAG / "panel_train200.json").write_text(json.dumps({
        "salt": SALT, "rule": "sha256(salt+prompt_id) ascending; first 200 of the train target set",
        "target_set": TARGET, "n_available": len(ids),
        "panel_prompt_ids": panel,
        "panel_sha256": hashlib.sha256("|".join(panel).encode()).hexdigest(),
    }, indent=2) + "\n")

    ps = torch.tensor(pi_star[order], dtype=torch.float64)
    pt = torch.tensor(pi_t[order], dtype=torch.float64)
    floor = 1e-12
    u = -(torch.log(ps.clamp_min(floor)) - torch.log(pt.clamp_min(floor)))
    u.requires_grad_(True)
    a = pair_form(u)
    b = centered_form(u)
    ga, = torch.autograd.grad(a.sum(), u, retain_graph=True)
    gb, = torch.autograd.grad(b.sum(), u)
    out["solver_check"] = {
        "target_set": TARGET,
        "train_panel_sha256": hashlib.sha256("|".join(panel).encode()).hexdigest()[:16],
        "prompts": len(panel),
        "zero_mass_candidates": int((ps < floor).sum()),
        "value_max_abs_diff": float((a - b).abs().max()),
        "value_max_rel_diff": float(((a - b).abs() / a.abs().clamp_min(1e-30)).max()),
        "grad_max_abs_diff": float((ga - gb).abs().max()),
        "initial_objective_mean": float(a.mean()),
        "initial_objective_median": float(a.median()),
        "u_rms": float(u.pow(2).mean().sqrt()),
    }

    # ---- the stored pair target against log(p*_a/p_t_a) - log(p*_b/p_t_b)
    pairs_path = ROOT / "targets" / TARGET / "pairs/train.jsonl"
    index = {pid: k for k, pid in enumerate(ids)}
    checked, worst, seen_cols = 0, 0.0, None
    if pairs_path.exists():
        want = set(panel)
        with pairs_path.open() as stream:
            for line in stream:
                row = json.loads(line)
                pid = row.get("prompt_id")
                if pid not in want:
                    continue
                if seen_cols is None:
                    seen_cols = sorted(row)
                cand_a = row.get("chosen_candidate_index")
                cand_b = row.get("rejected_candidate_index")
                stored = row.get("nbpo_logratio_target")
                if cand_a is None or cand_b is None or stored is None:
                    continue
                k = index[pid]
                expect = (np.log(max(pi_star[k, cand_a], floor)) - np.log(max(pi_t[k, cand_a], floor))
                          - np.log(max(pi_star[k, cand_b], floor)) + np.log(max(pi_t[k, cand_b], floor)))
                worst = max(worst, abs(float(stored) - float(expect)))
                checked += 1
                if checked >= 20000:
                    break
    out["pair_target_check"] = {
        "pairs_path": str(pairs_path), "pairs_checked": checked,
        "max_abs_deviation": worst,
        "identity": "stored nbpo_logratio_target == log(p*_a/p_t_a) - log(p*_b/p_t_b)",
        "note": ("The two pilot arms therefore regress on one target: the sampled-pair loss "
                 "uses this column and the all-candidate loss uses the same p* it is built "
                 "from."),
    }
    (DIAG / "loss_identity.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
