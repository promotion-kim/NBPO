"""Add factorized-adversary target columns to the stage-2 OS-RONPO dataset.

Same rows, pairs, and precomputed logps as build_os_ronpo_targets.py; only new
scalar target columns are appended, so a training arm on any column is a
matched-budget A/B against the existing top-mass / OS arms.

  target_konly_k{K}    : adversary restricted to the objective index k.
                         omega(k) prop exp(-costk/kappa) with
                         costk = mean_a cost(k,a)  (opponent fixed at uniform),
                         one opponent a_k ~ Uniform(A) per objective (same
                         one-sample-per-objective budget as OS), target
                         = sum_k omega(k) * zhat(k, a_k).
  target_konlyexp_k{K} : exact-expectation diagnostic of the same adversary,
                         sum_k omega(k) * mean_a zhat(k,a).
  target_aonly_k{K}    : adversary restricted to the opponent response a.
                         nu(a) prop exp(-costa/kappa) with costa = mean_k
                         cost(k,a) (objective fixed at uniform), target
                         = sum_a nu(a) * mean_k zhat(k,a).

Before writing anything the script recomputes the stored target_os_k0p05 with
the original code path and asserts exact agreement, so the cost/sigma math is
guaranteed identical to the shipped arms.
"""
import argparse
import hashlib
import numpy as np
from datasets import load_from_disk


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def row_tensors(example):
    names = example["objective_names"]
    S = np.array([example["normalized_objective_scores"][k] for k in names], dtype=float)
    scale = float(example.get("homogeneous_oracle_preference_scale", 8.0))
    ci, ri = int(example["chosen_index"]), int(example["rejected_index"])
    P = sigmoid(scale * (S[:, :, None] - S[:, None, :]))   # K x A x A
    cost = P.mean(axis=1)                                  # K x A (pi uniform)
    zhat = P[:, ci, :] - P[:, ri, :]                       # K x A
    return cost, zhat


def os_target(example, kappa):
    # verbatim reproduction of build_os_ronpo_targets.py for validation
    cost, zhat = row_tensors(example)
    K, A = cost.shape
    seed = int.from_bytes(hashlib.sha256(
        str(example.get("prompt_id") or example.get("prompt")).encode()).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    logits = -cost.reshape(-1) / kappa
    logits -= logits.max()
    sig = np.exp(logits); sig /= sig.sum()
    sigma = sig.reshape(K, A)
    omega = sigma.sum(axis=1)
    q = sigma / omega[:, None]
    out = {"target_uniform": float(zhat.mean())}
    ka = int(np.argmax(sig)); k_star, a_star = divmod(ka, A)
    out["target_topmass"] = float(zhat[k_star, a_star])
    out["target_fullexp"] = float((sigma * zhat).sum())
    os_val = 0.0
    for k in range(K):
        a_k = int(rng.choice(A, p=q[k]))
        os_val += omega[k] * zhat[k, a_k]
    out["target_os"] = float(os_val)
    return out


def factorized_targets(example, kappas):
    cost, zhat = row_tensors(example)
    K, A = cost.shape
    seed = int.from_bytes(hashlib.sha256(
        ("konly|" + str(example.get("prompt_id") or example.get("prompt"))).encode()).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    out = {}
    for kappa in kappas:
        tag = f"{kappa:g}".replace(".", "p")
        # objective-only adversary, opponent held at uniform
        costk = cost.mean(axis=1)                          # K
        w = np.exp(-(costk - costk.min()) / kappa); w /= w.sum()
        a_draw = rng.integers(0, A, size=K)
        out[f"target_konly_k{tag}"] = float(sum(w[k] * zhat[k, a_draw[k]] for k in range(K)))
        out[f"target_konlyexp_k{tag}"] = float((w * zhat.mean(axis=1)).sum())
        # response-only adversary, objective held at uniform
        costa = cost.mean(axis=0)                          # A
        nu = np.exp(-(costa - costa.min()) / kappa); nu /= nu.sum()
        out[f"target_aonly_k{tag}"] = float((nu * zhat.mean(axis=0)).sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--kappas", default="0.05")
    ap.add_argument("--num_proc", type=int, default=12)
    args = ap.parse_args()
    kappas = [float(x) for x in args.kappas.split(",")]

    ds = load_from_disk(args.input_dir)
    ref = ds["train"]
    for i in range(300):
        rep = os_target(ref[i], 0.05)
        for short, col in (("target_topmass", "target_topmass_k0p05"),
                           ("target_fullexp", "target_fullexp_k0p05"),
                           ("target_os", "target_os_k0p05")):
            stored = float(ref[i][col])
            assert abs(stored - rep[short]) < 1e-10, (i, col, stored, rep[short])
    print("[validate] reproduced stored topmass/fullexp/os k0p05 targets on 300 rows exactly")

    ds2 = ds.map(lambda ex: factorized_targets(ex, kappas), num_proc=args.num_proc,
                 desc="factorized adversary targets")
    ds2.save_to_disk(args.output_dir)
    tr = ds2["train"]
    print(f"[built] {args.output_dir} rows(train)={len(tr)}")
    for c in sorted(c for c in tr.column_names if c.startswith("target_")):
        v = np.array(tr[c], dtype=float)
        print(f"  {c:24s} mean={v.mean():+.4f} std={v.std():.4f} mean|.|={np.abs(v).mean():.4f}")


if __name__ == "__main__":
    main()
