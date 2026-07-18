"""Add OS-RONPO / full-exp / top-mass target columns to an EXISTING precomputed
RONPO dataset, reusing its (chosen, rejected) pairs and precomputed logps
(no re-decode, no GPU).  For each row we recompute the entropic adversary
sigma(k,a) at a target temperature kappa analytically (its fixed point is
sigma ∝ sigma0 * exp(-cost/kappa)), then form three scalar targets that only
differ in how the objective-response atoms are aggregated:

  target_topmass_k{K}  : zhat at argmax_(k,a) sigma       (top-1 truncation)
  target_os_k{K}       : sum_k omega(k) * zhat(k, a_k),  a_k ~ q(.|k)   (OS-RONPO)
  target_fullexp_k{K}  : sum_(k,a) sigma(k,a) * zhat(k,a)               (full expectation)

with  zhat(k,a) = P_k(chosen>a) - P_k(rejected>a),
      P_k(y>a)  = sigmoid(scale*(s_k[y]-s_k[a])),   s_k = normalized_objective_scores,
      cost(k,a) = mean_y sigmoid(scale*(s_k[y]-s_k[a]))          (pi = uniform),
      omega(k)  = sum_a sigma(k,a),  q(a|k) = sigma(k,a)/omega(k).

The (chosen,rejected) pairs were sampled policy-vs-policy under pi=uniform, so
reusing them as (y,y')~pi is exactly what OS-RONPO needs.  Only the scalar
target differs across arms => same rows, same logps, matched budget.

The no-adversary control is also emitted once per row:

  target_uniform        : mean_(k,a) zhat(k,a)

It is the exact kappa-to-infinity limit of the full-expectation target when
the adversary is uniform.  It does not depend on the requested kappa list.
"""
import argparse
import hashlib
import math
import numpy as np
from datasets import load_from_disk


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def row_targets(example, kappas):
    names = example["objective_names"]
    K = len(names)
    S = np.array([example["normalized_objective_scores"][k] for k in names], dtype=float)  # K x A
    A = S.shape[1]
    scale = float(example.get("homogeneous_oracle_preference_scale", 8.0))
    ci = int(example["chosen_index"]); ri = int(example["rejected_index"])

    # pairwise preference tensor P[k,y,a] = sigmoid(scale*(S[k,y]-S[k,a]))
    diff = S[:, :, None] - S[:, None, :]            # K x A(y) x A(a)
    P = sigmoid(scale * diff)                        # K x A x A
    cost = P.mean(axis=1)                            # K x A   (pi uniform over y)
    zhat = P[:, ci, :] - P[:, ri, :]                 # K x A   = zhat(k,a)

    # deterministic per-row RNG for the OS opponent sampling
    seed = int.from_bytes(hashlib.sha256(
        str(example.get("prompt_id") or example.get("prompt")).encode()).digest()[:8], "big")
    rng = np.random.default_rng(seed)

    # The exact no-adversary endpoint.  Keeping this outside the kappa loop
    # makes its semantics explicit and avoids creating duplicate columns for
    # every temperature.
    out = {"target_uniform": float(zhat.mean())}
    for kappa in kappas:
        tag = f"{kappa:g}".replace(".", "p")
        logits = -cost.reshape(-1) / kappa
        logits -= logits.max()
        sig = np.exp(logits); sig /= sig.sum()
        sigma = sig.reshape(K, A)                     # sigma(k,a), sums to 1
        omega = sigma.sum(axis=1)                     # omega(k)
        q = sigma / omega[:, None]                    # q(a|k)
        # top-mass: single argmax atom
        ka = int(np.argmax(sig)); k_star, a_star = divmod(ka, A)
        out[f"target_topmass_k{tag}"] = float(zhat[k_star, a_star])
        # full expectation
        out[f"target_fullexp_k{tag}"] = float((sigma * zhat).sum())
        # OS: one opponent per objective, weighted by omega
        os_val = 0.0
        for k in range(K):
            a_k = int(rng.choice(A, p=q[k]))
            os_val += omega[k] * zhat[k, a_k]
        out[f"target_os_k{tag}"] = float(os_val)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--kappas", default="0.05,0.01")
    ap.add_argument("--num_proc", type=int, default=12)
    ap.add_argument("--validate_only", action="store_true")
    args = ap.parse_args()
    kappas = [float(x) for x in args.kappas.split(",")]

    ds = load_from_disk(args.input_dir)
    splits = ds.keys() if hasattr(ds, "keys") else {"train": ds}

    # ---- validation: reproduce stored ronpo_objective_gap from our zhat ----
    ref = ds["train"] if "train" in splits else ds[list(splits)[0]]
    maxerr = 0.0
    for i in range(min(200, len(ref))):
        ex = ref[i]
        names = ex["objective_names"]
        S = np.array([ex["normalized_objective_scores"][k] for k in names], dtype=float)
        scale = float(ex.get("homogeneous_oracle_preference_scale", 8.0))
        ci, ri = int(ex["chosen_index"]), int(ex["rejected_index"])
        adv = int(ex["ronpo_adversary_response_index"]); ok = int(ex["ronpo_objective_index"])
        z = float(sigmoid(scale*(S[ok, ci]-S[ok, adv])) - sigmoid(scale*(S[ok, ri]-S[ok, adv])))
        maxerr = max(maxerr, abs(z - float(ex["ronpo_objective_gap"])))
    print(f"[validate] max |zhat - stored ronpo_objective_gap| over 200 rows = {maxerr:.3e}")
    assert maxerr < 1e-6, "zhat formula does not match stored target; aborting."
    if args.validate_only:
        # show target distributions for a quick sanity read
        ex0 = row_targets(ref[0], kappas)
        print("[sample row0 targets]", {k: round(v, 4) for k, v in ex0.items()})
        return

    def add(example):
        return row_targets(example, kappas)

    ds2 = ds.map(add, num_proc=args.num_proc, desc="OS/full-exp/top-mass targets")
    ds2.save_to_disk(args.output_dir)
    # report mean |target| per arm (sanity: OS/full-exp should be smaller-variance than top-mass)
    tr = ds2["train"] if "train" in splits else ds2[list(splits)[0]]
    cols = [c for c in tr.column_names if c.startswith("target_")]
    import numpy as _np
    print(f"[built] {args.output_dir}  rows(train)={len(tr)}")
    for c in sorted(cols):
        v = _np.array(tr[c], dtype=float)
        print(f"  {c:24s} mean={v.mean():+.4f} std={v.std():.4f} mean|.|={_np.abs(v).mean():.4f}")


if __name__ == "__main__":
    main()
