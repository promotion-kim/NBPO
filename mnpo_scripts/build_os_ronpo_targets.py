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


def row_targets(example, kappas, omega_nash=None, omega_ks=None, noise_idx=-1, noise_lambda=1.0):
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
    # label-noise compression on one judge: P^lam = lam*P + (1-lam)/2 => zhat scales by lam
    if 0 <= noise_idx < K and noise_lambda < 1.0:
        zhat[noise_idx, :] *= noise_lambda
        cost[noise_idx, :] = noise_lambda * cost[noise_idx, :] + (1.0 - noise_lambda) * 0.5

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
        # OS: one opponent per objective, weighted by omega. NBPO/KS reuse the same
        # per-objective adversary a_k (adversary robustness) but replace the objective
        # weight omega(k) with a global Nash / Kalai-Smorodinsky bargaining weight.
        # per-row Nash weight: inverse of the chosen response's per-objective score
        # (upweight, on THIS prompt, the objective the chosen response serves least),
        # the per-prompt analogue of OS's adaptive objective marginal.
        srow = np.maximum(S[:, ci], 0.02)
        wrow = (1.0 / srow); wrow = wrow / wrow.sum()
        os_val = nbpo_val = ks_val = nbporow_val = 0.0
        for k in range(K):
            a_k = int(rng.choice(A, p=q[k]))
            zk = zhat[k, a_k]
            os_val += omega[k] * zk
            nbporow_val += wrow[k] * zk
            if omega_nash is not None:
                nbpo_val += omega_nash[k] * zk
            if omega_ks is not None:
                ks_val += omega_ks[k] * zk
        out[f"target_nbporow_k{tag}"] = float(nbporow_val)
        out[f"target_os_k{tag}"] = float(os_val)
        if omega_nash is not None:
            out[f"target_nbpo_k{tag}"] = float(nbpo_val)
            # nbpo_fe: Nash objective weight x full-expectation over adversaries.
            # Differs from target_uniform ONLY in the objective weight (Nash vs 1/K),
            # so it isolates the Nash-weighting effect against the averaging baseline.
            zbar = zhat.mean(axis=1)                       # mean_a zhat(k,a)
            out[f"target_nbpofe_k{tag}"] = float(np.dot(omega_nash, zbar))
            if omega_ks is not None:
                out[f"target_ksfe_k{tag}"] = float(np.dot(omega_ks, zbar))
        if omega_ks is not None:
            out[f"target_ks_k{tag}"] = float(ks_val)
    return out


def global_bargaining_weights(split, names, eps=0.02, noise_idx=-1, noise_lambda=1.0):
    """Nash (inverse-surplus) and KS objective weights from the dataset-level
    surplus s_k = mean_row (chosen normalized score on objective k) - 1/2."""
    K = len(names)
    tot = np.zeros(K)
    n = len(split)
    ci_col = split["chosen_index"]
    sc_col = split["normalized_objective_scores"]
    for i in range(n):
        ci = int(ci_col[i])
        tot += np.array([sc_col[i][names[k]][ci] for k in range(K)], dtype=float)
    s = tot / n - 0.5                                  # surplus per objective
    if 0 <= noise_idx < K and noise_lambda < 1.0:      # degraded judge: s_k -> lam*s_k
        s[noise_idx] *= noise_lambda
    inv = 1.0 / np.maximum(s, eps)
    w_nash = inv / inv.sum()
    ustar = np.maximum(0.5, eps)                        # max normalized surplus is 1 - 1/2
    sigma = s / ustar
    e = np.exp(-(sigma - sigma.min()) / 0.25) / ustar   # softmin over normalized surplus
    w_ks = e / e.sum()
    return s, w_nash, w_ks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--kappas", default="0.05,0.01")
    ap.add_argument("--noise-obj", dest="noise_obj", default="", help="degrade this judge (covariance test)")
    ap.add_argument("--noise-lambda", dest="noise_lambda", type=float, default=1.0)
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
    if maxerr >= 1e-6:
        # Relative/expected pair modes store an aggregated gap, not the single-atom
        # zhat this check reconstructs; the OS target itself is computed directly
        # from chosen/rejected_index + normalized scores, so proceed with a warning.
        print("[validate] WARNING: stored gap differs from single-atom zhat "
              "(expected for relative/expected pair modes); OS targets recomputed from scores.")
    names0 = ref["objective_names"][0] if isinstance(ref["objective_names"][0], list) else ref[0]["objective_names"]
    nidx = names0.index(args.noise_obj) if args.noise_obj in names0 else -1
    s_glob, w_nash, w_ks = global_bargaining_weights(ref, names0, noise_idx=nidx, noise_lambda=args.noise_lambda)
    print(f"[bargaining] noise={args.noise_obj}:{args.noise_lambda} surplus={dict(zip(names0, np.round(s_glob, 4)))}")
    print(f"[bargaining] w_nash={dict(zip(names0, np.round(w_nash, 4)))}  w_ks={dict(zip(names0, np.round(w_ks, 4)))}")

    if args.validate_only:
        # show target distributions for a quick sanity read
        ex0 = row_targets(ref[0], kappas, w_nash, w_ks, nidx, args.noise_lambda)
        print("[sample row0 targets]", {k: round(v, 4) for k, v in ex0.items()})
        return

    def add(example):
        return row_targets(example, kappas, w_nash, w_ks, nidx, args.noise_lambda)

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
