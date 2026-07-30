#!/usr/bin/env python3
"""Verify the three kappa facts that constrain CODEX_PROMPT_8b_saferlhf_kappa_imbalance_20260717.md.

Replicates the target math of mnpo_scripts/build_os_ronpo_targets.py:row_targets exactly
(K=2 objectives, A=4 responses, homogeneous_oracle_preference_scale=8.0) on synthetic
normalized objective scores, and measures:

  (a) top-mass is invariant to kappa           -- argmax exp(-cost/kappa) = argmin cost, all kappa > 0
  (b) the two degenerate limits of kappa       -- entropy(sigma) and corr to each endpoint
  (c) which regime the repo's kappa values sit in

Run:  python analysis/ronpo_8b_reconstruction_20260714/verify_kappa_facts.py
"""
import numpy as np

K, A, SCALE, N_ROWS = 2, 4, 8.0, 3000
KAPPAS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 100.0]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    rng = np.random.default_rng(0)
    ent = {k: [] for k in KAPPAS}
    fe = {k: [] for k in KAPPAS}
    tm_by_kappa = {k: [] for k in KAPPAS}
    tm, unif = [], []

    for _ in range(N_ROWS):
        S = rng.random((K, A))
        ci, ri = 0, 1
        P = sigmoid(SCALE * (S[:, :, None] - S[:, None, :]))   # K x A(y) x A(a)
        cost = P.mean(axis=1)                                   # K x A, pi uniform over y
        zhat = P[:, ci, :] - P[:, ri, :]                        # K x A
        base_logits = -cost.reshape(-1)
        ka = int(np.argmax(base_logits))
        ks, as_ = divmod(ka, A)
        tm.append(float(zhat[ks, as_]))
        unif.append(float(zhat.mean()))                         # kappa -> inf limit

        for kappa in KAPPAS:
            l = base_logits / kappa
            l = l - l.max()
            sig = np.exp(l)
            sig /= sig.sum()
            ent[kappa].append(float(-(sig * np.log(sig + 1e-300)).sum() / np.log(K * A)))
            fe[kappa].append(float((sig.reshape(K, A) * zhat).sum()))
            kk = int(np.argmax(sig))
            k2, a2 = divmod(kk, A)
            tm_by_kappa[kappa].append(float(zhat[k2, a2]))

    tm = np.array(tm)
    unif = np.array(unif)

    # (a) invariance
    stack = np.stack([np.array(tm_by_kappa[k]) for k in KAPPAS])
    varies = int((stack.max(axis=0) - stack.min(axis=0) > 1e-12).sum())
    print(f"(a) top-mass target varies with kappa in {varies}/{N_ROWS} rows "
          f"(kappa range {min(KAPPAS)}..{max(KAPPAS)})")
    fe_stack = np.stack([np.array(fe[k]) for k in KAPPAS])
    fe_varies = int((fe_stack.max(axis=0) - fe_stack.min(axis=0) > 1e-6).sum())
    print(f"    full-exp target varies with kappa in {fe_varies}/{N_ROWS} rows")

    # (b) limits
    print(f"\n(b) {'kappa':>8} {'norm_entropy':>13} {'corr(fe,topmass)':>17} {'corr(fe,uniform)':>17}")
    for kappa in KAPPAS:
        f = np.array(fe[kappa])
        print(f"    {kappa:>8.3g} {np.mean(ent[kappa]):>13.4f} "
              f"{np.corrcoef(f, tm)[0, 1]:>17.4f} {np.corrcoef(f, unif)[0, 1]:>17.4f}")
    print(f"\n    top-mass mean={tm.mean():+.4f} std={tm.std():.4f}   (exact kappa->0 endpoint)")
    print(f"    uniform  mean={unif.mean():+.4f} std={unif.std():.4f}   (exact kappa->inf endpoint,"
          f" NO-ADVERSARY control)")
    print(f"    corr(top-mass, uniform) = {np.corrcoef(tm, unif)[0, 1]:.4f}")

    # (c) where the repo has actually looked
    print(f"\n(c) repo has only ever run kappa <= 0.05  ->  normalized entropy "
          f"<= {np.mean(ent[0.05]):.2f}, corr to hard argmax >= {np.corrcoef(np.array(fe[0.05]), tm)[0, 1]:.2f}."
          f"\n    The kappa > 0.05 half of the mechanism is unmeasured.")
    print("\nNOTE: entropies depend on the real cost distribution. Recompute this map on the "
          "actual\nprecomputed dataset before freezing the kappa grid; do not reuse these synthetic values.")


if __name__ == "__main__":
    main()
