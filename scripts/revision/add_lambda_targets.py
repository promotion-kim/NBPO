"""Add OS-RONPO target columns recomputed at alternative preference scales
(lambda sweep, feedback #18). Same rows/pairs/logps as the shipped stage-2
arms; only the scalar target changes, so each lambda arm is a matched A/B
against the lambda=8 OS arm. Pair selection itself was done at lambda=8 and is
deliberately reused so the sweep isolates the training-signal scale.

  target_os_k{K}_lam{L} : OS estimator with P = sigmoid(L*(s_y - s_a))
                          in both the adversary cost and the target zhat.
"""
import argparse
import hashlib
import numpy as np
from datasets import load_from_disk


def os_target_at_scale(example, kappa, scale):
    names = example["objective_names"]
    S = np.array([example["normalized_objective_scores"][k] for k in names], dtype=float)
    K, A = S.shape
    ci, ri = int(example["chosen_index"]), int(example["rejected_index"])
    P = 1.0 / (1.0 + np.exp(-scale * (S[:, :, None] - S[:, None, :])))
    cost = P.mean(axis=1)
    zhat = P[:, ci, :] - P[:, ri, :]
    logits = -cost.reshape(-1) / kappa
    logits -= logits.max()
    sig = np.exp(logits); sig /= sig.sum()
    sigma = sig.reshape(K, A)
    omega = sigma.sum(axis=1)
    q = sigma / omega[:, None]
    seed = int.from_bytes(hashlib.sha256(
        (f"lam{scale:g}|" + str(example.get("prompt_id") or example.get("prompt"))).encode()
    ).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    val = sum(omega[k] * zhat[k, int(rng.choice(A, p=q[k]))] for k in range(K))
    return float(val)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--kappa", type=float, default=0.05)
    ap.add_argument("--scales", default="4,16")
    ap.add_argument("--num_proc", type=int, default=12)
    args = ap.parse_args()
    scales = [float(x) for x in args.scales.split(",")]
    ktag = f"{args.kappa:g}".replace(".", "p")

    ds = load_from_disk(args.input_dir)

    def add(ex):
        return {f"target_os_k{ktag}_lam{s:g}": os_target_at_scale(ex, args.kappa, s)
                for s in scales}

    ds2 = ds.map(add, num_proc=args.num_proc, desc="lambda-sweep OS targets")
    ds2.save_to_disk(args.output_dir)
    tr = ds2["train"]
    print(f"[built] {args.output_dir} rows(train)={len(tr)}")
    for s in scales:
        c = f"target_os_k{ktag}_lam{s:g}"
        v = np.array(tr[c], dtype=float)
        print(f"  {c:26s} mean={v.mean():+.4f} std={v.std():.4f} mean|.|={np.abs(v).mean():.4f}")
    ref = np.array(tr["target_os_k0p05"], dtype=float)
    print(f"  {'target_os_k0p05 (lam8)':26s} mean={ref.mean():+.4f} std={ref.std():.4f} mean|.|={np.abs(ref).mean():.4f}")


if __name__ == "__main__":
    main()
