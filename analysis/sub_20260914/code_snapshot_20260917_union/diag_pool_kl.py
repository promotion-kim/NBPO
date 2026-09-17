"""Conditional pool fidelity of a fitted arm: KL(p* || p_neural) on its own pool.

This is the Pool KL column of tab:union-diagnostic-template. The numerically
important part -- the sequence log-likelihood under a checkpoint, summed over the
pool's own cached label positions, and the importance reweighting

    p_neural(i) proportional to p_t(i) exp{log pi_theta(y_i) - log pi_t(y_i)}

-- is imported from diag_neural_pools.py rather than retyped, so this and the
UF-4 projection diagnostic compute the same quantity.

What differs is the panel. The union panels solve each prompt on its own, so
p* comes from the arm's per-prompt solution artifact and the proximal centre is
the uniform measure over that prompt's eight learner occurrences, which is what
the solver was handed. Both facts are asserted against the stored artifact
before anything is scored, and the reported KL is over the prompts of the arm's
own dev split.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/work/uf4_20260910/code")
from diag_neural_pools import load_pool_rows, sequence_logprobs

ROOT = Path("/work/uf4_20260910")
POOL = 8


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="+", required=True,
                    help="arm directory names under arms/, e.g. uw1_pw_nbpo")
    ap.add_argument("--targets", nargs="+", required=True,
                    help="the target set whose p* pairs with each arm, same order")
    ap.add_argument("--pool", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--base", default="/work/models/bases/Qwen2.5-7B-Instruct")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if len(args.arms) != len(args.targets):
        raise SystemExit("--arms and --targets must pair up")

    from transformers import AutoModelForCausalLM
    device = "cuda"

    # one prompt set: the prompts every named arm solved on this split
    stars = {}
    for target in args.targets:
        npz = ROOT / "targets" / target / ("%s_per_prompt.npz" % args.split)
        z = np.load(npz, allow_pickle=True)
        pi = np.asarray(z["pi"], dtype=np.float64)
        if pi.shape[1] != POOL:
            raise SystemExit("%s: p* has %d columns, expected %d"
                             % (target, pi.shape[1], POOL))
        if float(np.abs(pi.sum(axis=1) - 1.0).max()) > 1e-9:
            raise SystemExit("%s: p* rows do not sum to one" % target)
        stars[target] = {str(p): pi[k] for k, p in enumerate(z["prompt_ids"])}
    ids = sorted(set.intersection(*[set(v) for v in stars.values()]))
    if not ids:
        raise SystemExit("the named arms share no solved prompt on this split")

    candidates = {pid: ["%s:learner:%d" % (pid, i) for i in range(POOL)] for pid in ids}
    wanted = {c for v in candidates.values() for c in v}
    rows = load_pool_rows(args.pool, wanted)
    missing = sorted(wanted - set(rows))
    if missing:
        raise SystemExit("missing cached tokenization for %d occurrences" % len(missing))
    order = [c for pid in ids for c in candidates[pid]]

    def score(path):
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16,
                                                     attn_implementation="sdpa",
                                                     local_files_only=True).to(device).eval()
        out = sequence_logprobs(model, rows, order, device, args.batch)
        del model
        torch.cuda.empty_cache()
        return dict(zip(order, out))

    base_logp = score(args.base)
    # the solver's proximal centre for a round-1 union panel is the uniform
    # measure over that prompt's eight learner draws
    pt = np.full(POOL, 1.0 / POOL)
    report = {"pool": args.pool, "split": args.split, "base": args.base,
              "prompts": len(ids),
              "proximal_centre": "uniform over the prompt's eight learner occurrences",
              "estimator": ("p_neural proportional to p_t * exp(log pi_theta - log pi_base), "
                            "normalized on the sampled support"),
              "note": ("importance reweighting on the sampled support; not the policy's "
                       "distribution over all responses"),
              "arms": {}}
    for arm, target in zip(args.arms, args.targets):
        arm_logp = score(str(ROOT / "arms" / arm))
        kl_star_neural, kl_star_t, tv, rms = [], [], [], []
        for pid in ids:
            cs = candidates[pid]
            delta = np.array([arm_logp[c] - base_logp[c] for c in cs], dtype=np.float64)
            logits = np.log(pt) + delta
            logits -= logits.max()
            p = np.exp(logits)
            p /= p.sum()
            ps = np.clip(stars[target][pid], 1e-12, None)
            kl_star_neural.append(float(np.sum(ps * (np.log(ps) - np.log(np.clip(p, 1e-12, None))))))
            kl_star_t.append(float(np.sum(ps * (np.log(ps) - np.log(pt)))))
            tv.append(float(0.5 * np.abs(p - ps).sum()))
            rms.append(float(np.sqrt(np.mean(delta ** 2))))
        report["arms"][arm] = {
            "target_set": target,
            "pool_kl_star_to_neural_mean": float(np.mean(kl_star_neural)),
            "pool_kl_star_to_centre_mean": float(np.mean(kl_star_t)),
            "fraction_of_prompts_closer_than_the_centre":
                float(np.mean(np.asarray(kl_star_neural) < np.asarray(kl_star_t))),
            "tv_star_to_neural_mean": float(np.mean(tv)),
            "logratio_rms_mean": float(np.mean(rms))}
        print(json.dumps({arm: report["arms"][arm]}), flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
