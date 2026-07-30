#!/usr/bin/env python3
"""Multi-seed worst-judge robustness (min_k u_k) across seeds, with pooled bootstrap.

u_k = P_k(pi > mu) = paired win rate of an arm's response over the reference mu's
response on the same prompt, judge k. worst_U = min_k u_k (the worst-off judge vs
the reference). Seed 42 evals live at <root>/eval/<arm>; seeds 43+ at
<root>/eval/s<seed>_<arm>. The reference eval (<root>/eval/<base>) is shared.
"""
import argparse, json, os
import numpy as np


def per_prompt(root, arm, obj):
    f = os.path.join(root, "eval", arm, f"eval_{obj}.jsonl")
    if not os.path.exists(f):
        return None
    d = {}
    for i, l in enumerate(open(f)):
        if not l.strip():
            continue
        r = json.loads(l); s = r.get("all_rm_scores") or []
        if s:
            d[str(r.get("prompt_id", r.get("prompt", i)))] = float(np.mean(s))
    return d


def arm_name(seed, arm, first_seed):
    return arm if seed == first_seed else f"s{seed}_{arm}"


def u_per_prompt(root, arm, base_arm, objs):
    """Return {prompt: [win_k...]} 0/0.5/1 per judge over shared prompts."""
    cols = {}
    for o in objs:
        a = per_prompt(root, arm, o); b = per_prompt(root, base_arm, o)
        if a is None or b is None:
            return None
        cols[o] = (a, b)
    keys = sorted(set.intersection(*[set(cols[o][0]) & set(cols[o][1]) for o in objs]))
    out = {}
    for k in keys:
        out[k] = np.array([1.0 if cols[o][0][k] > cols[o][1][k]
                           else (0.5 if cols[o][0][k] == cols[o][1][k] else 0.0) for o in objs])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--base", default="base")
    ap.add_argument("--objs", required=True)
    ap.add_argument("--arms", required=True)
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--pairs", default="", help="semicolon list a-b for pooled bootstrap on worst_U")
    args = ap.parse_args()
    objs = args.objs.split(","); arms = args.arms.split(","); seeds = [int(s) for s in args.seeds.split(",")]
    first = seeds[0]

    # per (arm, seed) -> {prompt: winvec}
    data = {}
    for arm in arms:
        for sd in seeds:
            data[(arm, sd)] = u_per_prompt(args.root, arm_name(sd, arm, first), args.base, objs)

    print(f"worst_U = min_k u_k  (seeds {seeds}, objs {objs})")
    print(f"{'arm':10s} " + "  ".join(f"s{sd}" for sd in seeds) + "     mean±SD")
    for arm in arms:
        ws = []
        for sd in seeds:
            d = data[(arm, sd)]
            ws.append(np.mean([v for v in d.values()], axis=0).min() if d else float("nan"))
        ws = np.array(ws)
        print(f"{arm:10s} " + "  ".join(f"{w:.3f}" for w in ws) + f"     {np.nanmean(ws):.3f}±{np.nanstd(ws):.3f}")

    for pair in [p for p in args.pairs.split(";") if p]:
        a1, a2 = pair.split("-")
        # pooled bootstrap: resample prompts within each seed, average worst_U diff over seeds
        diffs = []
        for r in range(3000):
            per_seed = []
            for sd in seeds:
                d1, d2 = data[(a1, sd)], data[(a2, sd)]
                if not d1 or not d2:
                    continue
                keys = np.array(sorted(set(d1) & set(d2)))
                idx = np.random.default_rng(r * 100 + sd).integers(0, len(keys), len(keys))
                bk = keys[idx]
                w1 = np.mean([d1[k] for k in bk], axis=0).min()
                w2 = np.mean([d2[k] for k in bk], axis=0).min()
                per_seed.append(w1 - w2)
            if per_seed:
                diffs.append(np.mean(per_seed))
        diffs = np.array(diffs)
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        sig = "SIG" if (lo > 0 or hi < 0) else "n.s."
        print(f"[pooled boot] {a1}-{a2} worst_U = {diffs.mean():+.4f}  95%CI[{lo:+.4f},{hi:+.4f}]  {sig}")


if __name__ == "__main__":
    main()
