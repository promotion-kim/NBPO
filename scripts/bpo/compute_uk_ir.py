#!/usr/bin/env python3
"""Theory-native multi-objective metrics from eval files.

For each arm we estimate the anchored utility u_k = P_k(pi > mu), the average
probability judge k prefers the arm's response to the reference mu's response on
the same prompt (mu = the --base arm). From {u_k} we report exactly the
quantities the NBPO guarantees speak to:

  worst_u   = min_k u_k                      (individual rationality / worst judge
                                              vs reference; averaging can push < 1/2)
  nash_welf = (prod_k max(u_k-1/2,0))^(1/K)  (Nash social welfare NBPO maximizes)
  ir_viol   = #{k: u_k < 1/2}                (judges left below the reference)

u_k is a paired win rate over shared prompts, so it needs no reward-scale
normalization: it is the same [0,1] anchored utility the theorems use.
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
        r = json.loads(l)
        s = r.get("all_rm_scores") or []
        if not s:
            continue
        key = str(r.get("prompt_id", r.get("prompt", i)))
        d[key] = float(np.mean(s))
    return d


def u_vector(root, arm, base_arm, objs):
    us, n = [], 0
    for o in objs:
        a = per_prompt(root, arm, o)
        b = per_prompt(root, base_arm, o)
        if a is None or b is None:
            return None
        keys = [k for k in a if k in b]
        n = len(keys)
        wins = [1.0 if a[k] > b[k] else (0.5 if a[k] == b[k] else 0.0) for k in keys]
        us.append(float(np.mean(wins)))
    return np.array(us), n


def boot_diff(root, arm1, arm2, base_arm, objs, stat, reps=2000):
    """Paired bootstrap CI on stat(arm1)-stat(arm2) over shared prompts."""
    def paired(arm):
        cols = {}
        for o in objs:
            a = per_prompt(root, arm, o); b = per_prompt(root, base_arm, o)
            keys = sorted(set(a) & set(b))
            cols[o] = (keys, a, b)
        return cols
    c1, c2 = paired(arm1), paired(arm2)
    keys = sorted(set.intersection(*[set(c1[o][0]) for o in objs], *[set(c2[o][0]) for o in objs]))
    keys = np.array(keys)
    n = len(keys)

    def stat_on(cols, idx):
        u = []
        for o in objs:
            _, a, b = cols[o]
            w = np.array([1.0 if a[k] > b[k] else (0.5 if a[k] == b[k] else 0.0) for k in keys[idx]])
            u.append(w.mean())
        return stat(np.array(u))
    ds = []
    for r in range(reps):
        idx = np.random.default_rng(r).integers(0, n, n)
        ds.append(stat_on(c1, idx) - stat_on(c2, idx))
    ds = np.array(ds)
    return ds.mean(), np.percentile(ds, 2.5), np.percentile(ds, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--base", default="base", help="reference arm mu")
    ap.add_argument("--objs", required=True, help="comma list, e.g. helpful,harmless,humor")
    ap.add_argument("--arms", required=True, help="comma list of arms to score")
    ap.add_argument("--nash-vs", default="", help="arm to bootstrap NBPO against (e.g. uniform)")
    ap.add_argument("--nbpo", default="", help="the NBPO arm name for the bootstrap")
    args = ap.parse_args()
    objs = args.objs.split(",")
    arms = args.arms.split(",")

    print(f"{'arm':12s} " + "  ".join(f"{o[:6]:>6s}" for o in objs) +
          "   worstU   NashW    IRviol")
    rows = {}
    for arm in arms:
        r = u_vector(args.root, arm, args.base, objs)
        if r is None:
            print(f"{arm:12s} (missing)"); continue
        u, n = r
        rows[arm] = u
        sp = u - 0.5
        nashw = float(np.prod(np.maximum(sp, 1e-9)) ** (1.0 / len(u))) if (sp > 0).all() else -1.0
        irv = int((u < 0.5).sum())
        print(f"{arm:12s} " + "  ".join(f"{x:6.3f}" for x in u) +
              f"   {u.min():.3f}   {nashw:+.4f}   {irv}   (n={n})")

    if args.nbpo and args.nash_vs and args.nbpo in rows and args.nash_vs in rows:
        for name, stat in [("worstU", lambda u: u.min()),
                           ("NashW", lambda u: float(np.prod(np.maximum(u - 0.5, 1e-9)) ** (1.0 / len(u))))]:
            m, lo, hi = boot_diff(args.root, args.nbpo, args.nash_vs, args.base, objs, stat)
            sig = "SIG" if (lo > 0 or hi < 0) else "n.s."
            print(f"[boot] {args.nbpo}-{args.nash_vs} {name}: {m:+.4f}  95%CI[{lo:+.4f},{hi:+.4f}]  {sig}")


if __name__ == "__main__":
    main()
