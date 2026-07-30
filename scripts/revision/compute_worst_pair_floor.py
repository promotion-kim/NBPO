"""Theory-faithful worst-pair win-rate eval against a frozen response pool
(feedback #21). For each evaluated policy m the pairwise floor is

    floor(m) = E_x[min_{k,a} P_k( y_m(x) > pool_a(x) )],
    P_k(y > a) = sigmoid(scale * (s_k(y) - s_k(a))),

where a indexes the 5 frozen pool slots (stage-1 policy decode seeds), k the
three reward objectives, and s_k are raw RM scores min-max normalized per
prompt over the pool responses (the training-time scale); the policy response
is mapped into that same per-prompt pool scale. This estimates the quantity
the inner adversary actually minimizes over, unlike the per-objective means
reported elsewhere. Bootstrap CIs are prompt-level.

Inputs: --pool_scored (test_{obj}.jsonl with the frozen pool + raw scores) and
--eval_scored (eval_{obj}.jsonl from the controlled eval; response_model_names
aligned with all_rm_scores).
"""
import argparse
import json
from pathlib import Path

import numpy as np

OBJS = ["skywork", "athene", "armo"]


def load_pool(pool_dir):
    pool = {}
    for obj in OBJS:
        for line in open(Path(pool_dir) / f"test_{obj}.jsonl"):
            r = json.loads(line)
            pool.setdefault(r["prompt"], {})[obj] = np.array(r["all_rm_scores"], dtype=float)
    return pool


def load_eval(eval_dir):
    ev = {}
    names = None
    for obj in OBJS:
        for line in open(Path(eval_dir) / f"eval_{obj}.jsonl"):
            r = json.loads(line)
            names = r["response_model_names"]
            ev.setdefault(r["prompt"], {})[obj] = np.array(r["all_rm_scores"], dtype=float)
    return ev, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool_scored", required=True)
    ap.add_argument("--eval_scored", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--scale", type=float, default=8.0)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--kappa", type=float, default=0.05)
    args = ap.parse_args()

    pool = load_pool(args.pool_scored)
    ev, model_names = load_eval(args.eval_scored)
    common = sorted(set(pool) & set(ev))
    print(f"[floor] prompts: pool={len(pool)} eval={len(ev)} common={len(common)}")

    A = len(next(iter(pool.values()))[OBJS[0]])
    M = len(model_names)
    # win[m, k, a, x]
    win = np.zeros((M, len(OBJS), A, len(common)))
    for xi, prompt in enumerate(common):
        for ki, obj in enumerate(OBJS):
            ps = pool[prompt][obj]
            lo, hi = ps.min(), ps.max()
            rng_ = hi - lo
            pn = (ps - lo) / rng_ if rng_ > 0 else np.full_like(ps, 0.5)
            for mi in range(M):
                y = ev[prompt][obj][mi]
                yn = (y - lo) / rng_ if rng_ > 0 else 0.5
                win[mi, ki, :, xi] = 1.0 / (1.0 + np.exp(-args.scale * (yn - pn)))

    rng = np.random.default_rng(42)
    idx = rng.integers(0, len(common), size=(args.bootstrap, len(common)))
    out = {"models": model_names, "objectives": OBJS, "num_prompts": len(common),
           "scale": args.scale, "kappa": args.kappa, "pool_slots": A, "per_model": {}}
    for mi, m in enumerate(model_names):
        grid = win[mi].mean(axis=2)                      # K x A
        per_prompt_floor = win[mi].min(axis=(0, 1))
        floor = float(per_prompt_floor.mean())
        k_star, a_star = np.unravel_index(grid.argmin(), grid.shape)
        boots = per_prompt_floor[idx].mean(axis=1)
        costs = win[mi].transpose(2, 0, 1)
        soft = -args.kappa * (
            np.log(np.exp(-costs / args.kappa).mean(axis=(1, 2)))
        )
        out["per_model"][m] = {
            "pairwise_floor": floor,
            "floor_ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "soft_pairwise_floor": float(soft.mean()),
            "argmin_objective": OBJS[k_star],
            "argmin_pool_slot": int(a_star),
            "worst_objective_mean": float(grid.mean(axis=1).min()),
            "grid": {OBJS[k]: [round(float(v), 4) for v in grid[k]] for k in range(len(OBJS))},
        }
        print(f"  {m:12s} floor={floor:.4f} CI=[{out['per_model'][m]['floor_ci95'][0]:.4f},"
              f"{out['per_model'][m]['floor_ci95'][1]:.4f}] argmin=({OBJS[k_star]},slot{a_star})")
    Path(args.output).write_text(json.dumps(out, indent=2) + "\n")
    print(f"[floor] wrote {args.output}")


if __name__ == "__main__":
    main()
