#!/usr/bin/env python3
"""Build FIXED-REFERENCE Anchored-BPO training pairs (beta = infinity BASELINE).

LEGACY LABEL -- this is NOT the finite-temperature NBPO of Algorithm 1. It
computes fixed-anchor surpluses s_k = P_k(pi > mu) - 1/2 against the frozen
reference (the beta -> infinity limit, where the adaptive opponent collapses to
mu), uses normalized/clipped static weights, always pairs against the FIRST
reference response, and computes no nu*, V_{k,beta}, d_k, or dual descent.
The finite-pool NBPO realization lives in scripts/nbpo/ (judge matrix ->
preference tensors -> solve_nbpo_dual -> build_nbpo_pairs -> loss_type=nbpo).
The `maxmin` rule below is the static one-hot on the pre-training worst
objective -- a legacy baseline, distinct from scripts/nbpo's adversarial
max-min controls. KS here is its own experiment line, not one of the current
paper's max-min controls.

Aggregates swap-averaged verdicts into per-judge anchored surpluses
s_k = win_rate - 1/2, computes the three weight rules (NBS inverse-surplus,
KS softmin over normalized surpluses, uniform), and writes pairs_train/test
jsonl with one signed target column per rule: bpo_target_{nbs,ks,unif}.
Pair = two policy responses per prompt; z_k = pref_k(y, y'') - pref_k(y', y'')
against the shared first reference response; target = sum_k w_k z_k.
(The pref.get(..., 0.5) fallbacks below are part of this frozen baseline; the
finite-pool NBPO path never imputes a missing comparison.)
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verdicts", type=Path, required=True)
    ap.add_argument("--policy-files", nargs="+", required=True, help="seed=path.json")
    ap.add_argument("--pair-mode", choices=["selfplay", "pvb"], default="selfplay",
                    help="selfplay: pair two policy samples; pvb: pair policy vs base reference")
    ap.add_argument("--base-files", nargs="+", default=[], help="seed=path.json (base gens, pvb mode)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--eps", type=float, default=0.02)
    ap.add_argument("--ks-beta", type=float, default=0.25)
    ap.add_argument("--test-prompts", type=int, default=125)
    ap.add_argument("--split-salt", default="bpo-v1")
    ap.add_argument("--all-seed-pairs", action="store_true")
    ap.add_argument("--noise-obj", default="", help="degrade this judge: P^lam = lam*P + (1-lam)/2")
    ap.add_argument("--noise-lambda", type=float, default=1.0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.verdicts.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r["valid"]]
    # label-noise compression on one judge: replace a (1-lambda) fraction of its
    # verdicts with coin flips (0.5), deterministically by task hash (skew-symmetry safe).
    if args.noise_obj and args.noise_lambda < 1.0:
        for r in rows:
            if r["objective"] == args.noise_obj:
                h = int(hashlib.sha256(f'{args.noise_obj}|{r["task_id"]}'.encode()).hexdigest()[:8], 16)
                if (h % 10000) / 10000.0 >= args.noise_lambda:
                    r["policy_win"] = 0.5
    # swap-average: mean of the two orders per (pid, obj, pseed, rseed)
    pref: dict[tuple, list] = defaultdict(list)
    for r in rows:
        pref[(r["prompt_id"], r["objective"], r["policy_seed"], r["ref_seed"])].append(r["policy_win"])
    pref = {k: sum(v) / len(v) for k, v in pref.items()}
    objectives = sorted({k[1] for k in pref})
    pseeds = sorted({k[2] for k in pref})
    rseeds = sorted({k[3] for k in pref})

    # anchored surplus s_k and per-response win rates (for the KS ideal)
    s, ustar = {}, {}
    for obj in objectives:
        vals = [v for k, v in pref.items() if k[1] == obj]
        s[obj] = sum(vals) / len(vals) - 0.5
        by_resp = defaultdict(list)
        for (pid, o, ps, rs), v in pref.items():
            if o == obj:
                by_resp[(pid, ps)].append(v)
        ustar[obj] = max(sum(v) / len(v) for v in by_resp.values()) - 0.5  # u*_k - 1/2

    def normalize(w):
        z = sum(w.values())
        return {k: v / z for k, v in w.items()}

    eps = args.eps
    w_nbs = normalize({o: 1.0 / max(s[o], eps) for o in objectives})
    sigma = {o: s[o] / max(ustar[o], eps) for o in objectives}
    import math
    w_ks = normalize({o: math.exp(-sigma[o] / args.ks_beta) / max(ustar[o], eps) for o in objectives})
    w_unif = {o: 1.0 / len(objectives) for o in objectives}
    worst_obj = min(objectives, key=lambda o: s[o])  # egalitarian: all weight on the worst-off judge
    w_maxmin = {o: (1.0 if o == worst_obj else 0.0) for o in objectives}

    def load(files):
        return {sp.split("=")[0]: {str(r["prompt_id"]): r for r in json.loads(Path(sp.split("=", 1)[1]).read_text())}
                for sp in files}
    policy = load(args.policy_files)
    base = load(args.base_files) if args.pair_mode == "pvb" else {}
    pids = sorted(set(k[0] for k in pref))
    test_ids = set(sorted(pids, key=lambda p: hashlib.sha256(
        f"{args.split_salt}|{p}".encode()).hexdigest())[:args.test_prompts])

    seed_pairs = (list(itertools.combinations(pseeds, 2))
                  if args.all_seed_pairs else [(pseeds[0], pseeds[1])])
    r0 = rseeds[0]
    pvb = args.pair_mode == "pvb"          # pair = (policy sa, base r0); z_k = 2 P_k(policy>base) - 1
    out = {"train": [], "test": []}
    for pid in pids:
      for sa, sb in seed_pairs:
        if pvb:
            if pid not in policy[sa] or pid not in base[r0]:
                continue
            z = {o: 2.0 * pref.get((pid, o, sa, r0), 0.5) - 1.0 for o in objectives}
            ya, yb = policy[sa][pid], base[r0][pid]
        else:
            if pid not in policy[sa] or pid not in policy[sb]:
                continue
            z = {o: pref.get((pid, o, sa, r0), 0.5) - pref.get((pid, o, sb, r0), 0.5) for o in objectives}
            ya, yb = policy[sa][pid], policy[sb][pid]
        t_unif = sum(w_unif[o] * z[o] for o in objectives)
        # orient chosen/rejected by the uniform target; per-rule targets keep their sign
        flip = -1.0 if t_unif < 0 else 1.0
        chosen, rejected = (ya, yb) if flip > 0 else (yb, ya)
        rec = {
            "prompt_id": pid, "prompt": str(ya["prompt"]),
            "policy_seed_a": sa, "policy_seed_b": sb, "reference_seed": r0,
            "chosen": str(chosen["generated_text"]), "rejected": str(rejected["generated_text"]),
            "bpo_z": z, "bpo_weights": {"nbs": w_nbs, "ks": w_ks, "unif": w_unif, "maxmin": w_maxmin},
            "bpo_target_nbs": flip * sum(w_nbs[o] * z[o] for o in objectives),
            "bpo_target_ks": flip * sum(w_ks[o] * z[o] for o in objectives),
            "bpo_target_unif": flip * t_unif,
            "bpo_target_maxmin": flip * sum(w_maxmin[o] * z[o] for o in objectives),
        }
        out["test" if pid in test_ids else "train"].append(rec)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split, recs in out.items():
        with (args.out_dir / f"pairs_{split}.jsonl").open("w") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n = sum(len(v) for k, v in pref.items() if False) or len(pref) // (len(objectives))
    summary = {
        "surplus": s, "ustar_minus_half": ustar, "sigma": sigma,
        "noise": {"obj": args.noise_obj, "lambda": args.noise_lambda},
        "weights": {"nbs": w_nbs, "ks": w_ks, "unif": w_unif, "maxmin": w_maxmin},
        "train_pairs": len(out["train"]), "test_pairs": len(out["test"]),
        "hoeffding_margin_per_judge": (2 * (0.5 / (n or 1)) ** 0.5) if n else None,
    }
    (args.out_dir / "bpo_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
