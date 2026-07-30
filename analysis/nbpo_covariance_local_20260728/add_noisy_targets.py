#!/usr/bin/env python3
"""Add fixed-row noisy-judge targets for the NBPO covariance test."""

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


def normalized(weights):
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}, total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verdicts", type=Path, required=True)
    ap.add_argument("--pairs-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--noise-obj", default="honesty")
    ap.add_argument("--noise-lambda", type=float, default=0.3)
    ap.add_argument("--eps", type=float, default=0.02)
    ap.add_argument("--ks-beta", type=float, default=0.25)
    args = ap.parse_args()

    rows = [json.loads(x) for x in args.verdicts.read_text().splitlines() if x.strip()]
    pref_list = defaultdict(list)
    for r in rows:
        if r["valid"]:
            key = (str(r["prompt_id"]), r["objective"],
                   str(r["policy_seed"]), str(r["ref_seed"]))
            pref_list[key].append(float(r["policy_win"]))
    clean = {k: sum(v) / len(v) for k, v in pref_list.items()}
    noisy = dict(clean)
    kept = 0
    affected = 0
    for key, value in clean.items():
        if key[1] != args.noise_obj:
            continue
        affected += 1
        h = hashlib.sha256(("noise42|" + "|".join(key)).encode()).hexdigest()
        if int(h[:16], 16) / 16**16 < args.noise_lambda:
            kept += 1
        else:
            noisy[key] = 0.5

    objectives = sorted({k[1] for k in clean})

    def moments(pref):
        surplus, ideal = {}, {}
        for obj in objectives:
            vals = [v for k, v in pref.items() if k[1] == obj]
            surplus[obj] = sum(vals) / len(vals) - 0.5
            by_response = defaultdict(list)
            for (pid, o, ps, rs), value in pref.items():
                if o == obj:
                    by_response[(pid, ps)].append(value)
            ideal[obj] = max(sum(v) / len(v) for v in by_response.values()) - 0.5
        return surplus, ideal

    def raw_weights(surplus, ideal):
        sigma = {o: surplus[o] / max(ideal[o], args.eps) for o in objectives}
        worst = min(objectives, key=lambda o: surplus[o])
        return {
            "nbs": {o: 1 / max(surplus[o], args.eps) for o in objectives},
            "ks": {o: math.exp(-sigma[o] / args.ks_beta) /
                        max(ideal[o], args.eps) for o in objectives},
            "unif": {o: 1.0 for o in objectives},
            "maxmin": {o: float(o == worst) for o in objectives},
        }, sigma

    s_clean, i_clean = moments(clean)
    s_noisy, i_noisy = moments(noisy)
    raw_clean, sigma_clean = raw_weights(s_clean, i_clean)
    raw_noisy, sigma_noisy = raw_weights(s_noisy, i_noisy)
    clean_den = {rule: sum(w.values()) for rule, w in raw_clean.items()}
    norm_noisy = {rule: normalized(w)[0] for rule, w in raw_noisy.items()}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    target_stats = defaultdict(list)
    for split in ("train", "test"):
        source = args.pairs_dir / f"pairs_{split}.jsonl"
        output = args.out_dir / f"pairs_{split}.jsonl"
        with source.open() as src, output.open("w") as dst:
            for line in src:
                rec = json.loads(line)
                pid = str(rec["prompt_id"])
                sa, sb = str(rec["policy_seed_a"]), str(rec["policy_seed_b"])
                rs = str(rec["reference_seed"])
                z = {o: noisy.get((pid, o, sa, rs), 0.5) -
                        noisy.get((pid, o, sb, rs), 0.5) for o in objectives}
                clean_t = sum(rec["bpo_z"][o] for o in objectives) / len(objectives)
                flip = -1.0 if clean_t < 0 else 1.0
                rec["bpo_z_noise03"] = z
                rec["bpo_weights_noise03"] = norm_noisy
                for rule in ("nbs", "ks", "unif", "maxmin"):
                    target = flip * sum(raw_noisy[rule][o] * z[o] for o in objectives)
                    target /= clean_den[rule]
                    key = f"bpo_target_{rule}_noise03"
                    rec[key] = target
                    target_stats[rule].append(target)
                dst.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary = {
        "noise": {
            "objective": args.noise_obj,
            "lambda": args.noise_lambda,
            "seed": 42,
            "mode": "swap-averaged preference retained with probability lambda, else 0.5",
            "affected": affected,
            "retained": kept,
        },
        "clean": {"surplus": s_clean, "ideal": i_clean, "sigma": sigma_clean,
                  "raw_weight_sum": clean_den},
        "noisy": {"surplus": s_noisy, "ideal": i_noisy, "sigma": sigma_noisy,
                  "normalized_weights": norm_noisy},
        "target_scale": "all noisy raw targets divided by the corresponding clean raw-weight sum",
        "rows": {rule: len(vals) for rule, vals in target_stats.items()},
    }
    (args.out_dir / "noise_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
