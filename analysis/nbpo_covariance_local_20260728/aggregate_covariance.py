#!/usr/bin/env python3
"""Aggregate the frozen clean-judge covariance evaluation."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


RULES = ("nbs", "ks", "unif", "maxmin")
OBJECTIVES = ("helpfulness", "harmlessness", "honesty")


def load(path: Path):
    grouped = defaultdict(list)
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            if row["valid"]:
                grouped[(str(row["prompt_id"]), row["objective"])].append(
                    float(row["policy_win"]))
    return {key: sum(values) / len(values) for key, values in grouped.items()}


def matrix(pref, keys):
    return np.array([[pref[(pid, obj)] for obj in OBJECTIVES] for pid in keys])


def interval(values):
    return [float(x) for x in np.percentile(values, [2.5, 97.5])]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    data, keys = {}, None
    for condition in ("clean", "noise03"):
        for rule in RULES:
            name = f"{condition}_{rule}"
            pref = load(args.root / "eval" / name / "verdicts.jsonl")
            current = {pid for pid, obj in pref if obj in OBJECTIVES}
            keys = current if keys is None else keys & current
            data[name] = pref
    keys = sorted(keys)
    arrays = {name: matrix(pref, keys) for name, pref in data.items()}
    rng = np.random.default_rng(42)
    boots = {rule: [] for rule in RULES}
    contrasts = {f"{b}-{r}": [] for b in ("nbs", "ks")
                 for r in ("unif", "maxmin")}
    for _ in range(2000):
        index = rng.integers(0, len(keys), len(keys))
        drift = {}
        for rule in RULES:
            clean = arrays[f"clean_{rule}"][index].mean(0)
            noisy = arrays[f"noise03_{rule}"][index].mean(0)
            drift[rule] = float(np.abs(noisy - clean).max())
            boots[rule].append(drift[rule])
        for name in contrasts:
            left, right = name.split("-")
            contrasts[name].append(drift[left] - drift[right])
    result = {
        "n_prompts": len(keys),
        "objectives": OBJECTIVES,
        "rules": {},
        "contrasts": {},
    }
    for rule in RULES:
        clean = arrays[f"clean_{rule}"].mean(0)
        noisy = arrays[f"noise03_{rule}"].mean(0)
        result["rules"][rule] = {
            "clean_u": dict(zip(OBJECTIVES, clean.tolist())),
            "noisy_u": dict(zip(OBJECTIVES, noisy.tolist())),
            "clean_avg": float(clean.mean()),
            "clean_worst": float(clean.min()),
            "noisy_avg": float(noisy.mean()),
            "noisy_worst": float(noisy.min()),
            "drift_linf": float(np.abs(noisy - clean).max()),
            "drift_linf_ci95": interval(boots[rule]),
        }
    passed = True
    for name, values in contrasts.items():
        ci = interval(values)
        result["contrasts"][name] = {
            "mean": float(np.mean(values)), "ci95": ci,
            "smaller_drift": ci[1] < 0,
        }
        passed &= ci[1] < 0
    result["decision"] = "PASS" if passed else "FAIL_OR_INCONCLUSIVE"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
