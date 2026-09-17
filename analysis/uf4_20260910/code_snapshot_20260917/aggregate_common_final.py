"""Aggregate the final evaluation on the prompts common to every evaluated method.

The per-method valid counts already in the manuscript (1,967 and 1,960) are each
method's own denominator. They are NOT the common set, and reporting them side by
side compares arms on slightly different prompts. This computes the intersection:
a prompt counts only when EVERY method has both presentation orders parsed for
ALL four criteria.

Uncertainty is a whole-prompt paired bootstrap. Each replicate resamples prompts
and carries every method, rubric and order for that prompt together, then
recomputes the minimum-objective statistic inside the replicate rather than
bootstrapping the minimum of the point estimates.

Seed spread and prompt uncertainty are reported separately and never merged.
"""
from __future__ import annotations

import argparse, hashlib, json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/work/uf4_20260910")
CRITERIA = ("instruction_following", "truthfulness", "honesty", "helpfulness")


def file_hash(p):
    h = hashlib.sha256()
    with open(p, "rb") as s:
        for c in iter(lambda: s.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load_arm(arm):
    path = ROOT / "evaluation/final_eval" / arm / "judgments.jsonl"
    orders, status = defaultdict(dict), defaultdict(int)
    with path.open() as stream:
        for line in stream:
            r = json.loads(line)
            status[r["status"]] += 1
            if r["status"] == "ok":
                orders[(r["criterion"], r["prompt_id"])][r["order"]] = r["value_for_policy"]
    value = {k: 0.5 * (v[0] + v[1]) for k, v in orders.items() if len(v) == 2}
    complete = {p for (c, p) in value if all((cc, p) in value for cc in CRITERIA)}
    return {"value": value, "complete_prompts": complete, "status": dict(status),
            "sha256": file_hash(path), "n_judgments": sum(status.values())}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--planned", type=int, default=2000)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--out", default=str(ROOT / "analysis/final_eval_common.json"))
    args = ap.parse_args()

    arms = {a: load_arm(a) for a in args.arms}
    common = set.intersection(*[a["complete_prompts"] for a in arms.values()])
    common = sorted(common)
    rng = np.random.default_rng(20260911)
    idx = rng.integers(0, len(common), size=(args.bootstrap, len(common)))

    report = {"planned_final_prompts": args.planned, "arms": list(arms),
              "common_prompts": len(common),
              "per_method_own_denominator": {a: len(v["complete_prompts"]) for a, v in arms.items()},
              "judgment_status": {a: v["status"] for a, v in arms.items()},
              "judgments_sha256": {a: v["sha256"] for a, v in arms.items()},
              "denominator_note": ("per_method_own_denominator is each arm's own valid count; "
                                   "common_prompts is the intersection used for every number "
                                   "below, so the arms are compared on identical prompts"),
              "bootstrap": "%d whole-prompt paired replicates; every method, rubric and order for "
                           "a prompt is resampled together and the minimum objective is "
                           "recomputed inside each replicate" % args.bootstrap,
              "results": {}}

    per_arm = {}
    for arm, d in arms.items():
        mat = np.array([[d["value"][(c, p)] for c in CRITERIA] for p in common])  # (P, K)
        per_arm[arm] = mat
        boot_means = mat[idx].mean(axis=1)                      # (B, K)
        boot_min = boot_means.min(axis=1)                       # min recomputed per replicate
        report["results"][arm] = {
            "n_common": len(common),
            **{c: {"win_rate": float(mat[:, k].mean()),
                   "ci95": [float(np.quantile(boot_means[:, k], .025)),
                            float(np.quantile(boot_means[:, k], .975))]}
               for k, c in enumerate(CRITERIA)},
            "min_objective": {"point": float(mat.mean(axis=0).min()),
                              "ci95": [float(np.quantile(boot_min, .025)),
                                       float(np.quantile(boot_min, .975))]}}
        print(json.dumps({"arm": arm, "n_common": len(common),
                          **{c: round(float(mat[:, k].mean()), 4) for k, c in enumerate(CRITERIA)},
                          "min": round(float(mat.mean(axis=0).min()), 4)}), flush=True)

    # seed spread, kept separate from prompt uncertainty
    families = defaultdict(list)
    for arm in arms:
        families[arm.rsplit("_s", 1)[0]].append(arm)
    report["seed_spread"] = {}
    for fam, members in families.items():
        if len(members) < 2:
            continue
        entry = {}
        for k, c in enumerate(CRITERIA):
            vals = [float(per_arm[m][:, k].mean()) for m in sorted(members)]
            entry[c] = {"seeds": sorted(members), "values": vals,
                        "mean": float(np.mean(vals)),
                        "sample_sd": float(np.std(vals, ddof=1))}
        report["seed_spread"][fam] = entry
        print(json.dumps({"family": fam, "n_seeds": len(members),
                          **{c: "%.4f+-%.4f" % (entry[c]["mean"], entry[c]["sample_sd"])
                             for c in CRITERIA}}), flush=True)
    report["seed_spread_note"] = ("sample SD across policy seeds under one frozen teacher; it is "
                                  "not prompt uncertainty and the two are never combined")
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"written": args.out, "common_prompts": len(common),
                      "planned": args.planned}), flush=True)


if __name__ == "__main__":
    main()
