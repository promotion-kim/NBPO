"""Regenerate data/uf_tradeoffs.csv from the live common-set aggregator.

Two reasons this had to change. The file still carried the pre-repair
single-seed points on the superseded 1967/1960 denominators -- numbers the
appendix explicitly supersedes -- so the figure was plotting retracted values
while its caption said the sweep was pending. And it was hand-maintained, so it
would have drifted again at the next arm.

One row per evaluated policy seed rather than per method. A family mean has no
defined prompt-bootstrap interval (the campaign keeps prompt uncertainty and
seed spread separate and never pools them), while a seed-level point has its own
interval and the scatter across a method's seeds is itself the panel's main
finding. Dominance marks stay descriptive over the plotted rows, as the plotter's
own docstring requires.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POD = ("kubectl exec -n p-aipr nbpo-judge -c main -- bash -lc "
       "'cat /work/uf4_20260910/analysis/final_eval_common_auto.json'")
CRITERIA = (("if", "instruction_following"), ("truth", "truthfulness"),
            ("honesty", "honesty"), ("help", "helpfulness"))
LABEL = {"nbpo_mse": "NBPO", "fixedref_mse": "Fixed-reference Nash",
         "btrm_mse": "BT-RM--Nash", "util_mse": "Game-utilitarian",
         "maxmin_mse": "Global game-maxmin",
         # The baselines and the declared weight sweep, so a measured arm is
         # plotted instead of sitting in the planned block for another cycle.
         "dpo_uniform_mse": "Scalarized DPO (uniform)",
         "dpo_if_only_mse": "Scalarized DPO (IF only)",
         "dpo_truth_only_mse": "Scalarized DPO (truth only)",
         "dpo_honesty_only_mse": "Scalarized DPO (honesty only)",
         "dpo_help_only_mse": "Scalarized DPO (help only)",
         "dpo_help_heavy_mse": "Scalarized DPO (help heavy)",
         "dpo_truth_heavy_mse": "Scalarized DPO (truth heavy)",
         "prosper_mse": "PROSPER (adapt.)", "mopo_mse": "MOPO (adapt.)"}
HEADER = (["method_id", "method_label", "dataset", "test_hash", "protocol_id", "judge_id",
           "n_prompts", "n_seeds", "seed_ids", "status", "uncertainty_type"]
          + [f"{k}_{s}" for k, _ in CRITERIA for s in ("mean", "ci_low", "ci_high")])
PROTOCOL = "order-averaged win rate vs the fixed base response, ties 0.5"
JUDGE = "local Qwen3-14B rev 40c06982"
UNCERTAINTY = ("2000 whole-prompt paired bootstrap; prompt variability only, "
               "never pooled with seed spread")


def main():
    env = dict(os.environ, KUBECONFIG="/home/sjkim/.kube/aipr-kubeconfig.yaml")
    raw = subprocess.run(POD, shell=True, capture_output=True, text=True,
                         timeout=300, env=env).stdout
    if not raw.strip():
        print(json.dumps({"status": "aggregator unreachable; csv left alone"}), file=sys.stderr)
        return 1
    report = json.loads(raw)
    results, n_common = report["results"], report["common_prompts"]
    # judgments_sha256 is a per-arm mapping, not a string; stringifying it put a
    # truncated dict repr into the provenance column. Digest the mapping so the
    # column identifies this exact set of judgments in one value.
    import hashlib
    test_hash = hashlib.sha256(
        json.dumps(report["judgments_sha256"], sort_keys=True).encode()).hexdigest()[:16]

    rows = [{"method_id": "base", "method_label": "Base", "dataset": "UF-4",
             "test_hash": test_hash,
             "protocol_id": ("reference point for every axis: win rates are measured against "
                             "this policy's cached responses, so it sits at 0.5 by definition "
                             "and is not judged against itself"),
             "judge_id": JUDGE, "n_prompts": n_common, "n_seeds": 1, "seed_ids": "n/a",
             "status": "reference_not_plotted", "uncertainty_type": ""}]
    for arm in sorted(results):
        family, _, seed = arm.rpartition("_s")
        if family not in LABEL:
            continue
        row = {"method_id": arm, "method_label": "%s (seed %s)" % (LABEL[family], seed),
               "dataset": "UF-4", "test_hash": test_hash, "protocol_id": PROTOCOL,
               "judge_id": JUDGE, "n_prompts": n_common, "n_seeds": 1, "seed_ids": seed,
               "status": "reported", "uncertainty_type": UNCERTAINTY}
        for short, long in CRITERIA:
            entry = results[arm][long]
            lo, hi = entry["ci95"]
            # The estimate is `win_rate`. An earlier version fell back to the
            # interval midpoint when it could not find a `mean` key, which is
            # not the estimate: for BT-RM seed 42 instruction following it gave
            # 0.5162 against the actual 0.5159. Fail loudly instead.
            if "win_rate" not in entry:
                raise KeyError("%s/%s has no win_rate; keys are %s"
                               % (arm, long, sorted(entry)))
            row[short + "_mean"] = "%.4f" % float(entry["win_rate"])
            row[short + "_ci_low"] = "%.4f" % lo
            row[short + "_ci_high"] = "%.4f" % hi
        rows.append(row)
    # A declared row with no evaluated seed stays visible as planned, so the
    # figure's caption can count what is still missing without a hand-kept list.
    evaluated = {a.rpartition("_s")[0] for a in results}
    planned = [(fam, lab) for fam, lab in sorted(LABEL.items()) if fam not in evaluated]
    for mid, label in planned:
        rows.append({"method_id": mid, "method_label": label, "dataset": "UF-4",
                     "status": "planned"})

    out = ROOT / "data/uf_tradeoffs.csv"
    tmp = out.with_suffix(".csv.tmp")
    with tmp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=HEADER, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in HEADER})
    tmp.replace(out)
    plotted = sum(1 for r in rows if r.get("status") == "reported")
    print(json.dumps({"csv": str(out), "plotted_seed_points": plotted,
                      "n_common_prompts": n_common, "test_hash": test_hash,
                      "planned_rows": sum(1 for r in rows if r.get("status") == "planned"),
                      "planned": [r["method_label"] for r in rows
                                  if r.get("status") == "planned"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
