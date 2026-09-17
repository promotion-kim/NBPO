"""Paired whole-prompt bootstrap between the two pilot arms' fresh responses.

The two arms answered the same 200 development prompts against the same four
reference responses, so their difference is paired by prompt. A prompt enters
only when all four of its reference comparisons parsed in both orders for the
objective in question, and the minimum is recomputed inside every replicate.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

DIAG = Path("/work/uf4_20260910/analysis/diag_20260914")
CRIT = ("instruction_following", "truthfulness", "honesty", "helpfulness")


def per_prompt(arm):
    cells = defaultdict(lambda: defaultdict(dict))
    with (DIAG / "fresh_verdicts" / ("%s.jsonl" % arm)).open() as stream:
        for line in stream:
            r = json.loads(line)
            if r["status"] != "ok":
                continue
            cells[(r["prompt_id"], r["criterion"])][r["right"]][r["order"]] = r["value_for_left"]
    out = {}
    for (pid, crit), refs in cells.items():
        vals = [0.5 * (v[0] + v[1]) for v in refs.values() if 0 in v and 1 in v]
        if len(vals) == 4:
            out.setdefault(crit, {})[pid] = float(np.mean(vals))
    return out


def main():
    a = per_prompt("diag_proj_sampled_s42")
    b = per_prompt("diag_proj_all_s42")
    shared = sorted(set.intersection(*[set(a[c]) for c in CRIT], *[set(b[c]) for c in CRIT]))
    rng = np.random.default_rng(20260914)
    draws = rng.integers(0, len(shared), size=(2000, len(shared)))
    out = {"paired_prompts": len(shared), "replicates": 2000, "bootstrap_seed": 20260914,
           "arms": ["diag_proj_sampled_s42", "diag_proj_all_s42"], "objectives": {}}
    ma, mb = [], []
    for c in CRIT:
        x = np.array([a[c][p] for p in shared])
        y = np.array([b[c][p] for p in shared])
        diff = (y - x)[draws].mean(axis=1)
        out["objectives"][c] = {
            "sampled": float(x.mean()), "all_candidates": float(y.mean()),
            "difference": float((y - x).mean()),
            "ci95": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
        }
        ma.append(x[draws].mean(axis=1))
        mb.append(y[draws].mean(axis=1))
    dw = np.min(np.stack(mb), axis=0) - np.min(np.stack(ma), axis=0)
    out["w_min"] = {"difference": float(dw.mean()),
                    "ci95": [float(np.percentile(dw, 2.5)), float(np.percentile(dw, 97.5))]}
    (DIAG / "projection/pilot_fresh_ci.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
