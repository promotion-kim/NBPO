"""Build a results bundle from a campaign artifact on the pod.

The bundle is what fill_results_json.py accepts: a contract id, the artifact's
pod path, the artifact's sha256 as computed ON the pod (so the hash belongs to
the bytes the campaign wrote, not to a local copy), and one record per table
cell with its value and, where the artifact carries one, a 95% interval.

Two groups are supported, each a direct read of the declared statistics:

  screen   tab:dataset_readiness   {safe,uf,wild} x {N,C,D,Gamma,TV}, whichever
           the artifact's own cells block declares
  factorial tab:factorial_game     {ll,lh,hl,hh} x {C,D,Gamma,Delta,TV}

Nothing is computed here beyond selecting fields: if a statistic is absent from
the artifact the cell is left out of the bundle and stays unmeasured.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

KUBECONFIG = "/home/sjkim/.kube/aipr-kubeconfig.yaml"
POD = ["kubectl", "exec", "-n", "p-aipr", "nbpo-judge", "-c", "main", "--", "bash", "-lc"]


def pod(cmd, timeout=180):
    env = dict(os.environ, KUBECONFIG=KUBECONFIG)
    r = subprocess.run(POD + [cmd], capture_output=True, text=True, timeout=timeout,
                       env=env)
    if r.returncode != 0:
        raise SystemExit("pod command failed: %s" % (r.stderr or r.stdout)[:300])
    return r.stdout


def fetch(path):
    raw = pod("cat %s" % path)
    sha = pod("sha256sum %s | cut -d' ' -f1" % path).strip()
    return json.loads(raw), sha


SCREEN_SUFFIXES = ("Gamma", "TV", "N", "C", "D")   # Gamma/TV before the single letters


def screen_cells(doc):
    """tab:dataset_readiness cells, for whichever row the artifact declares.

    The artifact names its own row prefix in its cells block (safe, uf or
    wild), so one function serves all three rows and the prefix is never
    guessed here. A statistic the artifact does not carry is simply absent:
    the checklist-native row has no gamma* or TV because prompt-specific items
    give no fixed objective identity to define the aggregate game over, and
    those cells stay unmeasured rather than being filled with a substitute.
    """
    cells = doc.get("cells")
    if not cells:
        raise SystemExit("the screening artifact has no cells block")
    surrogate = doc.get("surrogate") or {}
    out = {}
    for key, value in cells.items():
        suffix = next((s for s in SCREEN_SUFFIXES if key.endswith(s)), None)
        if suffix is None:
            raise SystemExit("unrecognized screening cell %r" % key)
        if value is None:
            continue
        if suffix == "N":
            rec = {"value": value,
                   "denominator": "prompts whose whole screening panel parsed",
                   "prompts_read": doc.get("n_prompts_read"),
                   "unit_of_analysis": doc.get("unit_of_analysis")}
        elif suffix == "C":
            rec = {"value": value,
                   "definition": "repeated oriented 3-cycle fraction, both repeat panels",
                   "eligible_triples": doc.get("eligible_triples_repeated"),
                   "single_panel_rate": doc.get("C_single_panel"),
                   "unique_response_rate": doc.get("C_repeated_unique"),
                   "tie_margin": doc.get("delta_tie_margin")}
        elif suffix == "D":
            rec = {"value": value,
                   "definition": doc.get("D_definition",
                       "cross-objective disagreement on strictly resolved pairs"),
                   "denominator": doc.get("disagreement_denominator")}
            # the mean over objective pairs can hide one conflicting pair, and
            # the pooled and per-prompt weightings can disagree, so carry both
            for extra in ("D_per_objective_pair", "D_disagreement_prompt_mean",
                          "D_disagreement_prompt_sd", "D_prompts_contributing"):
                if doc.get(extra) is not None:
                    rec[extra] = doc[extra]
        elif suffix == "Gamma":
            rec = {"value": value,
                   "definition": "max_p min_k [V_k(p)-d_k] on the 8-response audit surrogate",
                   "beta": surrogate.get("beta"),
                   "not_the_policy_contract": True}
            cert = surrogate.get("gamma_star_certificate")
            if cert:
                # SLSQP stops at the 400-iteration cap on these panels, so the
                # achieved value is a feasible lower bound and the certificate
                # carries the tangent-plane upper bound and the gap.
                rec["certificate"] = {k: cert.get(k) for k in
                                      ("certified_upper_bound", "rho_star_gap",
                                       "slsqp_status", "slsqp_iterations",
                                       "converged", "hit_iteration_cap")}
        else:   # TV
            rec = {"value": value,
                   "definition": "total variation between exact NBPO and utilitarian targets"}
        out[key] = rec
    return out


def factorial_cells(doc):
    out = {}
    for cell in ("ll", "lh", "hl", "hh"):
        v = doc["cells"][cell]
        if not v["instances_usable"]:
            continue
        for key, stat in (("C", "C"), ("D", "D"), ("Gamma", "gamma_star"),
                          ("Delta", "Delta_M"), ("TV", "target_TV")):
            s = v[stat]
            if s["mean"] is None:
                continue
            out[cell + key] = {"value": s["mean"], "ci95": s["ci95"], "sd": s["sd"],
                               "instance_seeds": s["n"],
                               "uncertainty": "instance-seed variation, not policy seeds"}
    return out


# tab:stress_results declares helpfulness and harmlessness as the two
# objectives, in that order, so W_1 is helpfulness and W_2 is harmlessness.
# The fresh-eval artifact keys them by name and sorts alphabetically, which puts
# harmlessness first: the mapping is written out here rather than inferred from
# position.
FRESH_COLUMNS = [("Wone", "helpfulness"), ("Wtwo", "harmlessness")]


def fresh_cells(doc, prefix, seeds):
    """One row of tab:stress_results from a fresh_eval complete.json."""
    wins, cis = doc["win_rates"], doc.get("win_rate_ci95") or {}
    out = {prefix + "S": {"value": seeds,
                          "definition": ("frozen generation replicates for the base row, "
                                         "training seeds for a trained arm"),
                          "note": doc.get("arm_tag")},
           prefix + "N": {"value": doc["n_prompts_complete"],
                          "definition": "test prompts with every scheduled verdict parsed",
                          "prompts_judged": doc.get("n_prompts_judged")}}
    for key, objective in FRESH_COLUMNS:
        record = {"value": wins[objective], "objective": objective,
                  "definition": ("order-averaged win rate against the frozen "
                                 "four-response base reference bank, ties 0.5")}
        if cis.get(objective):
            record["ci95"] = cis[objective]
        out[prefix + key] = record
    out[prefix + "Min"] = {"value": doc["min_objective"],
                           "ci95": doc.get("min_objective_ci95"),
                           "definition": ("minimum over objectives, recomputed inside "
                                          "every bootstrap replicate")}
    return out


def delta_cells(doc, prefix):
    """The Delta column for one row: the paired difference against NBPO."""
    return {prefix + "Diff": {
        "value": doc["delta_min"], "ci95": doc["delta_min_ci95"],
        "definition": ("this row's seed-wise minimum minus NBPO's, paired by prompt, "
                       "with the minimum recomputed inside every bootstrap replicate"),
        "n_common_prompts": doc["n_common_prompts"],
        "row_min_on_common": doc["row_min_on_common"],
        "nbpo_min_on_common": doc["reference_min_on_common"],
        "excludes_zero": doc["excludes_zero"]}}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--group", required=True,
                    choices=("screen", "factorial", "fresh", "delta"))
    ap.add_argument("--prefix", help="fresh only: the row prefix, e.g. base or nbpo")
    ap.add_argument("--seeds", type=int, default=1,
                    help="fresh only: replicates for the base row, seeds for an arm")
    ap.add_argument("--artifact", required=True, help="path on the pod")
    ap.add_argument("--contract-id", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    doc, sha = fetch(args.artifact)
    if args.group == "screen":
        cells = screen_cells(doc)
    elif args.group == "factorial":
        cells = factorial_cells(doc)
    elif args.group == "fresh":
        if not args.prefix:
            raise SystemExit("--prefix is required for the fresh group")
        cells = fresh_cells(doc, args.prefix, args.seeds)
    else:
        if not args.prefix:
            raise SystemExit("--prefix is required for the delta group")
        cells = delta_cells(doc, args.prefix)
    if not cells:
        raise SystemExit("no measured cells in %s" % args.artifact)
    bundle = {"contract_id": args.contract_id, "artifact": args.artifact,
              "sha256": sha, "group": args.group, "cells": cells}
    Path(args.out).write_text(json.dumps(bundle, indent=2) + "\n")
    print(json.dumps({"bundle": args.out, "sha256": sha, "cells": sorted(cells)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
