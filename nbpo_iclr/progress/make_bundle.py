"""Build a results bundle from a campaign artifact on the pod.

The bundle is what fill_results_json.py accepts: a contract id, the artifact's
pod path, the artifact's sha256 as computed ON the pod (so the hash belongs to
the bytes the campaign wrote, not to a local copy), and one record per table
cell with its value and, where the artifact carries one, a 95% interval.

Two groups are supported, each a direct read of the declared statistics:

  screen   tab:dataset_readiness   safeN safeC safeD safeGamma safeTV
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


def screen_cells(doc):
    cells = doc.get("cells")
    if not cells:
        raise SystemExit("the screening artifact has no cells block")
    out = {"safeN": {"value": cells["safeN"],
                     "denominator": "prompts whose whole screening panel parsed",
                     "prompts_read": doc.get("n_prompts_read")},
           "safeC": {"value": cells["safeC"],
                     "definition": "repeated oriented 3-cycle fraction, both repeat panels",
                     "eligible_triples": doc.get("eligible_triples_repeated"),
                     "single_panel_rate": doc.get("C_single_panel"),
                     "unique_response_rate": doc.get("C_repeated_unique"),
                     "tie_margin": doc.get("delta_tie_margin")},
           "safeD": {"value": cells["safeD"],
                     "definition": "cross-objective disagreement on strictly resolved pairs",
                     "denominator": doc.get("disagreement_denominator")},
           "safeGamma": {"value": cells["safeGamma"],
                         "definition": "max_p min_k [V_k(p)-d_k] on the 8-response audit surrogate",
                         "beta": (doc.get("surrogate") or {}).get("beta"),
                         "not_the_policy_contract": True},
           "safeTV": {"value": cells["safeTV"],
                      "definition": "total variation between exact NBPO and utilitarian targets"}}
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--group", required=True, choices=("screen", "factorial"))
    ap.add_argument("--artifact", required=True, help="path on the pod")
    ap.add_argument("--contract-id", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    doc, sha = fetch(args.artifact)
    cells = screen_cells(doc) if args.group == "screen" else factorial_cells(doc)
    if not cells:
        raise SystemExit("no measured cells in %s" % args.artifact)
    bundle = {"contract_id": args.contract_id, "artifact": args.artifact,
              "sha256": sha, "group": args.group, "cells": cells}
    Path(args.out).write_text(json.dumps(bundle, indent=2) + "\n")
    print(json.dumps({"bundle": args.out, "sha256": sha, "cells": sorted(cells)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
