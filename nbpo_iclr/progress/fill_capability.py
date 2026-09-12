"""Fill tab:general-capability from the artifacts, and stop hand-editing it.

Table 3 was the last hand-maintained exhibit: seven benchmark columns over nine
methods, each cell a seed mean plus a sample standard deviation, each column read
out of a different artifact family. Every arm that landed today cost a manual
edit, and roughly twenty-eight cells are still to come from scalarized DPO,
PROSPER and MOPO. Hand-editing that many cells is how a stale number gets in.

Reads, per column:
  IFEval, GSM8K  evaluations/deterministic_uf4_<arm>_v1/{ifeval,gsm8k}_*.summary.json
  HB             analysis/harmbench_uf4_<n>arm.json/*.summary.json  (latest n)
  XS             analysis/xstest_local[_<n>arm]/xstest_local_scores.json (latest)
  AE2, AH2       analysis/alpaca_arena_uf4[_<n>arm]/*.summary.json    (latest)
  MT             analysis/mtbench[_<n>arm]/mtbench_scores.json        (latest)

The base row is not a measurement for AE2 and AH2, which are win rates against
it; those stay dashes. A column is written only for the seeds that actually have
it, and the Seeds entry reports the objective panel's seed count, so a column
with fewer seeds than that is named in the printed report rather than silently
averaged.
"""
from __future__ import annotations

import json
import os
import re
import statistics as st
import subprocess
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
TEX = PAPER / "main_v6.tex"
BEGIN, END = "% BEGIN AUTO TAB3 BODY", "% END AUTO TAB3 BODY"
ROOT = "/work/uf4_20260910"
REPAIR = "/work/nbpo_repair_20260909"

FAMILIES = [("Base", "base", ["base"]),
            ("NBPO", "nbpo_mse", None),
            ("Fixed-reference Nash", "fixedref_mse", None),
            ("BT-RM--Nash", "btrm_mse", None),
            ("Game-utilitarian", "util_mse", None),
            ("Global game-maxmin", "maxmin_mse", None),
            ("Scalarized DPO (uniform)", "dpo_uniform_mse", None),
            ("PROSPER (adapt.)", "prosper_mse", None),
            ("MOPO (adapt.)", "mopo_mse", None)]
COLUMNS = ("ifeval", "gsm8k", "hb", "xs", "ae2", "ah2", "mt")
DIGITS = {"ifeval": 4, "gsm8k": 4, "hb": 4, "xs": 4, "ae2": 4, "ah2": 4, "mt": 2}


# Pinned rather than inherited: the reporter runs these fillers as subprocesses,
# and relying on KUBECONFIG being exported meant two of the three exhibits went
# stale whenever the daemon was started from a shell without it.
KUBECONFIG = "/home/sjkim/.kube/aipr-kubeconfig.yaml"


def pod(cmd):
    full = ("kubectl exec -n p-aipr nbpo-judge -c main -- bash -lc %s"
            % json.dumps(cmd))
    env = dict(os.environ, KUBECONFIG=KUBECONFIG)
    try:
        return subprocess.run(full, shell=True, capture_output=True, text=True,
                              timeout=180, env=env).stdout
    except Exception:
        return ""


def collect():
    """arm -> {column: value}, read off the pod by a script that lives there.

    An earlier version inlined the collector as a heredoc through pod(), which
    JSON-quotes its argument: the newlines came out as literal backslash-n and
    the heredoc never closed. The collector is a file on the pod instead.
    """
    raw = pod("cd %s/code && python3 collect_capability.py" % ROOT)
    if not raw.strip():
        raise SystemExit("could not read the capability artifacts from the pod")
    return json.loads(raw)


def seed_count(tex, label):
    """The Seeds entry the objective table reports for this family, if any."""
    m = re.search(re.escape(label) + r" & \$(\d)\$ &", tex)
    return m.group(1) if m else None


def main():
    data = collect()
    arms = data["arms"]
    tex = TEX.read_text(encoding="utf-8")
    obj = tex.split("% BEGIN AUTO TAB1 BODY")[1].split("% END AUTO TAB1 BODY")[0]

    lines, report = [], {}
    for label, prefix, explicit in FAMILIES:
        members = explicit if explicit else sorted(
            a for a in arms if a.startswith(prefix + "_s"))
        seeds = seed_count(obj, label) if label != "Base" else "1"
        cells, sds, got = [], [], {}
        for col in COLUMNS:
            vals = [arms[a][col] for a in members if col in arms.get(a, {})]
            if label == "Base" and col in ("ae2", "ah2"):
                cells.append("--"); sds.append(""); continue
            if not vals:
                cells.append(r"\pending"); sds.append(""); continue
            d = DIGITS[col]
            cells.append("$%.*f$" % (d, st.mean(vals)))
            sds.append("$%.*f$" % (d, st.stdev(vals)) if len(vals) > 1 else "")
            got[col] = len(vals)
        report[label] = {"seeds_in_objective_table": seeds, "cells": got,
                         "members": members}
        lines.append("%s & %s & %s\\\\" % (label, ("$%s$" % seeds) if seeds else r"\pending",
                                           " & ".join(cells)))
        if any(sds):
            lines.append("\\quad\\textit{sample SD over seeds} & & %s\\\\" % " & ".join(sds))

    body = "\n".join(lines)
    if BEGIN not in tex or END not in tex:
        print(json.dumps({"status": "markers missing; add %s / %s to the table"
                          % (BEGIN, END)}), file=sys.stderr)
        return 1
    start = tex.index(BEGIN) + len(BEGIN)
    tex = tex[:start] + "\n" + body + "\n" + tex[tex.index(END):]
    tmp = TEX.with_suffix(".tex.cap_tmp")
    tmp.write_text(tex, encoding="utf-8")
    os.replace(tmp, TEX)
    print(json.dumps({"rows": len(FAMILIES), "sources": data["sources"],
                      "per_family": report}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
