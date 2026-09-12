"""Fill tab:feedback-cost from the artifacts, the last hand-maintained table.

Four of its columns change every time an arm finishes, and twelve cells are
still to come from scalarized DPO, PROSPER and MOPO. The conventions are the
subtle part and are taken from the caption rather than invented:

  Unique labels  elicited semantic comparisons, shared by every method that
                 draws on this pool: 4 objectives x 10000 prompts x C(8,2).
  Teacher        shared, not per method; the column repeats the one figure.
  Policy         PER SEED, and the trainer-reported runtime times the four
                 devices held -- not the controller's wall total, which also
                 covers loading and checkpointing and is reported in the text.
  Gen./eval.     per seed, the measured cost of one arm's fresh-response
                 generation plus its independent judging.
  Total          teacher + policy + gen/eval.
  Trials         zero by declaration; no sweep was run.

A family's Policy and Gen./eval. are means over the seeds that have them, so a
family with one finished seed shows that seed's cost rather than a third of it.
"""
from __future__ import annotations

import json
import os
import statistics as st
import subprocess
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
TEX = PAPER / "main_v6.tex"
BEGIN, END = "% BEGIN AUTO COST BODY", "% END AUTO COST BODY"
KUBECONFIG = "/home/sjkim/.kube/aipr-kubeconfig.yaml"
ROOT = "/work/uf4_20260910"
SHARED_TEACHER = 1.8
UNIQUE_LABELS = r"$1.12\times10^{6}$"
# Every row is now a per-seed cost averaged over that method's seeds, so the
# first row is "NBPO" rather than "NBPO, seed 42": it has three seeds like the
# others, and quoting one of them beside four family means is not comparable.
ROWS = [("NBPO", "nbpo_mse", "0"),
        ("Fixed-reference Nash", "fixedref_mse", "0"),
        ("Game-utilitarian", "util_mse", "0"),
        ("BT-RM--Nash", "btrm_mse", "0"),
        ("Global game-maxmin", "maxmin_mse", "0"),
        ("PROSPER (adapt.)", "prosper_mse", None),
        ("MOPO (adapt.)", "mopo_mse", None),
        ("DPO (uniform / sweep)", "dpo_", "0")]


def pod(cmd):
    full = ("kubectl exec -n p-aipr nbpo-judge -c main -- bash -lc %s" % json.dumps(cmd))
    env = dict(os.environ, KUBECONFIG=KUBECONFIG)
    try:
        return subprocess.run(full, shell=True, capture_output=True, text=True,
                              timeout=180, env=env).stdout
    except Exception:
        return ""


def collect():
    """Read the cost inputs from a collector that lives on the pod.

    Not a heredoc through pod(): that helper JSON-quotes its argument, so
    newlines arrive as literal backslash-n and the heredoc never closes. The
    same mistake cost a cycle on the capability filler; the collector is a file.
    """
    raw = pod("cd %s/code && python3 collect_cost_table.py" % ROOT)
    if not raw.strip():
        raise SystemExit("could not read the cost artifacts from the pod")
    return json.loads(raw)


def main():
    data = collect()
    policy, gen = data["policy"], data["gen_eval"]
    lines, report = [], {}
    for label, prefix, trials in ROWS:
        seeds = sorted(a for a in policy if a.startswith(prefix))
        gseeds = sorted(a for a in gen if a.startswith(prefix))
        pol = st.mean(policy[a] for a in seeds) if seeds else None
        ge = st.mean(gen[a] for a in gseeds) if gseeds else None
        cells = [UNIQUE_LABELS, "$%.1f$" % SHARED_TEACHER]
        cells.append("$%.1f$" % pol if pol else r"\pending")
        cells.append("$%.1f$" % ge if ge else r"\pending")
        # Total EXCLUDES the shared teacher. The teacher column repeats one
        # shared figure across nine rows; adding it into each total would
        # charge a single 1.8 GPU-hours nine times. The hand-maintained table
        # had this right and the first version of this filler did not.
        cells.append("$%.1f$" % (pol + ge) if (pol and ge) else r"\pending")
        cells.append("$%s$" % trials if trials is not None else r"\pending")
        lines.append("%s & %s\\\\" % (label, " & ".join(cells)))
        report[label] = {"policy_seeds": seeds, "gen_eval_seeds": gseeds,
                         "policy": round(pol, 3) if pol else None,
                         "gen_eval": round(ge, 3) if ge else None}

    tex = TEX.read_text(encoding="utf-8")
    if BEGIN not in tex or END not in tex:
        print(json.dumps({"status": "markers missing", "marker": BEGIN}), file=sys.stderr)
        return 1
    start = tex.index(BEGIN) + len(BEGIN)
    tex = tex[:start] + "\n" + "\n".join(lines) + "\n" + tex[tex.index(END):]
    tmp = TEX.with_suffix(".tex.cost_tmp")
    tmp.write_text(tex, encoding="utf-8")
    os.replace(tmp, TEX)
    print(json.dumps({"rows": len(ROWS), "per_row": report}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
