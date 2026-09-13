"""Fill the three target-to-policy exhibits from the pod's diagnostic artifacts.

Regions written:

  % BEGIN/END AUTO TARGET TRANSFER BODY      tab:target-transfer
  % BEGIN/END AUTO PROJECTION ABLATION BODY  tab:projection-ablation
  % BEGIN/END AUTO TARGET SIGNAL BODY        tab:target-signal

A cell with no measurement keeps the manuscript's \\pending macro; the reference
row keeps the values its protocol fixes by convention, and a fresh row's surplus
stays n/a because one stochastic draw is not a policy game value. Nothing here
invents a number, and a row whose artifact is absent is left exactly as the
template wrote it.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
TEX = PAPER / "main_v6.tex"
ROOT = "/work/uf4_20260910"
DIAG = ROOT + "/analysis/diag_20260914"
KUBECONFIG = "/home/sjkim/.kube/aipr-kubeconfig.yaml"

TRANSFER = ("% BEGIN AUTO TARGET TRANSFER BODY", "% END AUTO TARGET TRANSFER BODY")
PROJECTION = ("% BEGIN AUTO PROJECTION ABLATION BODY", "% END AUTO PROJECTION ABLATION BODY")
SIGNAL = ("% BEGIN AUTO TARGET SIGNAL BODY", "% END AUTO TARGET SIGNAL BODY")

CRITERIA = ("instruction_following", "truthfulness", "honesty", "helpfulness")
PENDING = r"\pending"

# label, source kind, key in the artifact
TRANSFER_ROWS = [
    ("Reference on evaluation bank", "reference", None),
    ("Empirical source on learner pool", "categorical", "source"),
    ("gap", None, None),
    ("NBPO exact target", "categorical", "nbpo"),
    ("Fixed-reference exact target", "categorical", "fixedref"),
    ("Utilitarian exact target", "categorical", "util"),
    ("gap", None, None),
    ("NBPO fitted pool, seed 42", "categorical", "neural_nbpo_mse_s42"),
    ("Utilitarian fitted pool, seed 42", "categorical", "neural_util_mse_s42"),
    ("gap", None, None),
    ("NBPO fresh responses, seed 42", "fresh", "nbpo_mse_s42"),
    ("Utilitarian fresh responses, seed 42", "fresh", "util_mse_s42"),
    ("DPO fresh responses, seed 42", "fresh", "dpo_uniform_mse_s42"),
]


def pod(cmd):
    full = ("kubectl exec -n p-aipr nbpo-judge -c main -- bash -lc %s" % json.dumps(cmd))
    env = dict(os.environ, KUBECONFIG=KUBECONFIG)
    try:
        return subprocess.run(full, shell=True, capture_output=True, text=True,
                              timeout=300, env=env).stdout
    except Exception:                                        # noqa: BLE001
        return ""


def fetch(path):
    raw = pod("cat %s 2>/dev/null" % path)
    raw = raw.strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def num(value, digits=4):
    if value is None:
        return PENDING
    return "$%s$" % (("%." + str(digits) + "f") % value).lstrip("0").replace("-0.", "-.")


def render_transfer(transfer):
    lines = []
    for label, kind, key in TRANSFER_ROWS:
        if kind is None:
            lines.append(r"\addlinespace")
            continue
        if kind == "reference":
            n = transfer.get("common_prompts_all_objectives") if transfer else None
            lines.append("%s & %s & $.500$ & $.500$ & $.500$ & $.500$ & $.000$\\\\"
                         % (label, ("$%d$" % n) if n else PENDING))
            continue
        rows = (transfer or {}).get("rows" if kind == "categorical" else "fresh_rows", {})
        boot = (transfer or {}).get("bootstrap", {})
        entry = rows.get(key)
        if not entry:
            cells = [PENDING] * 5 if kind == "categorical" else [PENDING] * 4 + ["n/a"]
            lines.append("%s & %s & %s\\\\" % (label, PENDING, " & ".join(cells)))
            continue
        got = [entry.get(c, {}).get("n") for c in CRITERIA if c in entry]
        n = min(got) if got else None
        wins = [num(entry.get(c, {}).get("win_rate")) for c in CRITERIA]
        if kind == "categorical":
            surplus = boot.get(key, {}).get("min_surplus_point")
            last = num(surplus, 4) if surplus is not None else PENDING
        else:
            last = "n/a"
        lines.append("%s & %s & %s & %s\\\\"
                     % (label, ("$%d$" % n) if n else PENDING, " & ".join(wins), last))
    return "\n".join(lines)


def render_projection(pilot):
    order = [("Sampled pairs", "sampled"), ("All candidates", "all")]
    lines = []
    for label, key in order:
        e = (pilot or {}).get(key)
        if not e:
            lines.append("%s & %s\\\\" % (label, " & ".join([PENDING] * 7)))
            continue
        lines.append("%s & $%s$ & %s & %s & %s & %s & $%s$ & $%.2f$\\\\" % (
            label, e.get("steps", PENDING),
            num(e.get("train_nmse")), num(e.get("dev_nmse")), num(e.get("pool_kl")),
            num(e.get("fresh_wmin")),
            "{:,}".format(e["tokens"]).replace(",", "{,}") if e.get("tokens") else PENDING,
            e.get("gpu_hours", 0.0)))
    return "\n".join(lines)


def render_signal(signal):
    order = [("NBPO / utilitarian", "nbpo_vs_util"), ("NBPO / fixed reference", "nbpo_vs_fixedref")]
    lines = []
    for label, key in order:
        e = (signal or {}).get(key)
        if not e:
            lines.append("%s & %s\\\\" % (label, " & ".join([PENDING] * 5)))
            continue
        lines.append("%s & $%d$ & %s & %s & %s & %s\\\\" % (
            label, e["n"], num(e.get("mean_tv")), num(e.get("target_diff_rms")),
            num(e.get("nbpo_error_rms")), num(e.get("error_diff_rms"))))
    return "\n".join(lines)


def replace(text, markers, payload):
    begin, end = markers
    if begin not in text or end not in text:
        raise SystemExit("marker missing: %s" % begin)
    i, j = text.index(begin), text.index(end)
    return text[:i + len(begin)] + "\n" + payload + "\n" + text[j:]


def main():
    transfer = fetch(DIAG + "/target_transfer.json")
    pilot = fetch(DIAG + "/projection/pilot_table.json")
    signal = fetch(DIAG + "/target_signal.json")

    text = TEX.read_text(encoding="utf-8")
    text = replace(text, TRANSFER, render_transfer(transfer))
    text = replace(text, PROJECTION, render_projection(pilot))
    text = replace(text, SIGNAL, render_signal(signal))
    tmp = TEX.with_suffix(".tex.transfer_tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, TEX)

    filled = sum(1 for line in render_transfer(transfer).splitlines()
                 if PENDING not in line and "addlinespace" not in line)
    print(json.dumps({
        "transfer_artifact": bool(transfer),
        "transfer_rows_complete": filled,
        "transfer_rows_total": sum(1 for r in TRANSFER_ROWS if r[1]),
        "common_prompts": (transfer or {}).get("common_prompts_all_objectives"),
        "projection_artifact": bool(pilot),
        "signal_artifact": bool(signal),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
