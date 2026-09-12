#!/usr/bin/env python3
r"""Write measured UF-4 numbers into Table 1, and say when the prose needs a human.

Only the marked regions are touched: the table body between AUTO TAB1 BODY and
the two count macros between AUTO EXHIBIT NUMBERS. Everything else in the
manuscript -- the professor's sentences, the review responses, the theory -- is
read-only here.

The numbers come from code/aggregate_common_final.py on the cluster, which
intersects the prompts every evaluated arm parsed in both orders and runs the
whole-prompt paired bootstrap with the minimum objective recomputed inside each
replicate. This script does no statistics of its own; it renders what that
aggregator measured.

Two things it deliberately does not do. It never invents a row: a method with no
finished evaluation stays \pending. And it never rewrites an interpretive
sentence -- "only helpfulness has an interval excluding 0.5" is a claim about
the current numbers, so when the set of attributes whose interval excludes 0.5
changes, the script reports that the caption is stale instead of writing new
prose. A wrong number is recoverable; a confidently wrong sentence is not.
"""
from __future__ import annotations

import json, os, subprocess, sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

KST = timezone(timedelta(hours=9))
PAPER = Path("/home/sjkim/MNPO/nbpo_iclr")
TEX = PAPER / "main_v6.tex"
PROG = PAPER / "progress"
ROOT = "/work/uf4_20260910"
KUBECONFIG = "/home/sjkim/.kube/aipr-kubeconfig.yaml"
POD = ["kubectl", "exec", "-n", "p-aipr", "nbpo-judge", "-c", "main", "--", "bash", "-lc"]

CRITERIA = ("instruction_following", "truthfulness", "honesty", "helpfulness")
# Display order and section headers exactly as the manuscript declares them.
LAYOUT = [
    ("row", "Base (defined reference)", None),
    ("gap", None, None),
    ("head", "Matched mechanism controls", None),
    ("row", "NBPO", "nbpo_mse"),
    ("row", "Fixed-reference Nash", "fixedref_mse"),
    ("row", "BT-RM--Nash", "btrm_mse"),
    ("row", "Game-utilitarian", "util_mse"),
    ("row", "Global game-maxmin", "maxmin_mse"),
    ("gap", None, None),
    ("head", "External baselines", None),
    ("row", "Scalarized DPO (uniform)", "dpo_uniform_mse"),
    ("row", "PROSPER (adapt.)", "prosper_mse"),
    ("row", "MOPO (adapt.)", "mopo_mse"),
]
BEGIN_BODY, END_BODY = "% BEGIN AUTO TAB1 BODY", "% END AUTO TAB1 BODY"
BEGIN_NUM, END_NUM = "% BEGIN AUTO EXHIBIT NUMBERS", "% END AUTO EXHIBIT NUMBERS"
BEGIN_CP, END_CP = "% BEGIN AUTO CROSSPLAY BODY", "% END AUTO CROSSPLAY BODY"
BEGIN_CPM, END_CPM = "% BEGIN AUTO CROSSPLAY MATRICES", "% END AUTO CROSSPLAY MATRICES"
# How the manuscript names each bank policy, and the seeds behind it.
CROSSPLAY_ROWS = [("Base", "base", "--"), ("NBPO", "nbpo_mse_s42", "$1$"),
                  ("Fixed-reference Nash", "fixedref_mse_s42", "$1$")]
RUBRIC_LABEL = {"instruction_following": "IF", "truthfulness": "Truth",
                "honesty": "Honesty", "helpfulness": "Help"}
# Declared floor for writing any cross-play cell into the manuscript.
MIN_CROSSPLAY_PROMPTS = 100


def pod(cmd, timeout=1800):
    env = dict(os.environ, KUBECONFIG=KUBECONFIG)
    r = subprocess.run(POD + [cmd], capture_output=True, text=True, timeout=timeout, env=env)
    return r.stdout if r.returncode == 0 else ""


def tex_escape(text):
    """Escape an arm name for LaTeX: these contain underscores (maxmin_mse_s42)."""
    out = []
    for ch in str(text):
        if ch in "#$%&_{}":
            out.append("\\" + ch)
        elif ch == "\\":
            out.append("\\textbackslash{}")
        elif ch in "~^":
            out.append("\\char`\\" + ch + "{}")
        else:
            out.append(ch)
    return "".join(out)


def crossplay_summary():
    """The aggregator's output, or None when no pair has been judged yet."""
    raw = pod("cat %s/analysis/crossplay_summary.json 2>/dev/null; true" % ROOT)
    if not raw.strip():
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def render_crossplay(cp):
    r"""Both cross-play exhibits. A policy the aggregator has no row for stays \pending."""
    stats, mats = cp["statistics"], cp["matrices"]
    n = cp["n_common_prompts"]
    # The floor is applied per statistic, not to the strictest global
    # intersection. Each statistic has its own declared prompt set -- the
    # reference comparison needs one pair, the bank statistics need more -- so
    # withholding the whole table because the narrowest set is thin would
    # discard usable measurements. A cell below the floor stays \pending; an
    # earlier version rendered an N=2 intersection as exactly 0.5000
    # everywhere, which reads as a clean "no difference" rather than as absent
    # data.
    def cell(s, key, fmt="$%.4f$"):
        if s is None or s.get(key) is None:
            return r"\pending"
        if s.get(key + "_n_prompts", 0) < MIN_CROSSPLAY_PROMPTS:
            return r"\pending"
        return fmt % s[key]
    competitor = next((p for p in cp["policies"]
                       if p not in {k for _, k, _ in CROSSPLAY_ROWS}), None)
    rows = list(CROSSPLAY_ROWS)
    rows.append(("Dev-selected competitor" + (" (%s)" % tex_escape(competitor) if competitor else ""),
                 competitor, "$1$" if competitor else "\\pending"))

    body = []
    for label, key, seeds in rows:
        s = stats.get(key) if key else None
        if s is None:
            body.append("%s & \\pending & \\pending & \\pending & \\pending & "
                        "\\pending\\\\" % label)
            continue
        # N is per statistic, so the column reports the reference comparison's
        # own count rather than one number that fits none of the three.
        ncell = s.get("W_ref_min_n_prompts")
        body.append("%s & %s & %s & %s & %s & %s\\\\"
                    % (label, seeds,
                       ("$%s$" % _tex_thousands(ncell)) if ncell else r"\pending",
                       cell(s, "W_ref_min"), cell(s, "W_bank_min"),
                       cell(s, "min_s_bank", "$%+.4f$")))

    order = [k for _, k, _ in CROSSPLAY_ROWS] + ([competitor] if competitor else [])
    mrows = []
    for ci, crit in enumerate(("instruction_following", "truthfulness",
                               "honesty", "helpfulness")):
        if ci:
            mrows.append("\\midrule")
        for ri, (label, key, _) in enumerate(rows):
            cells = []
            for q in order + [None] * (4 - len(order)):
                if key is None or q is None or key not in mats.get(crit, {}) \
                        or q not in mats[crit].get(key, {}):
                    cells.append("\\pending")
                else:
                    cells.append("$%.4f$" % mats[crit][key][q])
            name = RUBRIC_LABEL[crit] if ri == 0 else ""
            mrows.append("%s & %s & %s\\\\" % (name, label, " & ".join(cells)))
    return "\n".join(body), "\n".join(mrows)


def evaluated_arms():
    """Main-body arms only.

    The dev-selection runs write under evaluation/final_eval/devsel_<arm> so
    they sit beside these results, but they are judged on policy_dev, which is
    disjoint from the final 2,000. Including them makes the common-prompt
    intersection empty, so they must be excluded here rather than filtered
    downstream: the aggregator would otherwise be asked for a common set that
    cannot exist.
    """
    out = pod(f"ls -d {ROOT}/evaluation/final_eval/*/complete.json 2>/dev/null; true")
    names = sorted(Path(p).parent.name for p in out.split() if p.strip())
    return [n for n in names if not n.startswith("devsel_")]


def aggregate(arms):
    """Recompute the common set and the paired intervals over exactly these arms."""
    target = f"{ROOT}/analysis/final_eval_common_auto.json"
    env = ("PYTHONPATH=%s/code OMP_NUM_THREADS=8 HF_DATASETS_OFFLINE=1" % ROOT)
    cmd = (f"cd {ROOT} && {env} python3 code/aggregate_common_final.py "
           f"--arms {' '.join(arms)} --out {target} >/dev/null 2>&1; "
           f"cat {target} 2>/dev/null")
    raw = pod(cmd)
    if not raw.strip():
        return None
    return json.loads(raw)


def fmt(x):
    return "$%.4f$" % x


def excludes_half(ci):
    return ci[0] > 0.5 or ci[1] < 0.5


def render(report):
    """Build the table body, and collect which attributes clear 0.5 for every seed."""
    results, spread = report["results"], report.get("seed_spread", {})
    families = {}
    for arm in results:
        families.setdefault(arm.rsplit("_s", 1)[0], []).append(arm)

    lines, bold_attrs = [], {}
    for kind, label, family in LAYOUT:
        if kind == "gap":
            lines.append("\\addlinespace")
            continue
        if kind == "head":
            lines.append("\\multicolumn{7}{l}{\\textit{%s}}\\\\" % label)
            continue
        if family is None:                      # the defined reference
            lines.append("%s & -- & $0.500$ & $0.500$ & $0.500$ & $0.500$ & $0.500$\\\\" % label)
            continue
        members = sorted(families.get(family, []))
        if not members:
            lines.append("%s & \\pending & \\pending & \\pending & \\pending & \\pending & "
                         "\\pending\\\\" % label)
            continue
        # Column value is the mean over the evaluated seeds of this family.
        cells = []
        for c in CRITERIA:
            vals = [results[m][c]["win_rate"] for m in members]
            mean = sum(vals) / len(vals)
            # Bold marks an attribute whose paired interval excludes 0.5 for
            # EVERY evaluated seed of the family -- the caption's own wording.
            allsig = all(excludes_half(results[m][c]["ci95"]) for m in members)
            bold_attrs.setdefault(c, []).append((label, allsig))
            cells.append(("$\\mathbf{%.4f}$" % mean) if allsig else fmt(mean))
        mins = [results[m]["min_objective"]["point"] for m in members]
        min_mean = sum(mins) / len(mins)
        min_sig = all(excludes_half(results[m]["min_objective"]["ci95"]) for m in members)
        bold_attrs.setdefault("min_objective", []).append((label, min_sig))
        cells.append(("$\\mathbf{%.4f}$" % min_mean) if min_sig else fmt(min_mean))
        lines.append("%s & $%d$ & %s\\\\" % (label, len(members), " & ".join(cells)))
        if len(members) >= 2 and family in spread:
            sds = [spread[family][c]["sample_sd"] for c in CRITERIA]
            min_sd = _sd([results[m]["min_objective"]["point"] for m in members])
            lines.append("\\quad\\textit{sample SD over seeds} & & %s & $%.4f$\\\\"
                         % (" & ".join("$%.4f$" % s for s in sds), min_sd))
    return "\n".join(lines), bold_attrs


def _sd(vals):
    n = len(vals)
    if n < 2:
        return 0.0
    m = sum(vals) / n
    return (sum((v - m) ** 2 for v in vals) / (n - 1)) ** 0.5


def replace_region(text, begin, end, payload):
    i, j = text.index(begin), text.index(end)
    return text[:i + len(begin)] + "\n" + payload + "\n" + text[j:]


def main():
    arms = evaluated_arms()
    if not arms:
        print(json.dumps({"status": "no evaluated arms"})); return 0
    report = aggregate(arms)
    if report is None:
        print(json.dumps({"status": "aggregator produced nothing", "arms": arms}),
              file=sys.stderr)
        return 1
    body, bold = render(report)
    own = sorted(report["per_method_own_denominator"].values())
    common = report["common_prompts"]
    numbers = "\n".join([
        "% Factual counts written by progress/fill_exhibits.py. The caption prose reads",
        "% these macros, so the table body and the sentence describing it cannot drift",
        "% apart. Interpretive sentences are NOT auto-written.",
        "\\newcommand{\\tabonecommon}{%s}" % _tex_thousands(common),
        "\\newcommand{\\tabonevalidrange}{%s}" % (
            "$%s$" % _tex_thousands(own[0]) if own[0] == own[-1]
            else "$%s$--$%s$" % (_tex_thousands(own[0]), _tex_thousands(own[-1]))),
    ])

    cp = crossplay_summary()
    cp_body, cp_matrices = render_crossplay(cp) if cp else (None, None)
    cp_withheld = bool(cp) and cp_body is None

    # Re-read immediately before writing: the reporter and a human may both be
    # editing, and only these marked regions may change.
    text = TEX.read_text(encoding="utf-8")
    regions = [(BEGIN_BODY, END_BODY, body), (BEGIN_NUM, END_NUM, numbers)]
    if cp_body is not None:
        regions += [(BEGIN_CP, END_CP, cp_body), (BEGIN_CPM, END_CPM, cp_matrices)]
    for begin, end, payload in regions:
        if begin not in text or end not in text:
            print(json.dumps({"status": "markers missing", "marker": begin}), file=sys.stderr)
            return 1
        text = replace_region(text, begin, end, payload)
    tmp = TEX.with_suffix(".tex.fill_tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, TEX)

    sig = {c: [lab for lab, ok in v if ok] for c, v in bold.items()}
    out = {"written_kst": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S"),
           "arms": arms, "common_prompts": common,
           "crossplay_withheld_below_floor": cp_withheld,
           "crossplay_prompt_floor": MIN_CROSSPLAY_PROMPTS,
           "crossplay": ({"n_common_prompts": cp["n_common_prompts"],
                          "policies": cp["policies"],
                          "pairs_missing_for_full_bank": cp["pairs_missing_for_full_bank"],
                          "statistics": cp["statistics"]} if cp else "not measured yet"),
           "per_method_own_denominator": report["per_method_own_denominator"],
           "attributes_whose_interval_excludes_half": sig,
           "caption_prose_note": ("The caption's interpretive sentences are hand-written. "
                                  "Compare them against attributes_whose_interval_excludes_half "
                                  "above; if they disagree, the prose is stale."),
           "results": {a: {c: report["results"][a][c] for c in CRITERIA} for a in arms},
           "min_objective": {a: report["results"][a]["min_objective"] for a in arms},
           "seed_spread": report.get("seed_spread", {})}
    tmp = PROG / "exhibit_fill.json.tmp"
    tmp.write_text(json.dumps(out, indent=2) + "\n")
    os.replace(tmp, PROG / "exhibit_fill.json")
    print(json.dumps({"arms": len(arms), "common_prompts": common,
                      "significant": sig,
                      "crossplay_withheld_below_floor": cp_withheld,
                      "crossplay": (("%d policies, %d common prompts, %d pairs missing"
                                     % (len(cp["policies"]), cp["n_common_prompts"],
                                        len(cp["pairs_missing_for_full_bank"])))
                                    if cp else "not measured yet")},
                     ensure_ascii=False))
    return 0


def _tex_thousands(n):
    s = str(int(n))
    return s if len(s) <= 3 else "%s{,}%s" % (s[:-3], s[-3:])


if __name__ == "__main__":
    sys.exit(main())
