"""Localize the v0.1-to-v2.0 change by the two fields that actually differ.

v2.0 is not a refresh of v0.1, and three things changed at once: a much stronger
baseline (gpt-4.1 instead of gpt-4-0314), 246 of 750 prompts in 25 non-English
languages where v0.1 is English only, and 250 creative_writing prompts where
v0.1 has none. The language and category of every v2.0 prompt are recorded in
the frozen panel, so the last two are measurable rather than speculation.

This repeats the same paired whole-prompt bootstrap that produced the headline
arm-minus-base differences, restricted to each subset. The estimator is
unchanged: an arm's per-prompt score is the mean over its presentation orders,
a prompt enters only if both arms parsed every scheduled verdict, and the
minimum-free difference is resampled by whole prompt.

A subset interval is wider than the full-set one simply because it has fewer
prompts; that is stated rather than hidden, and no subset is presented as a
significance test the full set failed.
"""
from __future__ import annotations

import collections, json
from pathlib import Path

import numpy as np

ROOT = Path("/work/sub_20260914/eval_pairwise")
PANEL = Path("/work/sub_20260914/panel/eval_arenahard_v2.jsonl")
ARMS = ["uw1_sppo", "uw1_inpo_soft", "uw1_pw_nbpo", "uw1_prosper",
        "uw1_pw_fixedref", "uw1_dpo_soft"]
BENCH = "arenahard_v2"
ORDERS = 2
BOOT = 2000
SEED = 20260917


def load(tag):
    """prompt_id -> mean score over orders, only when every order parsed."""
    got = collections.defaultdict(dict)
    for line in (ROOT / tag / "verdicts.jsonl").open():
        if not line.strip():
            continue
        r = json.loads(line)
        v = r.get("value_for_arm", r.get("value_for_a"))
        if v is None:
            continue
        got[r["prompt_id"]][r["order"]] = float(v)
    return {p: sum(d.values()) / len(d) for p, d in got.items() if len(d) == ORDERS}


def paired(a, b, ids, rng):
    x = np.array([a[p] for p in ids])
    y = np.array([b[p] for p in ids])
    diff = float(x.mean() - y.mean())
    n = len(ids)
    reps = np.empty(BOOT)
    for i in range(BOOT):
        idx = rng.integers(0, n, n)
        reps[i] = x[idx].mean() - y[idx].mean()
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return diff, float(lo), float(hi), n


panel = {}
for line in PANEL.open():
    if line.strip():
        r = json.loads(line)
        panel[r["prompt_id"]] = r

subsets = {
    "all": lambda r: True,
    "English": lambda r: r.get("language") == "English",
    "non-English": lambda r: r.get("language") != "English",
    "hard_prompt": lambda r: r.get("category") == "hard_prompt",
    "  coding": lambda r: r.get("subcategory") == "coding",
    "  math": lambda r: r.get("subcategory") == "math",
    "creative_writing": lambda r: r.get("category") == "creative_writing",
}

base = load("uw_e72_base_%s" % BENCH)
out = {}
print("Arena-Hard v2.0, arm minus base, paired whole-prompt bootstrap (%d replicates)"
      % BOOT)
header = "%-18s" % "arm" + "".join("%-22s" % s.strip() for s in subsets)
print(header)
for arm in ARMS:
    a = load("uw_e72_%s_%s" % (arm, BENCH))
    row, cells = {}, []
    for name, keep in subsets.items():
        ids = sorted(p for p in set(a) & set(base) if keep(panel[p]))
        if len(ids) < 30:
            cells.append("%-22s" % "n<30")
            continue
        rng = np.random.default_rng(SEED)
        d, lo, hi, n = paired(a, base, ids, rng)
        star = "" if lo <= 0 <= hi else "*"
        row[name.strip()] = {"difference": d, "ci": [lo, hi], "prompts": n,
                             "excludes_zero": bool(star)}
        cells.append("%-22s" % ("%+.4f%s" % (d, star)))
    out[arm] = row
    print("%-18s" % arm + "".join(cells))

counts = {name: sum(1 for r in panel.values() if keep(r))
          for name, keep in subsets.items()}
print("\nprompts per subset: " + ", ".join("%s %d" % (k.strip(), v)
                                           for k, v in counts.items()))
print("* marks an interval excluding zero. Subset intervals are wider because "
      "they hold fewer prompts.")
dst = Path("/work/sub_20260914/analysis/ahv2_paired/breakdown.json")
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(json.dumps({"bench": BENCH, "bootstrap": BOOT, "orders": ORDERS,
                           "subset_prompt_counts": counts, "arms": out},
                          indent=1) + "\n")
print("written", dst)
