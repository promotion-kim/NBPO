"""Paired arm-minus-base differences on Arena-Hard v0.1 and v2.0, side by side.

Marginal intervals overlap almost completely on both benchmarks, which says
nothing: the seven policies are judged on the same prompts against the same
static baseline, so the pairing is the whole point. This runs the campaign's
existing paired whole-prompt bootstrap for every arm against the base row, on
each benchmark, and puts the two columns next to each other.

The base row is the untrained Qwen2.5-7B-Instruct, so a negative difference
means the trained arm is worse than not training at all on that benchmark.
"""
import json, subprocess, sys
from pathlib import Path

CODE = Path("/work/sub_20260914/code")
OUT = Path("/work/sub_20260914/analysis/ahv2_paired")
OUT.mkdir(parents=True, exist_ok=True)
ARMS = ["uw1_inpo_soft", "uw1_sppo", "uw1_pw_fixedref", "uw1_pw_nbpo",
        "uw1_prosper", "uw1_dpo_soft"]
BENCHES = [("arenahard", "v0.1"), ("arenahard_v2", "v2.0")]

rows = {}
for bench, label in BENCHES:
    base_tag = "uw_e72_base_%s" % bench
    if not Path("/work/sub_20260914/eval_pairwise", base_tag, "complete.json").exists():
        print(json.dumps({"skip": bench, "why": "no base row yet"})); continue
    for arm in ARMS:
        tag = "uw_e72_%s_%s" % (arm, bench)
        if not Path("/work/sub_20260914/eval_pairwise", tag, "complete.json").exists():
            print(json.dumps({"skip": tag, "why": "not judged yet"})); continue
        dst = OUT / ("%s__vs_base__%s.json" % (arm, bench))
        if not dst.exists():
            r = subprocess.run(
                [sys.executable, str(CODE / "paired_eval_diff.py"),
                 "--a", tag, "--b", base_tag, "--bootstrap", "2000",
                 "--out", str(dst)],
                capture_output=True, text=True)
            if r.returncode != 0:
                print(json.dumps({"failed": tag, "err": r.stderr[-300:]}))
                continue
        d = json.loads(dst.read_text())
        rows.setdefault(arm, {})[label] = d

def fmt(d):
    if d is None:
        return "%-28s" % "-"
    lo, hi = d["paired_difference_ci95"]
    star = "" if lo <= 0.0 <= hi else "  *"
    return "%+.4f [%+.4f, %+.4f]%s" % (d["paired_difference"], lo, hi, star)

print()
print("Arena-Hard, arm minus base, paired whole-prompt bootstrap (2,000 replicates)")
print("%-18s %-30s %-30s" % ("arm", "v0.1", "v2.0"))
for arm in ARMS:
    got = rows.get(arm, {})
    print("%-18s %-30s %-30s" % (arm, fmt(got.get("v0.1")), fmt(got.get("v2.0"))))
print("\n* marks an interval that excludes zero. Base is the untrained "
      "Qwen2.5-7B-Instruct, so a negative difference means the arm is worse "
      "than not training.")
(OUT / "summary.json").write_text(json.dumps(
    {arm: {lab: {"difference": d["paired_difference"],
                 "ci": d["paired_difference_ci95"],
                 "common_prompts": d.get("n_common_prompts")}
           for lab, d in got.items()} for arm, got in rows.items()}, indent=1) + "\n")
