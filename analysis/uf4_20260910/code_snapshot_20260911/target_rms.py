"""Target magnitude per arm: is the win-rate ordering an effective-step-size effect?

The manuscript's own caveat says L1 coefficient matching does not match realized
policy KL. If the arms that win also carry the largest Eq. (26) targets, then
what is being compared is partly how far each target pushes the policy, not the
aggregation rule. This reads the realized target distribution straight off each
released dev pair file.
"""
import json, math
from pathlib import Path

ROOT = Path("/work/uf4_20260910/targets")
SETS = {"nash_v1": "NBPO", "util_l1matched_v1": "Game-utilitarian",
        "fixedref_nash_v1": "Fixed-reference Nash", "maxmin_l1m_v1": "Global game-maxmin",
        "btrm_nash_v1": "BT-RM--Nash"}
print("%-22s %8s %9s %9s %9s %9s" % ("arm", "rows", "mean|t|", "rms", "p90|t|", "max|t|"))
for name, label in SETS.items():
    path = ROOT / name / "pairs" / "dev.jsonl"
    if not path.exists():
        print("%-22s (no dev pairs)" % label); continue
    vals = []
    with path.open() as f:
        for line in f:
            if line.strip():
                vals.append(float(json.loads(line)["nbpo_logratio_target"]))
    n = len(vals)
    absv = sorted(abs(v) for v in vals)
    mean_abs = sum(absv) / n
    rms = math.sqrt(sum(v * v for v in vals) / n)
    p90 = absv[int(0.90 * n)]
    print("%-22s %8d %9.4f %9.4f %9.4f %9.4f" % (label, n, mean_abs, rms, p90, absv[-1]))
