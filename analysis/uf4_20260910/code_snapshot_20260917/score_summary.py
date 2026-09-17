import json, sys
from pathlib import Path
root = Path(sys.argv[1] if len(sys.argv) > 1 else "/work/uf4_20260910/scores/v1")
tot = {"prompts":0,"gpm":0,"bt":0,"sec":0.0}
skew = diag = 0.0
absmax = 0.0
for s in range(4):
    d = root / ("shard%d" % s)
    c = json.loads((d / ("complete_shard%d.json" % s)).read_text())
    t = c["totals"]
    print("shard%d: prompts %d gpm_seq %d bt_seq %d seconds %.0f (%.1f gpm_seq/s)"
          % (s, t["prompts"], t["gpm_sequences"], t["bt_sequences"], t["seconds"],
             t["gpm_sequences"]/t["seconds"]))
    for k in tot: tot[k] += t[k.replace("gpm","gpm_sequences").replace("bt","bt_sequences").replace("sec","seconds")] if k in ("gpm","bt","sec") else t[k]
    for m in sorted(d.glob("chunk*.manifest.json")):
        r = json.loads(m.read_text())
        skew = max(skew, r["A_ref_skew_residual"]); diag = max(diag, r["A_ref_diagonal_residual"])
        absmax = max(absmax, r["A_policy_abs_max"])
print()
print("TOTAL prompts %d  gpm sequences %d  bt sequences %d" % (tot["prompts"], tot["gpm"], tot["bt"]))
print("wall on 4 GPUs %.1f min   aggregate %.1f gpm seq/s" % (tot["sec"]/4/60, tot["gpm"]/(tot["sec"]/4)))
print("A_ref skew residual max %.1e   diagonal max %.1e   A_policy |max| %.4f" % (skew, diag, absmax))
c0 = json.loads((root / "shard0/complete_shard0.json").read_text())
print("length budget:", json.dumps(c0["length_budget"]))
print("gpm teacher:", json.dumps(c0["gpm_teacher"]))
