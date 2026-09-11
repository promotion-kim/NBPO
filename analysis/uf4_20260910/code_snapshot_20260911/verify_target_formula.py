"""Gate for the PROSPER build: can the canonical target be rebuilt from masses alone?

PROSPER needs per-prompt objective weights, which the shared artifact path cannot
carry, so its targets have to be assembled outside build_rows. That is safe only
if the target really is a closed form in the solved candidate masses. This checks
that claim on every row of a released target set rather than on one sampled row.

If any row disagrees, the planned PROSPER script cannot reuse the formula and the
design has to change instead of being trusted.
"""
import json, math, sys
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1
            else "/work/uf4_20260910/targets/nash_v1/pairs/dev.jsonl")
n = worst = 0
worst_row = None
bad = 0
for line in path.open():
    if not line.strip():
        continue
    r = json.loads(line)
    predicted = (math.log(r["nbpo_weight_a"] / r["nbpo_center_a"])
                 - math.log(r["nbpo_weight_b"] / r["nbpo_center_b"]))
    err = abs(predicted - r["nbpo_logratio_target"])
    n += 1
    if err > worst:
        worst, worst_row = err, r.get("prompt_id")
    if err > 1e-9:
        bad += 1
print(json.dumps({"file": str(path), "rows": n, "rows_disagreeing_beyond_1e-9": bad,
                  "max_abs_error": worst, "worst_prompt": worst_row,
                  "verdict": ("target is a closed form in the solved masses; PROSPER may "
                              "assemble it outside build_rows"
                              if bad == 0 else
                              "FORMULA DOES NOT HOLD; PROSPER cannot reuse it")}))
