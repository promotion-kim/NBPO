"""Score the fixed-reference arm too, so tab:target-signal has both error terms."""
import json
from pathlib import Path

p = Path("/work/uf4_20260910/jobs/queue/54_uf4_diag_pool_logratios.json")
spec = json.loads(p.read_text())
spec["command"] = ["python3", "/work/uf4_20260910/code/diag_neural_pools.py",
                   "--arms", "nbpo_mse_s42", "util_mse_s42", "fixedref_mse_s42",
                   "--targets", "nash_v1", "util_l1matched_v1", "fixedref_nash_v1"]
tmp = p.with_suffix(".json.tmp")
tmp.write_text(json.dumps(spec, indent=2) + "\n")
tmp.replace(p)
print(" ".join(spec["command"]))
for name in ("nash_v1", "util_l1matched_v1", "fixedref_nash_v1"):
    d = Path("/work/uf4_20260910/targets") / name / "dev/solver/pi_star.npz"
    print("%-22s dev solver present: %s" % (name, d.exists()))
