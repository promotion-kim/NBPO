"""Re-dispatch the DPO materialization, and retry PROSPER on a larger dual budget.

The materialization failed on a missing calibration key, which is now present.

PROSPER failed on its own certificate: independent stationarity and the
extra-map residual both exceeded tolerance with the probability floor active.
The tolerances are not touched -- loosening them would manufacture a result.
What is raised is the declared iteration budget, --max-dual-calls, from its
default to 4000, which is a budget and not an acceptance criterion. If the
certificate still fails, the PROSPER row stays unmeasured with this reason
recorded.
"""
import json
import shutil
from pathlib import Path

Q = Path("/work/uf4_20260910/jobs/queue")
notes = []
for p in sorted(Q.glob("*.json")):
    spec = json.loads(p.read_text())
    job = spec["job_id"]
    if job == "sub_dpo_materialize":
        spec["retry_note"] = ("2026-09-14 22:12 KST: failed on KeyError 'weights_order'. "
                              "The calibration file now carries weights_order and "
                              "tie_threshold_standardized, which the materializer copies "
                              "into the dataset provenance. The pair labels are unchanged.")
        notes.append(job)
    elif job == "sub_solve_prosper":
        cmd = list(spec["command"])
        if "--max-dual-calls" not in cmd:
            cmd += ["--max-dual-calls", "4000"]
        else:
            cmd[cmd.index("--max-dual-calls") + 1] = "4000"
        spec["command"] = cmd
        spec["retry_note"] = ("2026-09-14 22:12 KST: the finite-pool certificate failed "
                              "(independent stationarity and extra-map residual over "
                              "tolerance, probability floor active). Tolerances unchanged; "
                              "only the declared dual-call budget is raised to 4000. If it "
                              "fails again the row stays unmeasured.")
        notes.append(job)
    else:
        continue
    for stale in (Path("/work/uf4_20260910/targets") / ("safe_prosper_v1"),):
        if job == "sub_solve_prosper" and stale.is_dir() and not (stale / "complete.json").exists():
            shutil.rmtree(stale)
            notes.append("removed stale %s" % stale.name)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(spec, indent=2) + "\n")
    tmp.replace(p)
print(json.dumps({"actions": notes}, indent=1))
