"""Make the UF weight arms wait for the campaign's training arms, not just rank below them.

Priority alone was not enough. The UF arms sit at 88-91 and the campaign's arms
at 74-80, but the controller does not preempt and only compares jobs that are
READY at the same moment. While the campaign's solve chain was being repaired it
had no GPU job registered at all, so a UF arm took all four cards twice in a
row -- once at 21:04 and again at 21:46.

A dependency fixes what a priority number cannot: each UF arm now waits until
the campaign's registered training arms are DONE, so leftover capacity is the
only capacity it can use.
"""
import json
from pathlib import Path

Q = Path("/work/uf4_20260910/jobs/queue")
specs = {}
for p in Q.glob("*.json"):
    s = json.loads(p.read_text())
    specs[s["job_id"]] = (p, s)

campaign = sorted(j for j in specs if j.startswith("uf4_train_safe"))
if not campaign:
    raise SystemExit("no campaign training arms registered yet; not gating anything")

changed = []
for job, (path, spec) in sorted(specs.items()):
    if not job.startswith("uf4_train_dpo_") or spec["priority"] < 85:
        continue
    deps = sorted(set(spec.get("depends_on") or []) | set(campaign))
    if deps == sorted(spec.get("depends_on") or []):
        continue
    spec["depends_on"] = deps
    spec["gate_note"] = ("2026-09-14 22:10 KST: waits for the submission campaign's "
                         "training arms. Priority 89-91 was not sufficient because the "
                         "controller only orders jobs that are READY simultaneously, and "
                         "this arm twice took all four cards while the campaign's arms were "
                         "not yet registered.")
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(spec, indent=2) + "\n")
    tmp.replace(path)
    changed.append((job, spec["priority"], len(deps)))

print(json.dumps({"campaign_arms_gating_on": campaign,
                  "gated": [{"job": j, "priority": p, "deps": d} for j, p, d in changed]},
                 indent=1))
