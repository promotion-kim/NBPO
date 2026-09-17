"""Mark the PROSPER solve terminal so a new failure is visible again.

The job has failed six times on its own certificate and the row is recorded as
unmeasured in additional_findings.md. Leaving a live spec in the queue keeps the
reporter printing it as the current blocker every cycle, which would hide the
next real failure behind a permanent one.

The spec is renamed rather than deleted, following the queue's existing
.superseded_label_conflict convention, so the decision and the command remain
on disk. The dependent make_train job is left BLOCKED, which is the honest
state: its input does not exist.
"""
import json
from pathlib import Path

Q = Path("/work/uf4_20260910/jobs/queue")
moved = []
for p in sorted(Q.glob("*.json")):
    spec = json.loads(p.read_text())
    if spec["job_id"] != "sub_solve_prosper":
        continue
    spec["terminal_note"] = (
        "2026-09-15 04:45 KST: closed as unmeasured after six attempts. The finite-pool "
        "certificate fails on this panel's tensors (independent stationarity over "
        "tolerance, probability floor active, inner optimizer without a certified "
        "solution). The declared dual budget was raised to 4000 and reproduced the same "
        "failure; tolerances were not relaxed, because a value obtained that way is not a "
        "certified solution. The five other representations certified on the same tensors, "
        "so this is specific to the PROSPER adaptation. Recorded in "
        "nbpo_iclr/additional_findings.md; tab:stress_results keeps \\pending for this row.")
    dest = p.with_suffix(".json.terminal_unmeasured")
    dest.write_text(json.dumps(spec, indent=2) + "\n")
    p.unlink()
    moved.append({"from": p.name, "to": dest.name})
print(json.dumps({"moved": moved}, indent=1))
