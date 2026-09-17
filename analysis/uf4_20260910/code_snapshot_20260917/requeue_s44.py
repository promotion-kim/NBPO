"""Put PROSPER seed 44 back in the queue now that the 08:45 freeze has passed.

It was stopped at 05:12 because its chain would have finished at 08:50, past
the manuscript freeze, while holding all four cards through the endgame. That
constraint is gone, and the seed is the last missing one in Table 1's PROSPER
row. Changing the spec bytes is what makes the controller drop the terminal
record and schedule it again; the note records why it is back rather than
leaving a silent requeue.
"""
import json
import time
from pathlib import Path

p = Path("/work/uf4_20260910/jobs/queue/70_train_prosper_mse_s44.json")
spec = json.loads(p.read_text())
spec["requeued"] = {
    "at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "why": ("the 08:45 KST manuscript freeze that made this job's 08:50 completion useless "
            "has passed, so its four-card occupancy no longer costs a deadline"),
    "previous_stop": "jobs/runs/uf4_train_prosper_mse_s44/stopped_by_operator.json",
    "starts_from": "scratch; no checkpoint existed before step 1250",
}
tmp = p.with_suffix(".json.tmp")
tmp.write_text(json.dumps(spec, indent=2) + "\n")
tmp.replace(p)
print("requeued with a recorded reason; controller will re-evaluate on its next poll")
