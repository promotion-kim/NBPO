"""Record why PROSPER seed 44 was stopped, next to its own run log."""
import json
import time
from pathlib import Path

run = Path("/work/uf4_20260910/jobs/runs/uf4_train_prosper_mse_s44")
run.mkdir(parents=True, exist_ok=True)
(run / "stopped_by_operator.json").write_text(json.dumps({
    "job_id": "uf4_train_prosper_mse_s44",
    "stopped_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "ran_for_minutes": 3,
    "reason": (
        "Dispatched automatically at 20:10Z when the diagnostics freed the cards. Measured "
        "PROSPER cost is 86 min of stepping plus five 15.5-min development evaluations, so its "
        "training would end 08:10 KST and its judging 08:50 KST, past the 08:45 KST manuscript "
        "freeze this shift is working to. It would have held all four cards through the endgame "
        "while contributing nothing to the 09:00 snapshot."),
    "recoverable_checkpoint": (
        "none: save_steps=1250 with save_only_model, and no export directory existed, so "
        "stopping at step ~30 discarded about three minutes of compute and no artifact"),
    "replacement_work": (
        "the optional base fresh-draw arm (200 responses, 6,400 verdicts) and one declared "
        "scalarized-DPO weight arm, both of which finish inside the deadline"),
    "how_to_resume": (
        "the controller records this as FAILED and will not repeat it, because a FAILED job "
        "re-runs only when its spec bytes change. To put it back in the queue, edit "
        "jobs/queue/70_train_prosper_mse_s44.json (any field, e.g. add a requeue note); the "
        "controller then drops the terminal record and schedules it from scratch."),
    "not_a_result": "no partial result was written and none should be reported",
}, indent=2) + "\n")
print("recorded the stop note")
