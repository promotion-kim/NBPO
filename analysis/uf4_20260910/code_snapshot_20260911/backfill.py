"""Fill the judge window with short appendix jobs instead of leaving cards idle.

When a 4-GPU training is next in line the controller deliberately holds every
free card for it, so during a 24-minute single-GPU judge the other three sit
idle. Measured durations: arc_challenge 5 min, hellaswag 20 min, mmlu 25 min.
Promoting arc_challenge and hellaswag to just ahead of the training lets three
cards finish ~25 minutes of real appendix work inside a window that was already
blocked by the judge, so the training start slips by about a minute rather than
the queue wasting 72 GPU-minutes. mmlu stays behind the training: at 25 minutes
it is the one that could overrun the window and actually delay the main body.
"""
import json, pathlib
Q = pathlib.Path("/work/uf4_20260910/jobs/queue")
STATE = json.load(open("/work/uf4_20260910/jobs/state.json"))["jobs"]
BACKFILL_PRIORITY = 77            # judge is 76, the next 4-GPU training is 78
PROMOTE = ("arc_challenge", "hellaswag")

changed = []
for path in sorted(Q.glob("*.json")):
    spec = json.loads(path.read_text())
    jid = spec["job_id"]
    if not jid.startswith("uf4_cap_") or not jid.endswith(PROMOTE):
        continue
    if STATE.get(jid, {}).get("state") not in ("READY", "PENDING", None):
        print("skip (%s): %s" % (STATE.get(jid, {}).get("state"), jid)); continue
    if spec.get("priority") == BACKFILL_PRIORITY:
        print("already: %s" % jid); continue
    old = spec.get("priority")
    spec["priority"] = BACKFILL_PRIORITY
    path.write_text(json.dumps(spec, indent=2) + "\n")
    changed.append((jid, old, spec["gpus"]))
for jid, old, gpus in changed:
    print("%-44s %s -> %d  (gpus %s)" % (jid, old, BACKFILL_PRIORITY, gpus))
print("promoted %d jobs" % len(changed))
