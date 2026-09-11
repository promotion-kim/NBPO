import json, datetime
ev = [json.loads(l) for l in open("/work/uf4_20260910/logs/queue_report.jsonl")]
start, dur = {}, []
for e in ev:
    if e["state"] == "RUNNING":
        start[e["job_id"]] = e["time"]
    elif e["state"] in ("DONE", "FAILED") and e["job_id"] in start:
        t0 = datetime.datetime.strptime(start[e["job_id"]], "%Y-%m-%dT%H:%M:%SZ")
        t1 = datetime.datetime.strptime(e["time"], "%Y-%m-%dT%H:%M:%SZ")
        dur.append((e["job_id"], e["state"], (t1 - t0).total_seconds() / 60))
import re
groups = {}
for n, st, m in dur:
    key = re.sub(r'(_s4\d|_shard\d|_mse|nbpo|util|base|fixedref|maxmin)', '', n).strip('_')
    groups.setdefault(key, []).append(m)
for k in sorted(groups):
    v = groups[k]
    print("%-38s n=%2d  min=%6.1f  med=%6.1f  max=%6.1f (minutes)" % (
        k, len(v), min(v), sorted(v)[len(v)//2], max(v)))
