import json, pathlib
Q = pathlib.Path("/work/uf4_20260910/jobs/queue")
S = json.load(open("/work/uf4_20260910/jobs/state.json"))["jobs"]
rows = []
for p in sorted(Q.glob("*.json")):
    s = json.loads(p.read_text())
    if s["gpus"] == 0:
        continue
    st = S.get(s["job_id"], {}).get("state")
    if st == "DONE":
        continue
    rows.append((s.get("priority", 100), s["job_id"], s["gpus"], st))
for prio, jid, g, st in sorted(rows):
    print("  %3d  %-44s %dgpu  %s" % (prio, jid, g, st))
print("total non-DONE GPU jobs:", len(rows))
