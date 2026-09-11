"""Fill the maxmin judge window with the three remaining mmlu runs.

When maxmin finishes, its judge takes one card for about 24 minutes while the
next 4-GPU training holds the other three. The three outstanding mmlu jobs are
25 minutes each on one card, so three of them fit that window almost exactly and
delay the training by about a minute. They were the jobs deliberately left
behind the trainings earlier, when the window was shorter than they are.
"""
import json, pathlib
Q = pathlib.Path("/work/uf4_20260910/jobs/queue")
S = json.load(open("/work/uf4_20260910/jobs/state.json"))["jobs"]
for path in sorted(Q.glob("*.json")):
    spec = json.loads(path.read_text())
    jid = spec["job_id"]
    if not (jid.startswith("uf4_cap_") and jid.endswith("mmlu")):
        continue
    state = S.get(jid, {}).get("state")
    if state in ("RUNNING", "DONE"):
        print("skip (%s): %s" % (state, jid)); continue
    if spec.get("priority") == 62:
        continue
    old = spec.get("priority")
    spec["priority"] = 62
    path.write_text(json.dumps(spec, indent=2) + "\n")
    print("  %-40s %s -> 62 (%s)" % (jid, old, state))
