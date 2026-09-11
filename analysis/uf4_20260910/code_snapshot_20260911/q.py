import json
from collections import Counter
s = json.load(open("/work/uf4_20260910/jobs/state.json"))["jobs"]
print("counts:", dict(Counter(e.get("state") for e in s.values())))
for st in ["RUNNING", "READY", "PENDING", "FAILED"]:
    names = [n for n, e in s.items() if e.get("state") == st]
    print("=== %s (%d) ===" % (st, len(names)))
    for n in names:
        e = s[n]
        print("  %-44s gpus=%s dev=%s prio=%s deps=%s" % (
            n, e.get("gpus"), e.get("devices"), e.get("priority"), e.get("deps")))
