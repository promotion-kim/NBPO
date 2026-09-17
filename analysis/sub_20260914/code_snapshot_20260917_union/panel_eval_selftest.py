"""Hand-computed check of panel_eval_matrix on a synthetic four-prompt case."""
import json, subprocess, sys, tempfile
from pathlib import Path

SUB = Path("/work/sub_20260914")
tag = "selftest_objwise"
root = SUB / "objectivewise" / tag
root.mkdir(parents=True, exist_ok=True)

# arm A: objective 0 values .9 .7 .5 .3 -> mean .60
#        objective 1 values .2 .4 .4 .6 -> mean .40  -> W_min .40
# arm B: objective 0 values .5 .5 .5 .5 -> mean .50
#        objective 1 values .8 .8 .8 .8 -> mean .80  -> W_min .50
# prompt p4 is dropped for arm B objective 1, so the common set is p1..p3 and
# the expected means change: A obj0 (.9+.7+.5)/3 = .70, obj1 (.2+.4+.4)/3 = .3333
#                           B obj0 .50, obj1 .80 -> W_min A .3333, B .50
vals = {("A", 0): [.9, .7, .5, .3], ("A", 1): [.2, .4, .4, .6],
        ("B", 0): [.5, .5, .5, .5], ("B", 1): [.8, .8, .8, .8]}
pids = ["p1", "p2", "p3", "p4"]
with (root / "verdicts.jsonl").open("w") as f:
    for (a, k), series in vals.items():
        for pid, v in zip(pids, series):
            if a == "B" and k == 1 and pid == "p4":
                f.write(json.dumps({"a": a, "b": "C", "prompt_id": pid, "item": k,
                                    "order": 0, "identical_text": False,
                                    "value_for_a": None, "status": "no_verdict"}) + "\n")
                continue
            f.write(json.dumps({"a": a, "b": "C", "prompt_id": pid, "item": k,
                                "order": 0, "identical_text": False,
                                "value_for_a": v, "status": "ok"}) + "\n")
(root / "complete.json").write_text(json.dumps(
    {"bank": ["A", "B"], "comparator": "C", "items_per_prompt": 2,
     "verdicts_sha256": "synthetic"}, indent=1) + "\n")

out = subprocess.run([sys.executable, str(SUB / "code/panel_eval_matrix.py"),
                      "--tag", tag, "--bootstrap", "200"],
                     capture_output=True, text=True)
print(out.stdout[-900:], out.stderr[-600:])
rep = json.loads((root / "matrix.json").read_text())
t = rep["table"]
expect = {"A": {"w1": 0.7, "w2": 1.0 / 3.0, "wmin": 1.0 / 3.0},
          "B": {"w1": 0.5, "w2": 0.8, "wmin": 0.5}}
bad = []
for arm, cells in expect.items():
    for key, want in cells.items():
        got = t[arm][key]["value"]
        if abs(got - want) > 1e-12:
            bad.append((arm, key, got, want))
if rep["prompts"]["common_to_every_cell"] != 3:
    bad.append(("common", rep["prompts"]))
# W_min must be a real minimum, and A's interval must sit below .5
if t["A"]["wmin"]["ci"][1] >= 0.5:
    bad.append(("A wmin ci", t["A"]["wmin"]["ci"]))
if not t["B"]["wmin"]["contains_half"]:
    bad.append(("B wmin should contain .5", t["B"]["wmin"]["ci"]))
print(json.dumps({"failures": bad, "passed": not bad}, indent=1))
import shutil
shutil.rmtree(root)
raise SystemExit(1 if bad else 0)
