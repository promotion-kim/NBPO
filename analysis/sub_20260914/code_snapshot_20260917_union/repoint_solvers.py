"""Point the queued solve jobs at the two-objective solver copies."""
import json
from pathlib import Path

Q = Path("/work/uf4_20260910/jobs/queue")
changed = []
for p in sorted(Q.glob("*.json")):
    spec = json.loads(p.read_text())
    if not spec["job_id"].startswith("sub_solve_"):
        continue
    cmd = list(spec["command"])
    new = []
    for token in cmd:
        if token.endswith("/solve_uf4_targets.py"):
            token = "/work/sub_20260914/code/solve_safe_targets.py"
        elif token.endswith("/solve_prosper_targets.py"):
            token = "/work/sub_20260914/code/solve_safe_prosper.py"
        new.append(token)
    if new == cmd:
        continue
    spec["command"] = new
    spec["cwd"] = "/work/sub_20260914/code"
    spec["env"] = dict(spec["env"],
                       PYTHONPATH="/work/pylibs_eval:/work/sub_20260914/code:"
                                  "/work/uf4_20260910/code:/work/nbpo_repair_20260909/code")
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(spec, indent=2) + "\n")
    tmp.replace(p)
    changed.append(spec["job_id"])
print(json.dumps({"repointed": changed}, indent=1))
