"""Point the solves at the complete-case split and re-dispatch them."""
import json
from pathlib import Path

Q = Path("/work/uf4_20260910/jobs/queue")
touched = []
for p in sorted(Q.glob("*.json")):
    spec = json.loads(p.read_text())
    if not spec["job_id"].startswith("sub_solve_"):
        continue
    cmd = list(spec["command"])
    for i, token in enumerate(cmd):
        if token == "safe_v1" and i > 0 and cmd[i - 1] == "--splits":
            cmd[i] = "safe_v1cc"
    spec["command"] = cmd
    spec["retry_note"] = ("2026-09-14 21:16 KST: second attempt. The first died on a "
                          "SyntaxError in the generated solver copies (now fixed), the "
                          "second because the solver requires scores for every split "
                          "prompt and the scorer excluded 44 train and 12 dev prompts "
                          "whose pairs never resolved in both orders. The split is now "
                          "the scored subset, 1,956 train and 488 dev, recorded in "
                          "splits/safe_v1cc/freeze.json. The test panel is unchanged.")
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(spec, indent=2) + "\n")
    tmp.replace(p)
    touched.append(spec["job_id"])
print(json.dumps({"re_dispatched": touched}, indent=1))
