"""Fourth solve attempt: drop PROSPER's unsupported flags, clear stale dirs, re-dispatch."""
import json
import shutil
from pathlib import Path

Q = Path("/work/uf4_20260910/jobs/queue")
NOTE = ("2026-09-14 21:50 KST, fourth attempt. The solve itself succeeded on the third "
        "(train 1,956 prompts, dual [5.336, 4.699]); what failed was dataset "
        "materialization, because prepare_nbpo_dataset validates the canonical target "
        "against the row's own masses through a JSON parser that keeps ten decimals. The "
        "Safe Nash solution concentrates mass, so masses near 6e-5 shifted the target by "
        "1e-6 under that parse against a 1e-9 tolerance. The solver copy now rounds the "
        "masses to ten decimals and derives the target from the rounded values. PROSPER "
        "additionally reads its paths from module constants, so its copy names the Safe "
        "score, pool and split directories and its spec drops the flags it never "
        "accepted.")
KEEP_FOR_PROSPER = {"--out-name", "--workers", "--beta", "--eta", "--weight-l1-from"}

# stale, incomplete outputs
removed = []
for root in (Path("/work/uf4_20260910/targets"), Path("/work/uf4_20260910/datasets")):
    for d in sorted(root.glob("safe_*")):
        if d.is_dir() and not ((d / "complete.json").exists() or (d / "dataset_dict.json").exists()):
            shutil.rmtree(d)
            removed.append(str(d))

touched = []
for p in sorted(Q.glob("*.json")):
    spec = json.loads(p.read_text())
    if not spec["job_id"].startswith("sub_solve_"):
        continue
    if spec["job_id"] == "sub_solve_prosper":
        cmd, out = list(spec["command"]), []
        i = 0
        while i < len(cmd):
            token = cmd[i]
            if token.startswith("--"):
                takes_value = i + 1 < len(cmd) and not cmd[i + 1].startswith("--")
                if token in KEEP_FOR_PROSPER:
                    out.append(token)
                    if takes_value:
                        out.append(cmd[i + 1])
                i += 2 if takes_value else 1
                continue
            out.append(token)
            i += 1
        spec["command"] = out
    spec["retry_note"] = NOTE
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(spec, indent=2) + "\n")
    tmp.replace(p)
    touched.append(spec["job_id"])
print(json.dumps({"removed_incomplete": removed, "re_dispatched": touched}, indent=1))
