"""Remove target directories left by aborted solves, and only those.

The solver creates targets/<name> with exist_ok=False so a rerun cannot silently
overwrite a finished target set. An attempt that died during pool loading still
left the directory behind, which then blocks the retry. Only directories with no
complete.json are removed -- a finished target set is never touched.
"""
import json, shutil
from pathlib import Path

T = Path("/work/uf4_20260910/targets")
removed, kept = [], []
for name in ("pros4_nbpo", "pros4_pw_nbpo", "pros4_pw_fixedref", "pros4_prosper"):
    d = T / name
    if not d.exists():
        continue
    if (d / "complete.json").exists():
        kept.append(str(d))
        continue
    contents = sorted(p.name for p in d.rglob("*"))[:8]
    shutil.rmtree(d)
    removed.append({"dir": str(d), "had": contents})
print(json.dumps({"removed_incomplete": removed, "kept_complete": kept}, indent=1))
