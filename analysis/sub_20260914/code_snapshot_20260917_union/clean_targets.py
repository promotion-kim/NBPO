"""Remove target directories left half-written by the failed solve attempts.

A directory is removed only when it has no complete.json, i.e. the solve that
created it never finished. Anything complete is left alone, and what was
removed is printed so the deletion is on the record.
"""
import json
import shutil
from pathlib import Path

T = Path("/work/uf4_20260910/targets")
D = Path("/work/uf4_20260910/datasets")
removed, kept = [], []
for root in (T, D):
    for d in sorted(root.glob("safe_*")):
        if not d.is_dir():
            continue
        complete = (d / "complete.json").exists() or (d / "dataset_dict.json").exists()
        if complete:
            kept.append(str(d))
            continue
        files = sum(1 for _ in d.rglob("*"))
        shutil.rmtree(d)
        removed.append({"path": str(d), "files_removed": files})
print(json.dumps({"removed_incomplete": removed, "kept_complete": kept}, indent=1))
