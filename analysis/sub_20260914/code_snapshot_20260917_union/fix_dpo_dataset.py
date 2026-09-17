"""Clear the half-written DPO dataset and re-dispatch its materialization.

materialize_dpo_dataset.py saves the dataset before it writes the provenance
file, so the run that died on the missing 'weights_order' key left a complete
dataset directory with no provenance -- and that copy was written by the system
datasets 5.0.1, which the trainer cannot read. The next two attempts then
refused to overwrite it, which is the right instinct on the script's part and a
dead end here.

So the directory is removed and the job re-dispatched. Its spec now points
PYTHONPATH at deps_train, so the rebuild is written by the library the trainer
reads with, and the calibration file already carries weights_order and
tie_threshold_standardized.
"""
import json
import shutil
from pathlib import Path

UF = Path("/work/uf4_20260910")
Q = UF / "jobs/queue"

target = UF / "datasets/dpo_safe_uniform_v1"
removed = None
if target.exists():
    had_provenance = (target / "dpo_dataset_provenance.json").exists()
    files = sum(1 for _ in target.rglob("*"))
    shutil.rmtree(target)
    removed = {"path": str(target), "files": files, "had_provenance": had_provenance}

bumped = []
for p in sorted(Q.glob("*.json")):
    spec = json.loads(p.read_text())
    if spec["job_id"] != "sub_dpo_materialize":
        continue
    spec["retry_note"] = (
        "2026-09-14 23:40 KST, fourth attempt. The first run saved the dataset and then "
        "died writing provenance because the calibration file lacked weights_order; the "
        "next two refused to overwrite what it had left behind. The directory is cleared, "
        "the calibration now carries weights_order and tie_threshold_standardized, and "
        "PYTHONPATH starts at deps_train so the dataset is written by the library the "
        "trainer reads it with.")
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(spec, indent=2) + "\n")
    tmp.replace(p)
    bumped.append({"job": spec["job_id"], "pythonpath": spec["env"].get("PYTHONPATH")})

print(json.dumps({"removed": removed, "re_dispatched": bumped}, indent=1))
