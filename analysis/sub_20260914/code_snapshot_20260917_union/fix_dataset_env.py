"""Materialize the training datasets with the trainer's own datasets library.

The solve wrote them with the system datasets 5.0.1, which types structured
columns as 'Json' and token columns as 'List'. The trainer imports datasets
from /work/nbpo_repair_20260909/deps_train, an older build whose feature
registry has neither name, so all five arms died on "Feature type 'Json' not
found" before a single step. The UF-4 datasets predate that upgrade, which is
why they load.

The pairs files are intact, so nothing is re-solved: prepare_nbpo_dataset is
re-run over the existing pairs with deps_train first on PYTHONPATH, so the
library that writes the dataset is the library that reads it. The wrongly-typed
dataset directories are removed first.

The solver copy is also patched so any later solve materializes under
deps_train rather than repeating this.
"""
import json
import shutil
from pathlib import Path

UF = Path("/work/uf4_20260910")
Q = UF / "jobs/queue"
DEPS = "/work/nbpo_repair_20260909/deps_train"
ARMS = ["safe_nash_v1", "safe_fixedref_v1", "safe_btrm_v1", "safe_util_v1",
        "safe_maxmin_v1"]
ARM_OF_JOB = {"nbpo": "safe_nash_v1", "fixedref": "safe_fixedref_v1",
              "btrm": "safe_btrm_v1", "util": "safe_util_v1",
              "maxmin": "safe_maxmin_v1"}
RETRY_NOTE = ("2026-09-14 22:40 KST: the first attempt died on a missing 'Json' feature "
              "type, because the dataset was written by datasets 5.0.1 while the trainer "
              "reads it with deps_train. The dataset is rebuilt under deps_train and this "
              "arm now waits for it.")

removed = []
for name in ARMS:
    d = UF / "datasets" / name
    if d.exists():
        shutil.rmtree(d)
        removed.append(name)

src_path = Path("/work/sub_20260914/code/solve_safe_targets.py")
src = src_path.read_text()
old = ('        env = dict(os.environ, HF_DATASETS_CACHE=str(out / "arrow_cache"),\n'
       '                   PYTHONDONTWRITEBYTECODE="1")')
new = ('        # The dataset must be written by the library that will read it: the\n'
       '        # trainer imports datasets from deps_train, whose feature registry has\n'
       '        # no Json or List type, and the system datasets 5.0.1 emits both.\n'
       '        env = dict(os.environ, HF_DATASETS_CACHE=str(out / "arrow_cache"),\n'
       '                   PYTHONDONTWRITEBYTECODE="1",\n'
       '                   PYTHONPATH="%s:%s" % (\n'
       '                       "/work/nbpo_repair_20260909/deps_train",\n'
       '                       os.environ.get("PYTHONPATH", "")))')
patched = old in src
if patched:
    src_path.write_text(src.replace(old, new, 1))

specs = []
for name in ARMS:
    specs.append({
        "job_id": "sub_materialize_%s" % name,
        "priority": 73, "gpus": 0, "depends_on": [],
        "cwd": "/work/nbpo_repair_20260909/code",
        "env": {"PYTHONPATH": DEPS + ":/work/nbpo_repair_20260909/code",
                "HF_DATASETS_CACHE": str(UF / "targets" / name / "arrow_cache2"),
                "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
                "PYTHONDONTWRITEBYTECODE": "1", "WANDB_MODE": "disabled"},
        "timeout_s": 43200,
        "command": ["python3", "-m", "mnpo_scripts.prepare_nbpo_dataset",
                    "--train", str(UF / "targets" / name / "pairs/train.jsonl"),
                    "--dev", str(UF / "targets" / name / "pairs/dev.jsonl"),
                    "--output", str(UF / "datasets" / name),
                    "--provenance", str(UF / "targets" / name / "dataset_provenance.json")],
        "artifacts": [str(UF / "datasets" / name / "dataset_dict.json")]})

existing = {json.loads(p.read_text())["job_id"] for p in Q.glob("*.json")}
queued = []
for spec in specs:
    if spec["job_id"] in existing:
        continue
    path = Q / ("%d_%s.json" % (spec["priority"], spec["job_id"]))
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(spec, indent=2) + "\n")
    tmp.replace(path)
    queued.append(spec["job_id"])

gated = []
for p in sorted(Q.glob("*.json")):
    spec = json.loads(p.read_text())
    job = spec["job_id"]
    changed = False
    if job == "sub_dpo_materialize":
        spec["env"] = dict(spec["env"],
                           PYTHONPATH=DEPS + ":/work/uf4_20260910/code:"
                                             "/work/nbpo_repair_20260909/code")
        spec["retry_note"] = (spec.get("retry_note", "") +
                              " Also switched to deps_train so the dataset is written by "
                              "the library the trainer reads it with.")
        changed = True
    if job.startswith("uf4_train_safe_"):
        key = job.split("_")[3]
        name = ARM_OF_JOB.get(key)
        if name:
            spec["depends_on"] = sorted(set(spec.get("depends_on") or [])
                                        | {"sub_materialize_%s" % name})
            spec["retry_note"] = RETRY_NOTE
            gated.append(job)
            changed = True
    if changed:
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(spec, indent=2) + "\n")
        tmp.replace(p)

print(json.dumps({"removed_datasets": removed, "solver_patched": patched,
                  "queued_materializations": queued, "arms_gated": gated}, indent=1))
