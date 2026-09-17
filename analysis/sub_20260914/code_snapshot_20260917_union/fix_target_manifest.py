"""Point each solve record at the dataset that now exists, then rewrite the configs.

make_train_jobs.py reads nbpo_expected_dataset_manifest_sha256 from the solve's
targets/<name>/complete.json, so regenerating the configs alone kept pinning the
hash of the dataset the solve built with datasets 5.0.1 -- the one the trainer
cannot read and which has since been replaced.

The rebuilt dataset is the same pairs re-encoded by deps_train, so the record is
updated to name it, with the superseded hash kept alongside rather than
overwritten silently. Then fresh make_train job ids write the configs against
it, because a DONE job never re-runs here.
"""
import hashlib
import json
import time
from pathlib import Path

UF = Path("/work/uf4_20260910")
Q = UF / "jobs/queue"
CODE = "/work/uf4_20260910/code"
ENV = {"PYTHONPATH": "/work/pylibs_eval:/work/uf4_20260910/code",
       "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
       "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
       "OPENBLAS_NUM_THREADS": "1", "WANDB_MODE": "disabled"}
ARMS = [("safe_nbpo_s42", "safe_nash_v1", 74),
        ("safe_fixedref_s42", "safe_fixedref_v1", 75),
        ("safe_util_s42", "safe_util_v1", 76),
        ("safe_maxmin_s42", "safe_maxmin_v1", 77),
        ("safe_btrm_s42", "safe_btrm_v1", 78)]

updated = []
for _arm, targets, _prio in ARMS:
    record_path = UF / "targets" / targets / "complete.json"
    manifest_path = UF / "datasets" / targets / "precompute_manifest.json"
    record = json.loads(record_path.read_text())
    actual = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if record.get("dataset_manifest_sha256") == actual:
        continue
    record["dataset_manifest_sha256_superseded"] = record.get("dataset_manifest_sha256")
    record["dataset_manifest_sha256"] = actual
    record["dataset_rebuild_note"] = (
        "2026-09-14 23:15 KST: the dataset was rebuilt from these same pairs under "
        "/work/nbpo_repair_20260909/deps_train, because the copy written by the system "
        "datasets 5.0.1 typed columns as Json and List and the trainer's library has "
        "neither type. Only the Arrow encoding changed, so only this hash moved; the "
        "pairs, the targets and the solver artifact are untouched.")
    record["dataset_rebuilt_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    tmp = record_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=2) + "\n")
    tmp.replace(record_path)
    updated.append({"targets": targets, "now": actual[:16],
                    "was": record["dataset_manifest_sha256_superseded"][:16]})

existing = {json.loads(p.read_text())["job_id"] for p in Q.glob("*.json")}
queued = []
for arm, targets, priority in ARMS:
    job = "sub_make_train_%s_v3" % arm
    if job in existing:
        continue
    spec = {"job_id": job, "priority": 73, "gpus": 0, "depends_on": [],
            "cwd": CODE, "env": ENV, "timeout_s": 43200,
            "command": ["python3", CODE + "/make_train_jobs.py",
                        "--targets", targets, "--arm", arm, "--seed", "42",
                        "--max-steps", "1250", "--priority", str(priority),
                        "--eval-steps", "1250"],
            "artifacts": [str(UF / "provenance" / ("train_job_%s.json" % arm))]}
    path = Q / ("%d_%s.json" % (spec["priority"], job))
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(spec, indent=2) + "\n")
    tmp.replace(path)
    queued.append(job)

# the stale configs must go, or make_train_jobs may refuse to overwrite them
removed = []
for arm, _t, _p in ARMS:
    cfg = UF / "configs" / ("%s.yaml" % arm)
    if cfg.exists():
        cfg.unlink()
        removed.append(cfg.name)

print(json.dumps({"records_updated": updated, "configs_removed": removed,
                  "queued": queued}, indent=1))
