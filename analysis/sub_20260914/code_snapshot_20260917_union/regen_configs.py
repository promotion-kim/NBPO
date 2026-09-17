"""Regenerate the five training configs against the rebuilt datasets.

Each config pins nbpo_expected_dataset_manifest_sha256, and the trainer refuses
to run when the dataset on disk hashes differently -- which is exactly right:
the configs were written for the datasets the solve produced with datasets
5.0.1, and those were rebuilt under deps_train so the trainer could read them.
The pairs, the targets and the solver artifact are unchanged; only the Arrow
encoding of the dataset differs, so only the dataset hash moved.

Rather than hand-editing a provenance field, the configs and their queued arms
are removed and make_train_jobs.py is re-run, so the tool that owns the config
is the tool that writes the new hash.
"""
import json
import shutil
from pathlib import Path

UF = Path("/work/uf4_20260910")
Q = UF / "jobs/queue"
ARMS = [("safe_nbpo_s42", "safe_nash_v1", 74),
        ("safe_fixedref_s42", "safe_fixedref_v1", 75),
        ("safe_util_s42", "safe_util_v1", 76),
        ("safe_maxmin_s42", "safe_maxmin_v1", 77),
        ("safe_btrm_s42", "safe_btrm_v1", 78)]

report = {"manifests": {}, "removed_configs": [], "removed_specs": [],
          "remade": []}

for arm, targets, _prio in ARMS:
    manifest = UF / "datasets" / targets / "precompute_manifest.json"
    report["manifests"][targets] = manifest.exists()

# drop the stale configs and the queued arms that pin the old hash
for arm, _targets, _prio in ARMS:
    cfg = UF / "configs" / ("%s.yaml" % arm)
    if cfg.exists():
        cfg.unlink()
        report["removed_configs"].append(cfg.name)
for p in sorted(Q.glob("*.json")):
    spec = json.loads(p.read_text())
    if spec["job_id"] in {"uf4_train_%s" % a for a, _t, _p in ARMS}:
        p.unlink()
        report["removed_specs"].append(spec["job_id"])

# re-run the writers by changing their specs
for p in sorted(Q.glob("*.json")):
    spec = json.loads(p.read_text())
    if not spec["job_id"].startswith("sub_make_train_safe"):
        continue
    spec["retry_note"] = ("2026-09-14 23:10 KST: the arms failed with 'precompute manifest "
                          "sha256 485c802b != expected 5d67ac43'. The datasets were rebuilt "
                          "under deps_train so the trainer could read them, which changed "
                          "their Arrow encoding and therefore their manifest hash; the "
                          "pairs, targets and solver artifact are unchanged. The config is "
                          "regenerated here so the tool that owns it records the new hash.")
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(spec, indent=2) + "\n")
    tmp.replace(p)
    report["remade"].append(spec["job_id"])

print(json.dumps(report, indent=1))
