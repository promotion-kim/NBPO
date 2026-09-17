"""Record each arm's materialized dataset, then generate its training job.

The solves ran with --skip-dataset so complete.json carries dataset_path null
and dataset_manifest_sha256 null. Materialization happened afterwards, behind
the prompt-level filter, so those two fields are filled in here from what is
actually on disk: the dataset directory and the sha256 of its
precompute_manifest.json, which is exactly what the solver would have written
and what the trainer checks at startup.

The previous record is kept under _superseded rather than overwritten, the same
discipline used when the Safe arms' pinned dataset hashes had to be corrected.

Training uses PROSPER Table 7: batch 128 (4 per device x 8 accumulation x 4
GPUs), LR 3e-7, weight decay 1e-6, seed 555134. max_steps 191 is three passes
over the 8148 surviving pairs; one pass is 64 steps, and at LR 3e-7 a single
pass would likely leave the policy nearly unmoved.
"""
import hashlib, json, subprocess, sys
from pathlib import Path

T = Path("/work/uf4_20260910/targets")
D = Path("/work/uf4_20260910/datasets")
S = Path("/work/sub_20260914")
ARMS = (("pros4_pw_nbpo_c", "pros_pw_nbpo"),
        ("pros4_pw_fixedref_c", "pros_pw_fixedref"),
        ("pros4_prosper", "pros_prosper"))
MAX_STEPS = 191


def fh(p):
    h = hashlib.sha256()
    with open(p, "rb") as s:
        for c in iter(lambda: s.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


out = {}
for target, arm in ARMS:
    cpath = T / target / "complete.json"
    rec = json.loads(cpath.read_text())
    manifest = D / target / "precompute_manifest.json"
    if not manifest.exists():
        raise SystemExit("no precompute manifest for %s" % target)
    if rec.get("dataset_path") and rec.get("dataset_manifest_sha256"):
        print("already recorded:", target)
    else:
        rec.setdefault("_superseded", []).append(
            {"dataset_path": rec.get("dataset_path"),
             "dataset_manifest_sha256": rec.get("dataset_manifest_sha256"),
             "why": "solved with --skip-dataset; materialized afterwards behind the "
                    "prompt-level representability filter"})
        rec["dataset_path"] = str(D / target)
        rec["dataset_manifest_sha256"] = fh(manifest)
        cpath.write_text(json.dumps(rec, indent=2) + "\n")
    out[target] = rec["dataset_manifest_sha256"][:16]

print(json.dumps({"recorded_manifests": out}, indent=1))

for target, arm in ARMS:
    cmd = [sys.executable, str(S / "code/make_pros_train_jobs.py"),
           "--targets", target, "--arm", arm,
           "--seed", "555134", "--max-steps", str(MAX_STEPS),
           "--eval-steps", str(MAX_STEPS), "--priority", "30",
           "--job-prefix", "pros2",
           "--model", "/work/models/bases/Qwen2.5-7B-Instruct",
           "--model-revision", "a09a35458c702b33eeacc393d103063234e8bc28",
           "--learning-rate", "3.0e-07", "--weight-decay", "1.0e-06",
           "--per-device-batch", "4", "--grad-accum", "8"]
    r = subprocess.run(cmd, capture_output=True, text=True,
                       cwd=str(S / "code"),
                       env={"PYTHONPATH": "/work/pylibs_eval", "PATH": "/usr/bin:/bin"})
    if r.returncode != 0:
        print("FAILED %s:\n  %s" % (arm, r.stderr.strip().splitlines()[-4:]))
    else:
        print("queued", arm, "|", (r.stdout or "").strip()[-160:])
