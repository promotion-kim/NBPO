"""Write overnight_diagnostics_manifest.json from the live queue, not from a plan.

One row per declared diagnostic unit with the fields the instruction asks for.
Status comes from the controller's own state file and the artifact on disk, so a
row cannot claim progress the cluster does not have. The existing method/weight/
seed execution table is untouched; this is a separate manifest.
"""
from __future__ import annotations

import glob
import hashlib
import json
import subprocess
import time
from pathlib import Path

ROOT = Path("/work/uf4_20260910")
DIAG = ROOT / "analysis/diag_20260914"
OUT = DIAG / "overnight_diagnostics_manifest.json"
JUDGE_REV = "40c069824f4251a91eefaf281ebe4c544efd3e18"


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as s:
        for c in iter(lambda: s.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def short(path):
    try:
        return file_hash(path)[:16]
    except OSError:
        return None


ROWS = [
    # experiment_id, exhibit_label, method, seed, job_id, artifact, gpu_hours
    ("bank_pilot10", "tab:target-transfer", "judge pilot", None,
     "uf4_diag_bank_pilot10", DIAG / "bank4/complete_first10.json", 0.2),
    ("bank_shard0", "tab:target-transfer", "categorical bank", None,
     "uf4_diag_bank_shard0", DIAG / "bank4/complete_shard0of4.json", 0.5),
    ("bank_shard1", "tab:target-transfer", "categorical bank", None,
     "uf4_diag_bank_shard1", DIAG / "bank4/complete_shard1of4.json", 0.5),
    ("bank_shard2", "tab:target-transfer", "categorical bank", None,
     "uf4_diag_bank_shard2", DIAG / "bank4/complete_shard2of4.json", 0.5),
    ("bank_shard3", "tab:target-transfer", "categorical bank", None,
     "uf4_diag_bank_shard3", DIAG / "bank4/complete_shard3of4.json", 0.5),
    ("fresh_gen_nbpo", "tab:target-transfer", "NBPO fresh", 42,
     "uf4_diag_fresh_gen_nbpo_mse_s42", DIAG / "fresh/complete_nbpo_mse_s42.json", 0.1),
    ("fresh_judge_nbpo", "tab:target-transfer", "NBPO fresh", 42,
     "uf4_diag_fresh_judge_nbpo_mse_s42", DIAG / "fresh_verdicts/complete_nbpo_mse_s42.json", 0.2),
    ("fresh_gen_util", "tab:target-transfer", "utilitarian fresh", 42,
     "uf4_diag_fresh_gen_util_mse_s42", DIAG / "fresh/complete_util_mse_s42.json", 0.1),
    ("fresh_judge_util", "tab:target-transfer", "utilitarian fresh", 42,
     "uf4_diag_fresh_judge_util_mse_s42", DIAG / "fresh_verdicts/complete_util_mse_s42.json", 0.2),
    ("fresh_gen_dpo", "tab:target-transfer", "uniform DPO fresh", 42,
     "uf4_diag_fresh_gen_dpo_uniform_mse_s42", DIAG / "fresh/complete_dpo_uniform_mse_s42.json", 0.1),
    ("fresh_judge_dpo", "tab:target-transfer", "uniform DPO fresh", 42,
     "uf4_diag_fresh_judge_dpo_uniform_mse_s42",
     DIAG / "fresh_verdicts/complete_dpo_uniform_mse_s42.json", 0.2),
    ("neural_pools", "tab:target-transfer", "fitted pool reweighting", 42,
     "uf4_diag_neural_pools", DIAG / "neural_pools.json", 0.3),
    ("projection_sampled", "tab:projection-ablation", "sampled-pair regression", 42,
     "uf4_diag_proj_sampled", ROOT / "arms/diag_proj_sampled_s42/config.json", 1.07),
    ("projection_all", "tab:projection-ablation", "all-candidate regression", 42,
     "uf4_diag_proj_all", ROOT / "arms/diag_proj_all_s42/config.json", 0.93),
    ("projection_fit", "tab:projection-ablation", "pilot fit and pool metrics", 42,
     "uf4_diag_pilot_fit", DIAG / "projection/pilot_table.json", 0.1),
    ("target_signal", "tab:target-signal", "target difference vs fit error", 42,
     None, DIAG / "target_signal.json", 0.0),
    ("conflict_report", "tab:target-signal", "cross-objective conflict", None,
     None, DIAG / "conflict_report.json", 0.0),
    ("aggregate", "tab:target-transfer", "bootstrap and table fill", None,
     None, DIAG / "target_transfer.json", 0.0),
]


def main():
    DIAG.mkdir(parents=True, exist_ok=True)
    state = json.loads((ROOT / "jobs/state.json").read_text())
    specs = set(state.get("queued_specs", []))
    panel = json.loads((DIAG / "panel_dev200.json").read_text())
    verdict_files = glob.glob(str(DIAG / "bank4/verdicts/*.done.json"))

    rows = []
    for exp, label, method, seed, job, artifact, gpu_h in ROWS:
        entry = state["jobs"].get(job, {}) if job else {}
        if artifact.exists():
            status = "DONE"
        elif job and entry.get("state") == "DONE":
            # the controller finished it; the row's declared artifact path is the
            # thing that is wrong, and saying NOT_STARTED here would be a lie
            status = "DONE_ARTIFACT_PATH_MISMATCH"
        elif job and entry.get("state") == "RUNNING":
            status = "RUNNING"
        elif job and entry.get("state") in ("READY", "PENDING"):
            status = entry["state"]
        elif job and entry.get("state") in ("FAILED", "BLOCKED"):
            status = entry["state"]
        elif job and job not in specs:
            status = "NOT_STARTED"
        elif job is None:
            status = "NOT_STARTED"
        else:
            status = "NOT_STARTED"
        progress = ""
        if exp.startswith("bank_"):
            progress = "%d/200 prompts judged" % len(verdict_files)
        rows.append({
            "experiment_id": exp,
            "exhibit_label": label,
            "prompt_ids_hash": panel["panel_sha256"][:16],
            "method": method,
            "seed": seed,
            "target_hash": None,
            "checkpoint_hash": None,
            "evaluator_revision": JUDGE_REV,
            "config_hash": short(ROOT / "code/diag_judge_bank.py") if "bank" in exp else None,
            "job_id": job,
            "status": status,
            "progress": progress,
            "artifact": str(artifact),
            "artifact_sha256": short(artifact),
            "ETA": None,
            "estimated_gpu_hours": gpu_h,
        })
    payload = {
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "panel": {"prompts": len(panel["panel_prompt_ids"]),
                  "sha256": panel["panel_sha256"],
                  "learner_occurrences": 8, "reference_occurrences": 4},
        "planned_verdicts": {"categorical": 60800, "fresh_per_arm": 6400,
                             "fresh_arms_queued": 3, "pilot_included_in_panel": True},
        "rows": rows,
    }
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(OUT)
    counts = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print(json.dumps({"written": str(OUT), "rows": len(rows), "status_counts": counts,
                      "prompts_judged": len(verdict_files)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
