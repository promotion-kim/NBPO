"""Queue the scalarized-DPO train-pair realization. CPU only, so it runs under training."""
import json, pathlib
Q = pathlib.Path("/work/uf4_20260910/jobs/queue")
R = "/work/uf4_20260910"
spec = {
    "job_id": "uf4_build_dpo_pairs_train", "priority": 67, "gpus": 0, "depends_on": [],
    "cwd": R,
    "env": {"PYTHONPATH": R + "/code", "OMP_NUM_THREADS": "8",
            "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "8",
            "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false"},
    "command": ["python3", R + "/code/build_dpo_pairs.py", "--split", "train"],
    "timeout_s": 21600,
    "artifacts": [R + "/dpo/v1/pairs/build_report_train.json"],
}
path = Q / ("%d_%s.json" % (spec["priority"], spec["job_id"]))
payload = json.dumps(spec, indent=2) + "\n"
if path.exists() and path.read_text() != payload:
    raise SystemExit("refusing to change existing spec")
if path.exists():
    print("unchanged", spec["job_id"])
else:
    path.write_text(payload)
    print("queued", spec["job_id"])
