"""Queue the three cross-play pairs that do not depend on competitor selection.

Pairs are named with the two policies in sorted order so the id is deterministic
and the reverse cell is never judged twice. The three pairs involving the
development-selected competitor are queued once that selection is recorded.
"""
import itertools, json, pathlib
R = pathlib.Path("/work/uf4_20260910")
Q = R / "jobs/queue"
ENV = {"PYTHONPATH": "/work/pylibs_eval:/work/uf4_20260910/code",
       "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
       "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "OMP_NUM_THREADS": "4",
       "WANDB_MODE": "disabled"}
FIXED = ["base", "fixedref_mse_s42", "nbpo_mse_s42"]

for a, b in itertools.combinations(sorted(FIXED), 2):
    tag = "%s__vs__%s" % (a, b)
    spec = {"job_id": "uf4_crossplay_judge_%s" % tag, "priority": 64, "gpus": 1,
            "cwd": str(R), "env": ENV,
            "depends_on": ["uf4_crossplay_gen_%s" % a, "uf4_crossplay_gen_%s" % b],
            "command": ["python3", str(R / "code/judge_crossplay_pair.py"),
                        "--policy-a", a, "--policy-b", b,
                        "--pool-a", str(R / ("pools/crossplay_%s" % a)),
                        "--pool-b", str(R / ("pools/crossplay_%s" % b))],
            "timeout_s": 21600,
            "artifacts": [str(R / ("evaluation/crossplay/pairs/%s/complete.json" % tag))]}
    p = Q / ("%d_%s.json" % (spec["priority"], spec["job_id"]))
    payload = json.dumps(spec, indent=2) + "\n"
    if p.exists() and p.read_text() != payload:
        raise SystemExit("refusing to change existing spec: %s" % p)
    if p.exists():
        print("unchanged", spec["job_id"]); continue
    p.write_text(payload)
    print("queued %-56s deps %s" % (spec["job_id"], spec["depends_on"]))
