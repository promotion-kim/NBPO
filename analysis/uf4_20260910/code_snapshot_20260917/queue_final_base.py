"""Queue the shared base responses on final_eval, which every method is judged against.

The independent judge compares each policy to ONE fixed base response per prompt,
so that cache is generated once here and shared by every method, seed and
objective. It is generated from the same original base under the same raw-policy
sampling as the training pool, so the reference the judge sees is a draw from the
same proposal the solver's occurrence measure assumes.

Sixteen draws per prompt are kept: index learner:0 is the single fixed reference
for the main comparison, and the remaining draws are what the supplementary
T=1 eight-sample evaluation needs, generated now rather than a second time.

This does not depend on the solve, so it fills the window where the CPU solve
would otherwise leave the GPUs idle.
"""
import json, pathlib
Q = pathlib.Path("/work/uf4_20260910/jobs/queue")
R = "/work/uf4_20260910"
for shard in range(4):
    spec = {"job_id": "uf4_pool_finaleval_base_shard%d" % shard,
            "priority": 45 + shard, "gpus": 1,
            "cwd": R,
            "env": {"PYTHONPATH": "/work/pylibs_eval:" + R + "/code",
                    "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
                    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                    "OMP_NUM_THREADS": "4", "WANDB_MODE": "disabled"},
            "command": ["python3", R + "/code/generate_uf4_pool.py",
                        "--splits", "v1", "--split-names", "final_eval",
                        "--out-name", "final_base_v1", "--shard", str(shard), "--shards", "4",
                        "--prompts-per-chunk", "25"],
            "timeout_s": 21600,
            "artifacts": [R + "/pools/final_base_v1/shard%d/complete_shard%d.json" % (shard, shard)]}
    (Q / ("45_finaleval_base_shard%d.json" % shard)).write_text(json.dumps(spec, indent=2) + "\n")
    print("queued", spec["job_id"])
