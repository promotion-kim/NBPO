import json, pathlib
Q = pathlib.Path("/work/uf4_20260910/jobs/queue")
R = "/work/uf4_20260910"
BASE = "/work/models/bases/Llama-3.1-8B-Instruct"
for i, task in enumerate(("mmlu", "hellaswag", "arc_challenge")):
    spec = {"job_id": "uf4_cap_base_%s" % task, "priority": 70 + i, "gpus": 1,
            "cwd": R,
            "env": {"PYTHONPATH": "/work/pylibs_lmeval:/work/pylibs_eval:" + R + "/code",
                    "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "0",
                    "HF_HOME": "/work/hf_cache",
                    "TOKENIZERS_PARALLELISM": "false",
                    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                    "OMP_NUM_THREADS": "4", "WANDB_MODE": "disabled"},
            "command": ["python3", R + "/code/run_lm_eval.py",
                        "--model-path", BASE, "--arm", "base", "--task", task],
            "timeout_s": 21600,
            "artifacts": [R + "/evaluation/capability/base/%s/results.json" % task]}
    (Q / ("70_cap_base_%s.json" % task)).write_text(json.dumps(spec, indent=2) + "\n")
    print("queued", spec["job_id"])
