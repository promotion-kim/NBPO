import json, pathlib
Q = pathlib.Path("/work/uf4_20260910/jobs/queue")
for shard in range(4):
    spec = {
        "job_id": f"uf4_pool_profile_shard{shard}",
        "priority": 20 + shard,
        "gpus": 1,
        "cwd": "/work/uf4_20260910",
        "env": {"PYTHONPATH": "/work/pylibs_eval:/work/uf4_20260910/code",
                "CUDA_VISIBLE_DEVICES": str(shard),
                "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
                "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                "VLLM_CACHE_ROOT": f"/work/uf4_20260910/logs/vllm_cache_gpu{shard}",
                "OMP_NUM_THREADS": "4", "WANDB_MODE": "disabled"},
        "command": ["python3", "/work/uf4_20260910/code/generate_uf4_pool.py",
                    "--splits", "v1", "--split-names", "policy_train",
                    "--out-name", "v1", "--shard", str(shard), "--shards", "4",
                    "--prompts-per-chunk", "25", "--max-chunks", "2"],
        "timeout_s": 10800,
        "artifacts": [f"/work/uf4_20260910/pools/v1/shard{shard}/complete_shard{shard}.json"],
    }
    (Q / f"20_uf4_pool_profile_shard{shard}.json").write_text(json.dumps(spec, indent=2) + "\n")
    print("wrote", shard)
