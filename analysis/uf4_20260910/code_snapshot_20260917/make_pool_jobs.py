"""Queue the full UF-4 pool and its teacher scoring, one chain per shard.

The train pool continues from the profile's chunk boundary with the same
settings file and the same per-candidate seeds, so the 200 profile prompts are
reused rather than regenerated. Scoring depends on its own shard only, so a
shard starts being scored as soon as it is generated instead of waiting for all
four.
"""
import json
import pathlib

Q = pathlib.Path("/work/uf4_20260910/jobs/queue")
BASE_ENV = {"HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "OMP_NUM_THREADS": "4", "WANDB_MODE": "disabled"}


def spec(job_id, priority, shard, command, artifacts, depends=()):
    return {"job_id": job_id, "priority": priority, "gpus": 1,
            "cwd": "/work/uf4_20260910",
            "env": {**BASE_ENV, "PYTHONPATH": "/work/pylibs_eval:/work/uf4_20260910/code",
                    "CUDA_VISIBLE_DEVICES": str(shard),
                    "VLLM_CACHE_ROOT": "/work/uf4_20260910/logs/vllm_cache_gpu%d" % shard},
            "command": command, "depends_on": list(depends),
            "timeout_s": 36000, "artifacts": artifacts}


written = []
for shard in range(4):
    gen_train = spec(
        "uf4_pool_train_shard%d" % shard, 30 + shard, shard,
        ["python3", "/work/uf4_20260910/code/generate_uf4_pool.py",
         "--splits", "v1", "--split-names", "policy_train", "--out-name", "v1",
         "--shard", str(shard), "--shards", "4", "--prompts-per-chunk", "25"],
        ["/work/uf4_20260910/pools/v1/shard%d/complete_shard%d.json" % (shard, shard)])
    gen_dev = spec(
        "uf4_pool_dev_shard%d" % shard, 34 + shard, shard,
        ["python3", "/work/uf4_20260910/code/generate_uf4_pool.py",
         "--splits", "v1", "--split-names", "policy_dev", "--out-name", "dev_v1",
         "--shard", str(shard), "--shards", "4", "--prompts-per-chunk", "25"],
        ["/work/uf4_20260910/pools/dev_v1/shard%d/complete_shard%d.json" % (shard, shard)])
    score_train = spec(
        "uf4_score_train_shard%d" % shard, 40 + shard, shard,
        ["python3", "/work/uf4_20260910/code/score_uf4_pool.py",
         "--pool", "v1", "--out-name", "v1", "--shard", str(shard)],
        ["/work/uf4_20260910/scores/v1/shard%d/complete_shard%d.json" % (shard, shard)],
        depends=["uf4_pool_train_shard%d" % shard])
    score_dev = spec(
        "uf4_score_dev_shard%d" % shard, 44 + shard, shard,
        ["python3", "/work/uf4_20260910/code/score_uf4_pool.py",
         "--pool", "dev_v1", "--out-name", "dev_v1", "--shard", str(shard)],
        ["/work/uf4_20260910/scores/dev_v1/shard%d/complete_shard%d.json" % (shard, shard)],
        depends=["uf4_pool_dev_shard%d" % shard])
    for name, body in (("30_pool_train", gen_train), ("34_pool_dev", gen_dev),
                       ("40_score_train", score_train), ("44_score_dev", score_dev)):
        path = Q / ("%s_shard%d.json" % (name, shard))
        path.write_text(json.dumps(body, indent=2) + "\n")
        written.append(path.name)
print(json.dumps({"written": written}, indent=1))
