"""Launch exactly four authorized independent pool workers; preserve all logs."""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from scripts.experiments.nbpo_repair_20260909.common import write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()
    if args.root != Path("/work/nbpo_repair_20260909"):
        raise ValueError("Unexpected campaign root")
    logs = args.root / "logs"
    logs.mkdir(exist_ok=True)
    if not args.run:
        log = (logs / "pool_supervisor.log").open("x")
        command = [sys.executable, "-m", "scripts.experiments.nbpo_repair_20260909.launch_pools", "--root", str(args.root), "--run"]
        p = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                             start_new_session=True, cwd=args.root / "code")
        write_json(logs / "pool_supervisor.json", {"pid": p.pid, "command": command, "launched_at": time.time()})
        print(f"Pool supervisor pid={p.pid}", flush=True)
        return
    procs = []
    for gpu in range(4):
        env = dict(os.environ)
        env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=f"{args.root}/code:/work/pylibs_eval",
                   OMP_NUM_THREADS="4", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                   TOKENIZERS_PARALLELISM="false", WANDB_MODE="disabled",
                   VLLM_CACHE_ROOT=f"{args.root}/cache/vllm{gpu}")
        cmd = [sys.executable, "-m", "scripts.experiments.nbpo_repair_20260909.generate_pool",
               "--root", str(args.root), "--shard", str(gpu)]
        log = (logs / f"pool_gpu{gpu}.log").open("x")
        p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env=env)
        procs.append((gpu, p))
        write_json(logs / f"pool_gpu{gpu}.launch.json", {"gpu": gpu, "pid": p.pid, "command": cmd, "launched_at": time.time()})
    results = {}
    for gpu, proc in procs:
        rc = proc.wait()
        results[str(gpu)] = rc
        write_json(logs / f"pool_gpu{gpu}.exit.json", {"returncode": rc, "finished_at": time.time()})
    write_json(logs / "pools_exit.json", results)
    raise SystemExit(int(any(results.values())))


if __name__ == "__main__":
    main()
