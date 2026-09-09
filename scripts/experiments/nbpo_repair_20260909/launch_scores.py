"""Start each GPU's frozen teacher scorer when its pool worker exits cleanly."""
import argparse
import concurrent.futures
import json
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
        raise ValueError("Unexpected task root")
    logs = args.root / "logs"
    if not args.run:
        log = (logs / "score_supervisor.log").open("x")
        command = [sys.executable, "-m", "scripts.experiments.nbpo_repair_20260909.launch_scores", "--root", str(args.root), "--run"]
        proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True, cwd=args.root / "code")
        write_json(logs / "score_supervisor.json", {"pid": proc.pid, "command": command, "launched_at": time.time()})
        print(f"Scoring supervisor pid={proc.pid}", flush=True)
        return

    def worker(gpu):
        sentinel = logs / f"pool_gpu{gpu}.exit.json"
        while not sentinel.exists():
            time.sleep(5)
        code = json.loads(sentinel.read_text())["returncode"]
        if code != 0:
            return {"gpu": gpu, "status": "generation_failed", "returncode": code}
        env = dict(os.environ)
        env.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false",
                   PYTHONPATH=f"{args.root}/code:/work/pylibs_eval", HF_HUB_OFFLINE="1", WANDB_MODE="disabled")
        command = [sys.executable, "-m", "scripts.experiments.nbpo_repair_20260909.score_pool",
                   "--root", str(args.root), "--shard", str(gpu), "--encoder", str(args.root / "assets/roberta-base")]
        with (logs / f"score_gpu{gpu}.log").open("x") as log:
            before = time.time()
            proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env=env)
            write_json(logs / f"score_gpu{gpu}.launch.json", {"gpu": gpu, "pid": proc.pid, "command": command, "launched_at": before})
            result = {"gpu": gpu, "returncode": proc.wait(), "seconds": time.time() - before}
        write_json(logs / f"score_gpu{gpu}.exit.json", result)
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(worker, range(4)))
    write_json(logs / "scores_exit.json", results)
    raise SystemExit(int(any(r["returncode"] != 0 for r in results)))


if __name__ == "__main__":
    main()
