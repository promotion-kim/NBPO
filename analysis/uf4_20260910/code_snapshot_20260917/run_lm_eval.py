"""Run one lm-evaluation-harness task under a fixed, declared recipe.

The recipe is the Open LLM Leaderboard v1 shot counts -- MMLU 5-shot,
ARC-Challenge 25-shot, HellaSwag 10-shot -- pinned here so every arm is measured
the same way and the shot count is never chosen per model. Harness version,
task revision, sample count and the exact metric keys are written next to the
numbers, because "MMLU 0.68" without those is not a comparable measurement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

FEWSHOT = {"mmlu": 5, "arc_challenge": 25, "hellaswag": 10}


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--task", required=True, choices=sorted(FEWSHOT))
    ap.add_argument("--out-root", default="/work/uf4_20260910/evaluation/capability")
    ap.add_argument("--batch-size", default="auto")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = ap.parse_args()

    out = Path(args.out_root) / args.arm / args.task
    if (out / "results.json").exists():
        print(json.dumps({"skipped": "already measured", "path": str(out)}), flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)

    import lm_eval
    from lm_eval.models.vllm_causallms import VLLM
    import transformers

    start = time.monotonic()
    model = VLLM(pretrained=args.model_path, dtype="bfloat16",
                 gpu_memory_utilization=args.gpu_memory_utilization,
                 max_model_len=args.max_model_len, tensor_parallel_size=1,
                 trust_remote_code=False, batch_size=args.batch_size)
    results = lm_eval.simple_evaluate(model=model, tasks=[args.task],
                                      num_fewshot=FEWSHOT[args.task],
                                      batch_size=args.batch_size, verbosity="WARNING")
    elapsed = time.monotonic() - start
    payload = {"arm": args.arm, "model_path": args.model_path, "task": args.task,
               "num_fewshot": FEWSHOT[args.task],
               "recipe": "Open LLM Leaderboard v1 shot counts, fixed for every arm",
               "harness": "lm-evaluation-harness %s" % lm_eval.__version__,
               "transformers": transformers.__version__,
               "backend": "vllm", "seconds": elapsed,
               "results": results["results"], "n_samples": results.get("n-samples"),
               "versions": results.get("versions"),
               "config": {k: v for k, v in results.get("config", {}).items()
                          if k in ("num_fewshot", "batch_size", "device", "limit",
                                   "bootstrap_iters", "gen_kwargs", "random_seed",
                                   "numpy_seed", "torch_seed", "fewshot_seed")}}
    (out / "results.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")
    metrics = {k: v for k, v in results["results"].get(args.task, {}).items()
               if isinstance(v, (int, float))}
    print(json.dumps({"arm": args.arm, "task": args.task,
                      "num_fewshot": FEWSHOT[args.task],
                      "metrics": metrics, "seconds": round(elapsed, 1),
                      "results_sha256": file_hash(out / "results.json")}), flush=True)


if __name__ == "__main__":
    main()
