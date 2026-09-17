#!/usr/bin/env python3
"""Queue the API-free IFEval/GSM8K scoring of the already-generated UF-4 responses.

The uf4_gen_bench_* jobs wrote responses/<label>/{ifeval,gsm8k}.jsonl but nothing
scored them, so tab:general-capability had no IFEval or GSM8K cell for a trained
arm. Scoring is pure CPU: these specs take zero GPUs so they run beside a
four-GPU training job instead of waiting behind it.

evaluate_responses.py refuses to overwrite an existing output directory, so a
rerun needs a new _v<N> suffix rather than a deletion.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPAIR_ROOT = "/work/nbpo_repair_20260909"
ENV = {
    "PYTHONPATH": f"{REPAIR_ROOT}/code:/work/pylibs_eval:/work/pylibs_ifeval:/work/nbpo_downstream_local_v1/code",
    "HF_HUB_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "OMP_NUM_THREADS": "4",
    "WANDB_MODE": "disabled",
}
METHODS = ("nbpo", "util")
SEEDS = (42, 43, 44)


def spec(method, seed, version, priority):
    label = f"uf4_{method}_mse_s{seed}"
    out = f"{REPAIR_ROOT}/evaluations/deterministic_{label}_v{version}"
    return {
        "job_id": f"uf4_det_{method}_mse_s{seed}" + (f"_v{version}" if version != 1 else ""),
        "priority": priority,
        "gpus": 0,
        "cwd": f"{REPAIR_ROOT}/code",
        "env": ENV,
        "depends_on": [f"uf4_gen_bench_{method}_mse_s{seed}"],
        "command": [
            "python3", "-m", "scripts.experiments.nbpo_repair_20260909.evaluate_responses",
            "--root", REPAIR_ROOT,
            "--mode", "deterministic",
            "--labels", label,
            "--base-label", "base",
            "--benchmarks", "ifeval", "gsm8k",
            "--out", out,
        ],
        "timeout_s": 7200,
        "artifacts": [f"{out}/evaluation_complete.json"],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, required=True, help="directory to write the specs into")
    ap.add_argument("--version", type=int, default=1)
    ap.add_argument("--priority", type=int, default=30)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for method in METHODS:
        for seed in SEEDS:
            s = spec(method, seed, args.version, args.priority)
            path = args.out_dir / f"{args.priority}_{s['job_id']}.json"
            path.write_text(json.dumps(s, indent=2) + "\n")
            print(path)


if __name__ == "__main__":
    main()
