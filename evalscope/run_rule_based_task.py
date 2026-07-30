# run_minerva_task.py
import os
import argparse
from evalscope import TaskConfig, run_task
from evalscope.constants import EvalType, JudgeStrategy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--datasets", default="ifeval", help="Comma-separated EvalScope dataset names.")
    parser.add_argument("--eval-batch-size", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--generation-seed", type=int, default=42)
    args = parser.parse_args()
    datasets = [name.strip() for name in args.datasets.split(",") if name.strip()]

    task_cfg = TaskConfig(
        model=args.model_name,
        api_url=f"http://127.0.0.1:{args.port}/v1",
        api_key="EMPTY",
        eval_type=EvalType.SERVICE,
        datasets=datasets,
        eval_batch_size=args.eval_batch_size,
        generation_config={
            "temperature": args.temperature,
            "seed": args.generation_seed,
            "do_sample": False,
            "max_tokens": 2048,
        },
    )

    run_task(task_cfg=task_cfg)


if __name__ == "__main__":
    main()
