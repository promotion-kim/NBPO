import os
import argparse

from evalscope import TaskConfig, run_task
from evalscope.constants import EvalType, JudgeStrategy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Served model name or API model id to evaluate.")
    parser.add_argument("--datasets", default="arena_hard", help="Comma-separated EvalScope dataset names.")
    parser.add_argument("--eval-batch-size", type=int, default=12)
    parser.add_argument("--judge-worker-num", type=int, default=12)
    parser.add_argument("--judge-model", default=os.getenv("JUDGE_MODEL", "gpt-5-mini"))
    args = parser.parse_args()

    api_key = os.getenv("API_KEY")
    api_url = os.getenv("API_URL")
    if not api_key or not api_url:
        raise RuntimeError("Set API_KEY and API_URL before running service-based LLM judge evaluation.")

    task_cfg = TaskConfig(
        model=args.model,
        generation_config={"max_tokens": 4096},
        api_url=api_url,
        api_key=api_key,
        eval_type=EvalType.SERVICE,
        datasets=[name.strip() for name in args.datasets.split(",") if name.strip()],
        eval_batch_size=args.eval_batch_size,
        judge_worker_num=args.judge_worker_num,
        judge_strategy=JudgeStrategy.AUTO,
        judge_model_args={
            "model_id": args.judge_model,
            "generation_config": {"reasoning_effort": "minimal"},
            "api_url": api_url,
            "api_key": api_key,
        },
    )

    run_task(task_cfg=task_cfg)


if __name__ == "__main__":
    main()
