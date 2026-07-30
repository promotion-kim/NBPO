from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer, Trainer
from trl import KTOConfig, KTOTrainer

_orig_get_train_sampler = KTOTrainer._get_train_sampler
if len(inspect.signature(_orig_get_train_sampler).parameters) == 1:
    def _get_train_sampler_compat(self, *args, **kwargs):
        return _orig_get_train_sampler(self)

    KTOTrainer._get_train_sampler = _get_train_sampler_compat

_orig_get_batch_samples = KTOTrainer.get_batch_samples
if len(inspect.signature(_orig_get_batch_samples).parameters) == 3:
    def _get_batch_samples_compat(self, *args, **kwargs):
        if len(args) >= 2 and hasattr(args[0], "generate") and isinstance(args[1], dict):
            return _orig_get_batch_samples(self, *args, **kwargs)
        return Trainer.get_batch_samples(self, *args, **kwargs)

    KTOTrainer.get_batch_samples = _get_batch_samples_compat

_orig_compute_loss = KTOTrainer.compute_loss
if "num_items_in_batch" not in inspect.signature(_orig_compute_loss).parameters:
    def _compute_loss_compat(self, model, inputs, return_outputs=False, *args, **kwargs):
        kwargs.pop("num_items_in_batch", None)
        return _orig_compute_loss(self, model, inputs, return_outputs=return_outputs)

    KTOTrainer.compute_loss = _compute_loss_compat

_orig_log = KTOTrainer.log
if len(inspect.signature(_orig_log).parameters) == 2:
    def _log_compat(self, logs, *args, **kwargs):
        return _orig_log(self, logs)

    KTOTrainer.log = _log_compat


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = load_dataset("json", data_files=args.train_file, split="train")
    eval_dataset = load_dataset("json", data_files=args.eval_file, split="train")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.eos_token

    config = KTOConfig(
        output_dir=str(output_dir),
        run_name=f"rev-smoke-kto-s{args.seed}",
        seed=args.seed,
        beta=args.beta,
        bf16=True,
        learning_rate=5.0e-7,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        optim="adamw_torch",
        weight_decay=0.0,
        max_steps=20,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=2048,
        max_prompt_length=1800,
        do_eval=True,
        eval_strategy="steps",
        eval_steps=10,
        logging_steps=1,
        log_level="info",
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
    )
    (output_dir / "config.json").write_text(config.to_json_string() + "\n", encoding="utf-8")

    trainer = KTOTrainer(
        model=args.model,
        ref_model=args.model,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    train_result = trainer.train()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    summary = {
        "method": "kto",
        "loss_type": "kto",
        "seed": args.seed,
        "backbone": args.model,
        "smoke": True,
        "status": "completed",
        "beta": args.beta,
        "train_examples": len(train_dataset),
        "eval_examples": len(eval_dataset),
        "output_dir": str(output_dir),
        "train_metrics_exists": (output_dir / "train_results.json").exists(),
        "trainer_state_exists": (output_dir / "trainer_state.json").exists(),
        "final_model_save_skipped": True,
    }
    (output_dir / "smoke_status.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("[smoke-status]", json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
