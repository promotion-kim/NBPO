from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path

import torch
from accelerate.hooks import remove_hook_from_module
from datasets import load_dataset
from transformers import AutoTokenizer, Trainer
from trl import KTOConfig, KTOTrainer


class CrossDeviceReference(torch.nn.Module):
    """Run the frozen reference on a second GPU and return only its logits."""

    def __init__(self, model: torch.nn.Module, target_device: str, return_device: torch.device):
        super().__init__()
        remove_hook_from_module(model, recurse=True)
        self.model = model.to(target_device).eval()
        self.target_device = torch.device(target_device)
        self.return_device = return_device
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def forward(self, *args, **kwargs):
        moved_args = tuple(value.to(self.target_device) if isinstance(value, torch.Tensor) else value for value in args)
        moved_kwargs = {
            key: value.to(self.target_device) if isinstance(value, torch.Tensor) else value
            for key, value in kwargs.items()
        }
        with torch.no_grad():
            output = self.model(*moved_args, **moved_kwargs)
        return type("ReferenceOutput", (), {"logits": output.logits.to(self.return_device)})()


def _patch_kto_trainer_compat() -> None:
    """Compatibility shims for the TRL/Transformers versions used on NHN."""

    orig_get_train_sampler = KTOTrainer._get_train_sampler
    if len(inspect.signature(orig_get_train_sampler).parameters) == 1:
        def _get_train_sampler_compat(self, *args, **kwargs):
            return orig_get_train_sampler(self)

        KTOTrainer._get_train_sampler = _get_train_sampler_compat

    orig_get_batch_samples = KTOTrainer.get_batch_samples
    if len(inspect.signature(orig_get_batch_samples).parameters) == 3:
        def _get_batch_samples_compat(self, *args, **kwargs):
            if len(args) >= 2 and hasattr(args[0], "generate") and isinstance(args[1], dict):
                return orig_get_batch_samples(self, *args, **kwargs)
            return Trainer.get_batch_samples(self, *args, **kwargs)

        KTOTrainer.get_batch_samples = _get_batch_samples_compat

    orig_compute_loss = KTOTrainer.compute_loss
    if "num_items_in_batch" not in inspect.signature(orig_compute_loss).parameters:
        def _compute_loss_compat(self, model, inputs, return_outputs=False, *args, **kwargs):
            kwargs.pop("num_items_in_batch", None)
            return orig_compute_loss(self, model, inputs, return_outputs=return_outputs)

        KTOTrainer.compute_loss = _compute_loss_compat

    orig_log = KTOTrainer.log
    if len(inspect.signature(orig_log).parameters) == 2:
        def _log_compat(self, logs, *args, **kwargs):
            return orig_log(self, logs)

        KTOTrainer.log = _log_compat


def _parse_report_to(value: str) -> list[str]:
    value = value.strip()
    if not value or value.lower() in {"none", "false", "0"}:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def main() -> None:
    _patch_kto_trainer_compat()
    # Use memory-efficient PyTorch SDPA but exclude the cuDNN SDPA backend,
    # which is the source of the known long-sequence mha_graph failure on this
    # image. Eager attention materializes a ~1-GiB softmax tensor and does not
    # fit KTO's required actual batch of two.
    if torch.cuda.is_available():
        torch.backends.cuda.enable_cudnn_sdp(False)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5.0e-7)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--save-total-limit", type=int, default=1)
    parser.add_argument("--report-to", default="wandb")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-final-save", action="store_true")
    parser.add_argument("--reference-device", default="")
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

    report_to = _parse_report_to(args.report_to)
    config_kwargs = {
        "output_dir": str(output_dir),
        "run_name": args.run_name or f"rev-q3-kto-b{str(args.beta).replace('.', 'p')}-s{args.seed}",
        "seed": args.seed,
        "beta": args.beta,
        "desirable_weight": 1.0,
        "undesirable_weight": 1.0,
        "bf16": True,
        "learning_rate": args.learning_rate,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": args.warmup_ratio,
        "optim": "adamw_torch",
        # KTO requires actual batch size > 1. On a single B200, AdamW's
        # multi-tensor foreach step adds enough temporary memory to exceed the
        # device by ~192 MiB at batch 2. The scalar-loop path is the same AdamW
        # update and removes that peak allocation.
        "optim_args": "foreach=False",
        "weight_decay": 0.0,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "precompute_ref_log_probs": False,
        "max_length": 2048,
        "max_prompt_length": 1800,
        "do_eval": not args.smoke,
        "eval_strategy": "no" if args.smoke else "steps",
        "eval_steps": args.eval_steps,
        "logging_steps": args.logging_steps,
        "log_level": "info",
        "save_strategy": "no" if args.smoke else "steps",
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "save_only_model": True,
        "save_safetensors": True,
        "report_to": report_to,
        "remove_unused_columns": False,
        "model_init_kwargs": {"attn_implementation": "sdpa", "torch_dtype": "bfloat16"},
    }
    if args.max_steps > 0:
        config_kwargs["max_steps"] = args.max_steps
    supported_config_keys = set(inspect.signature(KTOConfig).parameters)
    dropped_config_keys = sorted(k for k in config_kwargs if k not in supported_config_keys)
    config_kwargs = {k: v for k, v in config_kwargs.items() if k in supported_config_keys}
    if dropped_config_keys:
        (output_dir / "dropped_kto_config_keys.json").write_text(
            json.dumps(dropped_config_keys, indent=2) + "\n",
            encoding="utf-8",
        )
        print("[compat] dropped unsupported KTOConfig keys:", ",".join(dropped_config_keys))
    config = KTOConfig(**config_kwargs)
    (output_dir / "config.json").write_text(config.to_json_string() + "\n", encoding="utf-8")

    trainer_kwargs = {
        "model": args.model,
        "args": config,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
    }
    trainer_params = set(inspect.signature(KTOTrainer).parameters)
    if "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    trainer = KTOTrainer(**trainer_kwargs)
    if args.reference_device:
        if trainer.ref_model is None:
            raise RuntimeError("cross-device KTO requires an explicit frozen reference model")
        trainer.ref_model = CrossDeviceReference(
            trainer.ref_model, args.reference_device, trainer.accelerator.device
        )
        torch.cuda.empty_cache()
        print(
            f"[kto-memory] policy={trainer.accelerator.device} reference={args.reference_device}",
            flush=True,
        )
    # Transformers 5.13 serializes `optim_args=foreach=False` but its
    # get_optimizer_cls_and_kwargs path does not forward the flag for
    # adamw_torch. Force the same flag into the actual constructor kwargs and
    # fail closed if the resolved optimizer is no longer torch AdamW.
    optimizer_cls, optimizer_kwargs = trainer.get_optimizer_cls_and_kwargs(config, trainer.model)
    if not isinstance(optimizer_cls, type) or not issubclass(optimizer_cls, torch.optim.AdamW):
        raise RuntimeError(f"expected torch.optim.AdamW, resolved {optimizer_cls}")
    optimizer_kwargs = dict(optimizer_kwargs)
    optimizer_kwargs["foreach"] = False
    trainer.optimizer_cls_and_kwargs = (optimizer_cls, optimizer_kwargs)
    print("[kto-memory] forced actual torch.optim.AdamW foreach=False", flush=True)
    train_result = trainer.train()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    # Trainer.train() does not persist the final KTO weights when save_steps is
    # larger than the total number of optimizer steps.  Always materialize an
    # evaluation-ready model before the queue attempts its Hugging Face upload.
    if not args.skip_final_save:
        trainer.save_model(str(output_dir))

    status = {
        "method": "kto",
        "loss_type": "kto",
        "seed": args.seed,
        "backbone": args.model,
        "smoke": args.smoke,
        "skip_final_save": args.skip_final_save,
        "status": "completed",
        "beta": args.beta,
        "train_examples": len(train_dataset),
        "eval_examples": len(eval_dataset),
        "output_dir": str(output_dir),
        "run_name": config.run_name,
        "wandb_project": os.environ.get("WANDB_PROJECT"),
        "wandb_entity": os.environ.get("WANDB_ENTITY"),
        "train_metrics_exists": (output_dir / "train_results.json").exists(),
        "trainer_state_exists": (output_dir / "trainer_state.json").exists(),
    }
    (output_dir / ("smoke_status.json" if args.smoke else "run_status.json")).write_text(
        json.dumps(status, indent=2) + "\n",
        encoding="utf-8",
    )
    print("[run-status]", json.dumps(status, sort_keys=True))


if __name__ == "__main__":
    main()
