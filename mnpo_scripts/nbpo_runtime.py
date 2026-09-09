"""Measured ZeRO-2 precision, update and first-20-step throughput diagnostics."""
from __future__ import annotations

import json
import math
import os
import time
import torch
from transformers import TrainerCallback


class FP32RotaryBufferGuard:
    """Preserve exact FP32 RoPE frequencies across ZeRO's module.bfloat16().

    Parameters remain BF16. Merely casting a rounded buffer back to FP32 would
    not recover the pretrained frequencies; retain their original exact values
    before the distributed engine performs its whole-module conversion.
    """
    def __init__(self, model):
        self.snapshots = []
        self.restorations = []
        for name, module in model.named_modules():
            frequency = getattr(module, "inv_freq", None)
            if isinstance(frequency, torch.Tensor) and frequency.dtype == torch.float32:
                self.snapshots.append((name, module, frequency.detach().cpu().clone()))

    def restore_if_cast(self):
        for name, module, original in self.snapshots:
            current = module.inv_freq
            if current.dtype == torch.float32:
                continue
            if current.shape != original.shape:
                raise ValueError(f"RoPE shape changed before FP32 restoration: {name}")
            exact = original.to(device=current.device)
            self.restorations.append({
                "module": name, "before_dtype": str(current.dtype),
                "after_dtype": "torch.float32",
                "rounding_max_abs": float((current.float() - exact).abs().max()),
            })
            module.register_buffer("inv_freq", exact.clone(), persistent=False)
            if hasattr(module, "original_inv_freq"):
                module.original_inv_freq = exact.clone()
        return self.restorations


class NBPOPrecisionCallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer
        self.previous = []
        self.trainer._nbpo_rotary_buffer_guard = FP32RotaryBufferGuard(trainer.model)

    def on_train_begin(self, args, state, control, **kwargs):
        guard = self.trainer._nbpo_rotary_buffer_guard
        guard.restore_if_cast()
        os.makedirs(args.output_dir, exist_ok=True)
        rank = self.trainer.accelerator.process_index
        with open(os.path.join(args.output_dir, f"rotary_buffer_precision_rank{rank}.json"), "w") as handle:
            json.dump({"original_fp32_buffers": len(guard.snapshots),
                       "restorations": guard.restorations,
                       "forward_parameter_dtype": str(next(self.trainer.model.parameters()).dtype)}, handle, indent=2)

    def _optimizer(self):
        engine = getattr(self.trainer, "model_wrapped", None)
        optimizer = getattr(engine, "optimizer", None)
        masters = getattr(optimizer, "single_partition_of_fp32_groups", None)
        if masters is None:
            raise ValueError("FP32 master contract requires observed ZeRO-2 master partitions")
        if not masters or any(p.dtype != torch.float32 for p in masters):
            raise ValueError("ZeRO master parameters are not FP32")
        if int(engine.zero_optimization_stage()) != 2:
            raise ValueError("This prospective NBPO runtime requires ZeRO stage 2")
        return engine, optimizer, masters

    def on_step_begin(self, args, state, control, **kwargs):
        engine, optimizer, masters = self._optimizer()
        if state.global_step < 20:
            torch.cuda.synchronize()
            self.started = time.perf_counter()
            self.previous = [p.detach().clone() for p in masters]
            torch.cuda.reset_peak_memory_stats()

    def on_step_end(self, args, state, control, **kwargs):
        engine, optimizer, masters = self._optimizer()
        base = optimizer.optimizer
        moment_dtypes = set()
        for item in base.state.values():
            for key in ("exp_avg", "exp_avg_sq"):
                if key in item:
                    moment_dtypes.add(str(item[key].dtype))
        if moment_dtypes != {"torch.float32"}:
            raise ValueError(f"Adam first/second moments must actually be FP32: {moment_dtypes}")
        grad_norm = engine.get_global_grad_norm()
        grad_norm = float(grad_norm) if grad_norm is not None else None
        record = {
            "step": state.global_step, "loss_dtype": getattr(self.trainer, "_loss_runtime_dtype", None),
            "master_dtypes": sorted({str(p.dtype) for p in masters}),
            "adam_state_dtypes": sorted(moment_dtypes),
            "forward_parameter_dtype": str(next(self.trainer.model.parameters()).dtype),
            "logp_dtypes": getattr(self.trainer, "_logp_runtime_dtypes", {}),
            "rotary_buffer_restorations": len(self.trainer._nbpo_rotary_buffer_guard.restorations),
            "preclip_grad_norm": grad_norm,
            "clipped": grad_norm > args.max_grad_norm if grad_norm is not None else None,
            "cumulative_forward_tokens_this_rank": getattr(self.trainer, "_nbpo_token_counts", {}),
        }
        if record["loss_dtype"] != "torch.float32":
            raise ValueError(f"Observed loss dtype does not satisfy FP32 contract: {record['loss_dtype']}")
        if self.previous:
            squared = torch.zeros((), dtype=torch.float64, device=masters[0].device)
            for current, previous in zip(masters, self.previous):
                a, b = current.detach().reshape(-1), previous.reshape(-1)
                for start in range(0, a.numel(), 1048576):
                    delta = a[start:start+1048576] - b[start:start+1048576]
                    squared += delta.double().square().sum()
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(squared)
            torch.cuda.synchronize()
            record.update({"master_update_l2": math.sqrt(float(squared)),
                           "seconds_per_update_including_instrumentation": time.perf_counter() - self.started,
                           "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                           "peak_reserved_bytes": torch.cuda.max_memory_reserved()})
            self.previous = []
        os.makedirs(args.output_dir, exist_ok=True)
        rank = self.trainer.accelerator.process_index
        with open(os.path.join(args.output_dir, f"runtime_rank{rank}.jsonl"), "a") as handle:
            handle.write(json.dumps(record) + "\n")
        if args.nbpo_profile_updates > 0 and state.global_step >= args.nbpo_profile_updates:
            control.should_training_stop = True
        return control
