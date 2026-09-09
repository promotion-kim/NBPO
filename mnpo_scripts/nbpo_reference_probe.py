"""Read-only model probe of real sampled events, before any optimizer update."""
import argparse
import gc
import json
import time
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from mnpo_scripts.mnpo_config import MNPOConfig
from mnpo_scripts.mnpo_trainer import MNPOTrainer
from mnpo_scripts.pair_tokenization import pair_from_candidate_events
from mnpo_scripts.nbpo_runtime import FP32RotaryBufferGuard


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--selection", choices=("first", "longest"), default="first")
    parser.add_argument("--deepspeed-cast-ablation", action="store_true",
                        help="Ablate DeepSpeed's module.bfloat16 buffer conversion without an optimizer")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    groups = {}
    with open(args.pool) as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("role") == "learner":
                groups.setdefault(row["prompt_id"], []).append(row)
    if args.selection == "first":
        events = next(iter(groups.values()))
    else:
        events = max(groups.values(), key=lambda rows: max(len(row["input_ids"]) for row in rows))
        events = sorted(events, key=lambda row: len(row["input_ids"]), reverse=True)
    if len(events) < 3:
        raise ValueError("Need three actual learner events from one prompt")
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    rows = []
    for a, b in ((0, 1), (0, 2)):
        rows.append({**pair_from_candidate_events(events[a]["prompt_token_ids"], events[a], events[b]),
                     "prompt": events[a].get("prompt", ""), "chosen": events[a]["response"],
                     "rejected": events[b]["response"]})
    started = time.perf_counter()
    policy = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                local_files_only=True, attn_implementation="sdpa", use_cache=False)
    cfg = MNPOConfig(output_dir=str(output), bf16=True, report_to="none", max_length=2048,
                     max_prompt_length=1024, logp_reduction="sum", loss_type="nbpo",
                     max_history_t=1, history_weights=[1.], gradient_checkpointing=True,
                     gradient_checkpointing_kwargs={"use_reentrant": False},
                     remove_unused_columns=False, per_device_train_batch_size=1)
    trainer = MNPOTrainer(model=policy, args=cfg, train_dataset=Dataset.from_list(rows), tokenizer=tok)
    policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    policy = trainer.accelerator.prepare_model(policy)
    reference = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                local_files_only=True, attn_implementation="sdpa", use_cache=False)
    reference = reference.to(trainer.accelerator.device).eval().requires_grad_(False)
    report = {"model": args.model, "pool": args.pool, "selection": args.selection,
              "selection_scope": "diagnostic-only; no training pair rows changed",
              "mode": "single_gpu_actual_trainer_collator_no_optimizer",
              "reference_frozen": not any(p.requires_grad for p in reference.parameters()),
              "parameter_dtype": str(next(policy.parameters()).dtype), "cases": [],
              "absolute_initialization_tolerance": 1e-4}
    raw_policy = trainer.accelerator.unwrap_model(policy)
    rotary_guard = FP32RotaryBufferGuard(raw_policy)
    rotary_snapshot = [(module, module.inv_freq.detach().clone()) for module in raw_policy.modules()
                       if hasattr(module, "inv_freq") and module.inv_freq.dtype == torch.float32]
    modes = ("baseline", "deepspeed_bfloat16_cast", "restored_fp32_rotary") if args.deepspeed_cast_ablation else ("baseline",)
    for mode in modes:
      if mode == "deepspeed_bfloat16_cast":
        raw_policy.bfloat16()
      elif mode == "restored_fp32_rotary":
        rotary_guard.restore_if_cast()
      for n_pairs in (1, 2):
        batch = trainer.data_collator([trainer.train_dataset[i] for i in range(n_pairs)])
        batch = trainer._prepare_inputs(batch)
        policy.train()
        pc, pr, _, _, _ = trainer.concatenated_forward(policy, batch)
        with torch.no_grad():
            rc, rr, _, _, _ = trainer.concatenated_forward(reference, batch)
            rc2, rr2, _, _, _ = trainer.concatenated_forward(reference, batch)
        difference = torch.cat((pc.float() - rc.float(), pr.float() - rr.float())).detach()
        h = ((pc.float() - rc.float()) - (pr.float() - rr.float())).detach()
        report["cases"].append({"mode": mode, "pairs": n_pairs, "sequence_max_abs": float(difference.abs().max()),
             "policy_rotary_buffer_dtypes": sorted({str(m.inv_freq.dtype) for m, _ in rotary_snapshot}),
             "sequence_rms": float(difference.square().mean().sqrt()), "pair_h_rms": float(h.square().mean().sqrt()),
             "reference_repeat_max_abs": float(torch.cat((rc - rc2, rr - rr2)).abs().max()),
             "chosen_response_tokens": batch["chosen_labels"].ne(-100).sum(-1).tolist(),
             "rejected_response_tokens": batch["rejected_labels"].ne(-100).sum(-1).tolist(),
             "sequence_logp_dtype": str(pc.dtype), "loss_dtype_if_squared": str(h.square().dtype)})
        del pc, pr, rc, rr, rc2, rr2, batch, difference, h
        gc.collect()
    report["logp_runtime_dtypes"] = trainer._logp_runtime_dtypes
    report["rotary_buffer_restorations"] = rotary_guard.restorations
    report["elapsed_seconds"] = time.perf_counter() - started
    report["peak_memory_bytes"] = torch.cuda.max_memory_allocated()
    report["passed"] = all(case["sequence_max_abs"] <= 1e-4 for case in report["cases"]
                           if case["mode"] != "deepspeed_bfloat16_cast")
    (output / "reference_initialization.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if not report["passed"]:
        raise ValueError("Real sampled-event initialization mismatch; inspect saved probe")


if __name__ == "__main__":
    main()
