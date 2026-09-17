"""Train the UF-4 four-objective preference teacher on ModernBERT-base.

Two matched heads share this file, this backbone, this annotation, this split and
this seed, so the only difference between them is the preference model itself:

  gpm  P_k(A>B|x) = sigmoid((f_k(x,A,B) - f_k(x,B,A)) / 2)   antisymmetric by construction
  bt   P_k(A>B|x) = sigmoid(r_k(x,A) - r_k(x,B))             scalar reward per response

Both spend exactly two encoder passes per pair, so neither is quietly given more
compute. The BT head is fit to the SAME released annotations, never to the GPM's
predictions.

Masked criteria contribute to neither the loss nor its denominator. Selection is
mean dev NLL across the four objectives with worst-objective NLL as the
tie-break, both computed before any policy is trained.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, DistributedSampler

OBJECTIVES = ("instruction_following", "truthfulness", "honesty", "helpfulness")
ROOT = Path("/work/uf4_20260910")


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


def log(payload):
    if is_main():
        print(json.dumps(payload), flush=True)


class PairDataset(Dataset):
    def __init__(self, path):
        self.rows = []
        with open(path) as stream:
            for line in stream:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class Collator:
    """Serialize a pair under a symmetric, pre-declared length budget.

    The instruction is capped first and both responses then share what is left
    equally, so neither side of a comparison is truncated harder than the other.
    Every drop is counted; nothing is silently shortened.
    """

    KEYS = ("pairs", "sequences", "oversize_sequences", "dropped_tokens",
            "instruction_truncated_sequences", "response_truncated_sequences",
            "max_serialized_tokens")

    def __init__(self, tokenizer, head, max_length, instruction_cap):
        self.tok = tokenizer
        self.head = head
        self.max_length = max_length
        self.instruction_cap = instruction_cap
        self.stats = dict.fromkeys(self.KEYS, 0)
        self.ids = {name: tokenizer(name, add_special_tokens=False)["input_ids"]
                    for name in ("\n\n### INSTRUCTION\n", "\n\n### RESPONSE A\n",
                                 "\n\n### RESPONSE B\n", "\n\n### RESPONSE\n")}

    def _fit(self, instruction, parts):
        """parts: list of (marker, token_ids); returns the budgeted id list."""
        overhead = len(self.ids["\n\n### INSTRUCTION\n"]) + sum(len(self.ids[m]) for m, _ in parts)
        special = self.tok.num_special_tokens_to_add(pair=False)
        budget = self.max_length - special - overhead
        instr = instruction
        dropped = 0
        if len(instr) > self.instruction_cap:
            dropped += len(instr) - self.instruction_cap
            instr = instr[:self.instruction_cap]
            self.stats["instruction_truncated_sequences"] += 1
        remaining = budget - len(instr)
        if remaining < len(parts):                       # pathological: shrink the instruction
            keep = max(1, budget - len(parts))
            dropped += len(instr) - keep
            instr = instr[:keep]
            remaining = budget - len(instr)
        share = remaining // len(parts)
        kept = []
        for marker, ids in parts:
            if len(ids) > share:
                dropped += len(ids) - share
                ids = ids[:share]
                self.stats["response_truncated_sequences"] += 1
            kept.append((marker, ids))
        out = list(self.ids["\n\n### INSTRUCTION\n"]) + list(instr)
        for marker, ids in kept:
            out += list(self.ids[marker]) + list(ids)
        if dropped:
            self.stats["oversize_sequences"] += 1
            self.stats["dropped_tokens"] += dropped
        return out

    def _encode(self, instruction_ids, parts):
        self.stats["sequences"] += 1
        ids = self._fit(instruction_ids, parts)
        self.stats["max_serialized_tokens"] = max(self.stats["max_serialized_tokens"], len(ids))
        return self.tok.build_inputs_with_special_tokens(ids)

    def __call__(self, batch):
        # A fresh counter per batch: this runs in a DataLoader worker process, so
        # anything kept on self would never reach the training loop.
        self.stats = dict.fromkeys(self.KEYS, 0)
        sequences, targets, masks = [], [], []
        for row in batch:
            self.stats["pairs"] += 1
            instruction = self.tok(row["instruction"], add_special_tokens=False)["input_ids"]
            a = self.tok(row["response_a"], add_special_tokens=False)["input_ids"]
            b = self.tok(row["response_b"], add_special_tokens=False)["input_ids"]
            if self.head == "gpm":
                sequences.append(self._encode(instruction, [("\n\n### RESPONSE A\n", a),
                                                            ("\n\n### RESPONSE B\n", b)]))
                sequences.append(self._encode(instruction, [("\n\n### RESPONSE A\n", b),
                                                            ("\n\n### RESPONSE B\n", a)]))
            else:
                sequences.append(self._encode(instruction, [("\n\n### RESPONSE\n", a)]))
                sequences.append(self._encode(instruction, [("\n\n### RESPONSE\n", b)]))
            targets.append([0.5 if row["target"][o] is None else float(row["target"][o])
                            for o in OBJECTIVES])
            masks.append([bool(row["mask"][o]) for o in OBJECTIVES])
        width = max(len(s) for s in sequences)
        pad = self.tok.pad_token_id
        input_ids = torch.full((len(sequences), width), pad, dtype=torch.long)
        attention = torch.zeros((len(sequences), width), dtype=torch.long)
        for i, seq in enumerate(sequences):
            input_ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
            attention[i, :len(seq)] = 1
        return {"input_ids": input_ids, "attention_mask": attention,
                "target": torch.tensor(targets, dtype=torch.float32),
                "mask": torch.tensor(masks, dtype=torch.bool),
                "audit": dict(self.stats)}


class FourHeadEncoder(nn.Module):
    def __init__(self, path, head):
        super().__init__()
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(path, local_files_only=True)
        hidden = self.encoder.config.hidden_size
        self.head = head
        self.score = nn.Linear(hidden, len(OBJECTIVES))
        nn.init.zeros_(self.score.bias)
        nn.init.normal_(self.score.weight, std=0.02)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0]              # ModernBERT [CLS]
        scores = self.score(pooled)                       # (2B, K)
        first, second = scores[0::2], scores[1::2]
        if self.head == "gpm":
            # f(x,A,B) and f(x,B,A): the antisymmetric half-difference
            return (first - second) / 2.0
        return first - second                             # r(A) - r(B)


def autocast(device):
    """bf16 on CUDA; a no-op elsewhere so the CPU smoke path is the same code."""
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def masked_metrics(logits, target, mask):
    """Per-objective sums of NLL, Brier, correct and counts; ties excluded from accuracy."""
    probability = torch.sigmoid(logits.float())
    eps = 1e-6
    p = probability.clamp(eps, 1 - eps)
    nll = -(target * torch.log(p) + (1 - target) * torch.log(1 - p))
    brier = (p - target) ** 2
    decided = mask & (target != 0.5)
    correct = ((p > 0.5) == (target > 0.5)) & decided
    return {"nll": (nll * mask).sum(0), "brier": (brier * mask).sum(0),
            "n": mask.sum(0), "correct": correct.sum(0), "n_decided": decided.sum(0),
            "n_tie": (mask & (target == 0.5)).sum(0)}


def reduce_metrics(parts):
    total = {k: torch.zeros(len(OBJECTIVES), dtype=torch.float64, device=parts[0]["nll"].device)
             for k in parts[0]}
    for part in parts:
        for key, value in part.items():
            total[key] += value.double()
    if dist.is_initialized():
        for value in total.values():
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return total


def merge_audit(into, batch):
    for key, value in batch["audit"].items():
        into[key] = (max(into.get(key, 0), value) if key == "max_serialized_tokens"
                     else into.get(key, 0) + value)


def evaluate(model, loader, device, audit=None):
    model.eval()
    parts = []
    with torch.no_grad():
        for batch in loader:
            if audit is not None:
                merge_audit(audit, batch)
            with autocast(device):
                logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            parts.append(masked_metrics(logits, batch["target"].to(device),
                                        batch["mask"].to(device)))
    total = reduce_metrics(parts)
    out = {}
    for k, name in enumerate(OBJECTIVES):
        n = float(total["n"][k])
        decided = float(total["n_decided"][k])
        out[name] = {"nll": float(total["nll"][k]) / max(n, 1.0),
                     "brier": float(total["brier"][k]) / max(n, 1.0),
                     "accuracy_excl_ties": (float(total["correct"][k]) / decided
                                            if decided else None),
                     "n_labeled": n, "n_decided": decided,
                     "tie_fraction": float(total["n_tie"][k]) / max(n, 1.0)}
    nlls = [out[name]["nll"] for name in OBJECTIVES]
    out["_mean_nll"] = float(np.mean(nlls))
    out["_worst_nll"] = float(np.max(nlls))
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--head", required=True, choices=("gpm", "bt"))
    ap.add_argument("--pairs", default="v1")
    ap.add_argument("--out-name", required=True)
    ap.add_argument("--backbone", default=str(ROOT / "assets/ModernBERT-base"))
    ap.add_argument("--backbone-revision", default="8949b909ec900327062f0ebf497f51aef5e6f0c8")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.06)
    ap.add_argument("--global-batch-pairs", type=int, default=32)
    ap.add_argument("--micro-batch-pairs", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--instruction-cap", type=int, default=2048)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke-steps", type=int,
                    help="Stop after this many optimizer steps. Smoke only: the run is "
                         "written under its own name and is never a selectable teacher.")
    args = ap.parse_args()
    if args.epochs > args.max_epochs:
        raise ValueError("epochs exceeds the pre-declared max_epochs budget")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    cuda = torch.cuda.is_available()
    if world > 1:
        dist.init_process_group("nccl" if cuda else "gloo")
        if cuda:
            torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank) if cuda else torch.device("cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out = ROOT / "teacher" / args.out_name
    if is_main():
        out.mkdir(parents=True, exist_ok=False)
    if world > 1:
        dist.barrier()

    from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
    tokenizer = AutoTokenizer.from_pretrained(args.backbone, local_files_only=True)
    pair_dir = ROOT / "data" / f"pm_pairs_{args.pairs}"
    train = PairDataset(pair_dir / "pm_train.jsonl")
    dev = PairDataset(pair_dir / "pm_dev.jsonl")
    train_collate = Collator(tokenizer, args.head, args.max_length, args.instruction_cap)
    dev_collate = Collator(tokenizer, args.head, args.max_length, args.instruction_cap)
    train_audit, dev_audit = {}, {}

    accum = args.global_batch_pairs // (args.micro_batch_pairs * world)
    if accum * args.micro_batch_pairs * world != args.global_batch_pairs:
        raise ValueError("global batch must equal micro_batch * world * accumulation")
    train_sampler = DistributedSampler(train, shuffle=True, seed=args.seed, drop_last=True) if world > 1 else None
    dev_sampler = DistributedSampler(dev, shuffle=False, drop_last=False) if world > 1 else None
    train_loader = DataLoader(train, batch_size=args.micro_batch_pairs, sampler=train_sampler,
                              shuffle=(train_sampler is None), collate_fn=train_collate,
                              num_workers=4, drop_last=True, pin_memory=True)
    dev_loader = DataLoader(dev, batch_size=args.micro_batch_pairs, sampler=dev_sampler,
                            shuffle=False, collate_fn=dev_collate, num_workers=4, pin_memory=True)

    model = FourHeadEncoder(args.backbone, args.head).to(device)
    if world > 1:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank] if cuda else None)
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        (no_decay if name.endswith("bias") or "norm" in name.lower() else decay).append(parameter)
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                                   {"params": no_decay, "weight_decay": 0.0}],
                                  lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    steps_per_epoch = len(train_loader) // accum
    total_steps = int(steps_per_epoch * args.epochs)
    if args.smoke_steps:
        total_steps = min(total_steps, args.smoke_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(args.warmup_ratio * total_steps),
                                                total_steps)
    log({"phase": "setup", "head": args.head, "world": world, "train_pairs": len(train),
         "dev_pairs": len(dev), "accum": accum, "steps_per_epoch": steps_per_epoch,
         "total_steps": total_steps, "micro_batch_pairs": args.micro_batch_pairs})

    history, best = [], None
    start = time.monotonic()
    step = 0
    done = False
    for epoch in range(math.ceil(args.epochs)):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)
        for micro, batch in enumerate(train_loader):
            merge_audit(train_audit, batch)
            with autocast(device):
                logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            target = batch["target"].to(device)
            mask = batch["mask"].to(device)
            p = torch.sigmoid(logits.float()).clamp(1e-6, 1 - 1e-6)
            nll = -(target * torch.log(p) + (1 - target) * torch.log(1 - p))
            # One denominator: the number of labeled (pair, objective) cells.
            loss = (nll * mask).sum() / mask.sum().clamp(min=1)
            (loss / accum).backward()
            if (micro + 1) % accum:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % 20 == 0:
                log({"step": step, "loss": float(loss.detach()), "lr": scheduler.get_last_lr()[0],
                     "seconds": round(time.monotonic() - start, 1)})
            if step % args.eval_every == 0 or step == total_steps:
                metrics = evaluate(model, dev_loader, device,
                                   dev_audit if not dev_audit else None)
                record = {"step": step, "epoch": round(step / max(steps_per_epoch, 1), 3), **metrics}
                history.append(record)
                log({"phase": "dev", **{k: v for k, v in record.items() if k.startswith("_") or k == "step"}})
                key = (metrics["_mean_nll"], metrics["_worst_nll"])
                if best is None or key < (best["_mean_nll"], best["_worst_nll"]):
                    best = {**metrics, "step": step}
                    if is_main():
                        module = model.module if world > 1 else model
                        torch.save({"state_dict": module.state_dict(), "step": step,
                                    "head": args.head, "objectives": list(OBJECTIVES)},
                                   out / "best.pt")
                    if world > 1:
                        dist.barrier()
            if step >= total_steps:
                done = True
                break
        if done:
            break

    stats = {"train_collator": train_audit, "dev_collator": dev_audit}
    if world > 1:
        gathered = [None] * world
        dist.all_gather_object(gathered, stats)
    else:
        gathered = [stats]
    if is_main():
        merged = {}
        for part in gathered:
            for split, values in part.items():
                bucket = merged.setdefault(split, {})
                for key, value in values.items():
                    bucket[key] = (max(bucket.get(key, 0), value) if key == "max_serialized_tokens"
                                   else bucket.get(key, 0) + value)
        report = {"head": args.head, "backbone": args.backbone,
                  "backbone_revision": args.backbone_revision,
                  "pairs_dir": str(pair_dir),
                  "pairs_sha256": {split: file_hash(pair_dir / f"{split}.jsonl")
                                   for split in ("pm_train", "pm_dev")},
                  "args": vars(args), "world": world, "total_steps": total_steps,
                  "selection_rule": "lowest dev mean NLL over the four objectives; worst-objective NLL breaks ties",
                  "smoke_run": bool(args.smoke_steps),
                  "selected": best, "history": history,
                  "length_budget": merged,
                  "token_type_ids_used": False,
                  "seconds": time.monotonic() - start,
                  "source_sha256": file_hash(__file__)}
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        log({"phase": "done", "head": args.head, "best_step": best["step"],
             "mean_nll": best["_mean_nll"], "worst_nll": best["_worst_nll"],
             "seconds": round(report["seconds"], 1)})
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
