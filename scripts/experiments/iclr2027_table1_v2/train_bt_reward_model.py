#!/usr/bin/env python3
"""Train one objective-specific scalar Bradley--Terry reward model.

This is the scalar-reward arm of the matched 2x2. It is fit to **exactly the
judged comparisons the game tensors are built from** -- the same judgment bank,
the same swap averaging, the same ties -- so that ``BT-RM-Nash vs NBPO`` isolates
the objective representation and nothing else.

Architecture: ``meta-llama/Llama-3.1-8B-Instruct`` with a scalar head on the
**final non-padding response token**. Full fine-tuning, bf16, AdamW. No LoRA and
no quantization: both would make the reward arm a different experiment than the
policy arm, and the protocol forbids substituting either silently.

Labels: for one semantic pair the two presentation orders are combined as

    q_AB = 1.0 / 0.5 / 0.0  for [[A]] / [[TIE]] / [[B]]
    p_hat(A > B) = 0.5 * (q_AB + 1 - q_BA)

and the loss is the soft cross-entropy of ``mnpo_scripts.bt_reward``. Ties are
kept. Dropping them is the standard shortcut and it is wrong here: the tied
comparisons are disproportionately the ones the four objectives disagree about,
which is the entire subject of the paper.

Selection is on **validation soft BT NLL**, read against
``soft_label_entropy`` -- the floor a soft-label NLL cannot go below. Hard
accuracy (ties excluded), Brier, ECE, the reward/length correlation and the
cyclic-vs-acyclic split are reported as diagnostics and never select.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from mnpo_scripts.bt_reward import (
    brier_score,
    expected_calibration_error,
    hard_accuracy_excluding_ties,
    pearson,
    soft_bt_loss,
    soft_label_entropy,
    swap_averaged_probability,
    verdict_to_q,
)


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_pairs(verdicts_path: Path, objective: str, responses: dict):
    """Collapse the judgment bank's ordered rows into swap-averaged pairs.

    A row is one (semantic pair, objective, presentation order) judgment;
    ``policy_win`` is already expressed from the learner's point of view, so the
    two orders combine as ``p_hat = 0.5 * (q_learner_first + q_comparator_first)``
    -- which is the same swap average ``bt_reward.swap_averaged_probability``
    performs, written in this bank's convention.

    A semantic pair judged in only one order is **dropped and counted**. Half a
    swap average is a position-biased label, not a cheap extra example.

    ``responses`` maps ``seed -> {prompt_id: row}`` for both pools, because the
    bank stores response *ids* rather than texts.
    """
    both = defaultdict(dict)
    meta = {}
    for line in verdicts_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("objective") != objective or not row.get("valid", False):
            continue
        key = (row["prompt_id"], row["learner_pool"],
               row["learner_response_id"], row["comparator_response_id"])
        both[key][row["presentation_order"]] = float(row["policy_win"])
        meta[key] = (row["learner_seed"], row["comparator_seed"])
    pairs, dropped, unresolved = [], 0, 0
    for key, orders in both.items():
        if "learner_first" not in orders or "comparator_first" not in orders:
            dropped += 1
            continue
        p_hat = 0.5 * (orders["learner_first"] + orders["comparator_first"])
        pid = key[0]
        lseed, cseed = meta[key]
        a = responses.get(lseed, {}).get(pid)
        b = responses.get(cseed, {}).get(pid)
        if a is None or b is None:
            unresolved += 1
            continue
        pairs.append({"prompt_id": pid, "prompt": a["prompt"],
                      "response_a": a["generated_text"],
                      "response_b": b["generated_text"], "p_hat": p_hat})
    if unresolved:
        raise SystemExit(
            f"{unresolved} judged pairs reference responses that are not in the supplied "
            "pools. The reward model would be fit to comparisons whose text is unknown; "
            "supply every pool the bank was judged from.")
    return pairs, dropped


def load_response_pools(specs):
    """``seed=path.json`` -> ``{seed: {prompt_id: row}}``."""
    out = {}
    for spec in specs:
        seed, path = spec.split("=", 1)
        data = json.loads(Path(path).read_text())
        rows = data if isinstance(data, list) else data.get("responses", data)
        out[seed] = {r["prompt_id"]: r for r in rows}
    return out


class PairDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length):
        self.pairs = pairs
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def _encode(self, prompt, response):
        msgs = [{"role": "user", "content": prompt},
                {"role": "assistant", "content": response}]
        ids = self.tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)
        return ids[-self.max_length:]

    def __getitem__(self, i):
        p = self.pairs[i]
        return {"a": self._encode(p["prompt"], p["response_a"]),
                "b": self._encode(p["prompt"], p["response_b"]),
                "p_hat": p["p_hat"],
                "len_a": len(p["response_a"] or ""), "len_b": len(p["response_b"] or "")}


def collate(batch, pad_id):
    def pad(seqs):
        n = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), n), pad_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), n), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
            mask[i, :len(s)] = 1
        return ids, mask
    a_ids, a_mask = pad([b["a"] for b in batch])
    b_ids, b_mask = pad([b["b"] for b in batch])
    return {"a_ids": a_ids, "a_mask": a_mask, "b_ids": b_ids, "b_mask": b_mask,
            "p_hat": torch.tensor([b["p_hat"] for b in batch], dtype=torch.float32),
            "len_a": torch.tensor([b["len_a"] for b in batch], dtype=torch.float32),
            "len_b": torch.tensor([b["len_b"] for b in batch], dtype=torch.float32)}


class ScalarRewardModel(torch.nn.Module):
    """Backbone + a scalar head read at the last non-padding token."""

    def __init__(self, model_path, torch_dtype=torch.bfloat16, attn="sdpa"):
        super().__init__()
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(
            model_path, torch_dtype=torch_dtype, attn_implementation=attn)
        hidden = self.backbone.config.hidden_size
        self.head = torch.nn.Linear(hidden, 1, dtype=torch_dtype)
        torch.nn.init.normal_(self.head.weight, std=1.0 / math.sqrt(hidden + 1))
        torch.nn.init.zeros_(self.head.bias)

    def forward(self, input_ids, attention_mask):
        h = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        # index of the last attended token in each row
        last = attention_mask.sum(dim=1) - 1
        pooled = h[torch.arange(h.shape[0], device=h.device), last]
        return self.head(pooled).squeeze(-1).float()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    nll, floor, deltas, labels, lens = [], [], [], [], []
    for batch in loader:
        ra = model(batch["a_ids"].to(device), batch["a_mask"].to(device))
        rb = model(batch["b_ids"].to(device), batch["b_mask"].to(device))
        p = batch["p_hat"].to(device)
        nll.append(soft_bt_loss(ra, rb, p, reduction="none").cpu())
        floor.append(soft_label_entropy(p.cpu()))
        deltas.append((ra - rb).cpu())
        labels.append(p.cpu())
        lens.append((batch["len_a"] - batch["len_b"]))
    nll = torch.cat(nll).double()
    floor = torch.cat(floor).double()
    delta = torch.cat(deltas).double()
    label = torch.cat(labels).double()
    length = torch.cat(lens).double()
    p_pred = torch.sigmoid(delta)
    model.train()
    return {
        "soft_bt_nll": float(nll.mean()),
        "soft_label_entropy_floor": float(floor.mean()),
        "excess_nll_over_floor": float((nll - floor).mean()),
        "hard_accuracy_excluding_ties": hard_accuracy_excluding_ties(delta, label),
        "brier": brier_score(p_pred, label),
        "ece": expected_calibration_error(p_pred, label),
        "reward_margin_length_correlation": pearson(delta, length),
        "n_pairs": int(delta.numel()),
        "tie_fraction": float(((label - 0.5).abs() < 1e-9).double().mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verdicts", type=Path, required=True)
    ap.add_argument("--responses", nargs="+", required=True,
                    help="seed=path.json for EVERY pool the bank was judged from")
    ap.add_argument("--validation-verdicts", type=Path, default=None)
    ap.add_argument("--validation-responses", nargs="*", default=None)
    ap.add_argument("--objective", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--backbone-lr", type=float, default=1e-6)
    ap.add_argument("--head-lr-multiplier", type=float, default=5.0)
    ap.add_argument("--global-batch-size", type=int, default=64)
    ap.add_argument("--per-device-batch-size", type=int, default=1)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--validation-fraction", type=float, default=0.1)
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()

    set_all_seeds(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pools = load_response_pools(args.responses)
    pairs, dropped = load_pairs(args.verdicts, args.objective, pools)
    if not pairs:
        raise SystemExit(f"no complete swap-averaged pairs for objective "
                         f"{args.objective!r} in {args.verdicts}")
    if args.validation_verdicts:
        val_pools = load_response_pools(args.validation_responses or args.responses)
        val_pairs, val_dropped = load_pairs(args.validation_verdicts, args.objective,
                                            val_pools)
        train_pairs = pairs
    else:
        # Split by PROMPT, never by pair: two pairs of one prompt share responses,
        # so a pair-level split leaks the prompt's responses across the boundary --
        # the same defect the dataset split was rebuilt to remove.
        prompts = sorted({p["prompt_id"] for p in pairs})
        rng = random.Random(args.seed)
        rng.shuffle(prompts)
        n_val = max(1, int(len(prompts) * args.validation_fraction))
        val_ids = set(prompts[:n_val])
        val_pairs = [p for p in pairs if p["prompt_id"] in val_ids]
        train_pairs = [p for p in pairs if p["prompt_id"] not in val_ids]
        val_dropped = 0
    assert not ({p["prompt_id"] for p in train_pairs}
                & {p["prompt_id"] for p in val_pairs}), "RM train/val share a prompt"

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_path)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    pad_id = tok.pad_token_id

    train_ds = PairDataset(train_pairs, tok, args.max_length)
    val_ds = PairDataset(val_pairs, tok, args.max_length)
    coll = lambda b: collate(b, pad_id)
    train_loader = DataLoader(train_ds, batch_size=args.per_device_batch_size,
                              shuffle=True, collate_fn=coll, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.per_device_batch_size,
                            shuffle=False, collate_fn=coll)

    model = ScalarRewardModel(args.model_path).to(device)
    model.backbone.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    accum = max(1, args.global_batch_size // args.per_device_batch_size)
    opt = torch.optim.AdamW(
        [{"params": model.backbone.parameters(), "lr": args.backbone_lr},
         {"params": model.head.parameters(),
          "lr": args.backbone_lr * args.head_lr_multiplier}],
        weight_decay=args.weight_decay)
    total_steps = max(1, (len(train_loader) * args.epochs) // accum)
    warmup = int(total_steps * args.warmup_ratio)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warmup) if s < warmup
        else max(0.0, (total_steps - s) / max(1, total_steps - warmup)))

    record = {
        "objective": args.objective, "seed": args.seed, "epochs": args.epochs,
        "backbone_lr": args.backbone_lr,
        "head_lr": args.backbone_lr * args.head_lr_multiplier,
        "head_lr_multiplier": args.head_lr_multiplier,
        "global_batch_size": args.global_batch_size,
        "gradient_accumulation": accum, "max_length": args.max_length,
        "warmup_ratio": args.warmup_ratio, "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm, "full_finetune": True,
        "lora": False, "quantization": None, "dtype": "bfloat16",
        "loss": "soft_bradley_terry_swap_averaged", "ties_kept": True,
        "n_train_pairs": len(train_pairs), "n_val_pairs": len(val_pairs),
        "single_order_pairs_dropped": dropped + val_dropped,
        "verdicts": str(args.verdicts),
        "model_path": args.model_path,
        "selection_metric": "validation soft BT NLL (read against soft_label_entropy)",
    }
    record["validation_before_training"] = evaluate(model, val_loader, device)

    if not args.eval_only:
        model.train()
        step = 0
        for epoch in range(args.epochs):
            for i, batch in enumerate(train_loader):
                ra = model(batch["a_ids"].to(device), batch["a_mask"].to(device))
                rb = model(batch["b_ids"].to(device), batch["b_mask"].to(device))
                loss = soft_bt_loss(ra, rb, batch["p_hat"].to(device)) / accum
                loss.backward()
                if (i + 1) % accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    opt.step()
                    sched.step()
                    opt.zero_grad(set_to_none=True)
                    step += 1
                    if step % 20 == 0:
                        print(f"[rm:{args.objective}] epoch {epoch} step {step}/{total_steps} "
                              f"loss {float(loss) * accum:.4f}", flush=True)
        record["optimizer_steps"] = step

    record["validation_after_training"] = evaluate(model, val_loader, device)
    torch.save({"head": model.head.state_dict()}, args.out_dir / "head.pt")
    model.backbone.save_pretrained(args.out_dir / "backbone")
    tok.save_pretrained(args.out_dir / "backbone")
    (args.out_dir / "rm_record.json").write_text(json.dumps(record, indent=2))
    print(json.dumps({k: record[k] for k in
                      ("objective", "seed", "n_train_pairs", "n_val_pairs",
                       "validation_after_training")}, indent=1))


if __name__ == "__main__":
    main()
