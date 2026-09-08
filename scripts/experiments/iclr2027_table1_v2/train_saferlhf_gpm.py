#!/usr/bin/env python3
"""Anti-symmetric GPM vs a scalar Bradley-Terry baseline on SafeRLHF human labels.

The two models share **one backbone, one optimizer, one schedule and one data
order**; the only difference is the head:

``bt``   ``logit_k = r_k(h_y) - r_k(h_z)``     -- a scalar reward difference,
         transitive by construction, which is exactly the representational limit
         NBPO's game-valued objective claims to escape;
``gpm``  ``logit_k = 1/2 [ a_k(h_y,h_z) - a_k(h_z,h_y) ]`` -- a joint pair score
         antisymmetrized exactly, so ``P(y>z)+P(z>y)=1`` and ``P(y>y)=1/2`` hold
         for any parameters, and the model *can* represent a cycle.

Both heads are the same three-layer MLP; the GPM's first layer is twice as wide
on the input side because it reads a pair. That difference is reported rather
than hidden, and a capacity-matched BT variant (``--bt-width``) is available so
that "more parameters" is never the explanation.

Labels are the released binary ids. **No tie is invented** -- the schema cannot
express one -- and no label is softened.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from mnpo_scripts.gpm import (
    AntiSymmetricGPM, GPMProvenance, assert_not_scalar_decomposable,
)

OBJECTIVES = ("helpfulness", "harmlessness")
FIELD = {"helpfulness": "better_response_id", "harmlessness": "safer_response_id"}


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


class PairData(Dataset):
    def __init__(self, rows, tok, max_len):
        self.rows, self.tok, self.max_len = rows, tok, max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        # y is always response_0 and z always response_1; the LABEL carries the
        # direction. Randomising which is which would only add noise, since both
        # models are exactly antisymmetric in (y, z) by construction.
        return {
            "y": (r["prompt"], r["response_0"]),
            "z": (r["prompt"], r["response_1"]),
            "labels": [1.0 - float(r[FIELD[o]]) for o in OBJECTIVES],
            "len_y": len(r["response_0"]), "len_z": len(r["response_1"]),
            "prompt_sha256": r["prompt_sha256"],
        }


def collate(batch, tok, max_len):
    def enc(side):
        return tok([b[side][0] for b in batch], [b[side][1] for b in batch],
                   padding=True, truncation="longest_first", max_length=max_len,
                   return_tensors="pt")
    return {
        "y": enc("y"), "z": enc("z"),
        "labels": torch.tensor([b["labels"] for b in batch], dtype=torch.float32),
        "len_y": torch.tensor([b["len_y"] for b in batch], dtype=torch.float32),
        "len_z": torch.tensor([b["len_z"] for b in batch], dtype=torch.float32),
        "prompt_sha256": [b["prompt_sha256"] for b in batch],
    }


class ScalarBT(nn.Module):
    """The baseline: one scalar reward per objective, differenced."""

    def __init__(self, encoder, hidden, n_objectives, width=512, dropout=0.0):
        super().__init__()
        self.encoder = encoder
        self.n_objectives = n_objectives
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, width), nn.GELU(), nn.Dropout(dropout),
                          nn.Linear(width, width // 2), nn.GELU(),
                          nn.Linear(width // 2, 1))
            for _ in range(n_objectives)])

    def encode(self, input_ids, attention_mask):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask
                         ).last_hidden_state
        last = attention_mask.sum(dim=1) - 1
        return h[torch.arange(h.shape[0], device=h.device), last]

    def reward(self, h, k):
        return self.heads[k](h).squeeze(-1)

    def logit(self, h_y, h_z, k):
        return self.reward(h_y, k) - self.reward(h_z, k)


def build(kind, encoder_name, n_obj, width, bt_width, dropout, device):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(encoder_name)
    enc = AutoModel.from_pretrained(encoder_name)
    hidden = enc.config.hidden_size
    if kind == "gpm":
        model = AntiSymmetricGPM(enc, hidden, n_obj, width=width, dropout=dropout)
    else:
        model = ScalarBT(enc, hidden, n_obj, width=bt_width or width, dropout=dropout)
    return model.to(device), tok, hidden


def head_parameter_count(model):
    return sum(p.numel() for p in model.heads.parameters())


@torch.no_grad()
def predict(model, loader, device, amp_dtype):
    model.eval()
    out = {o: {"p": [], "y": [], "logit": []} for o in OBJECTIVES}
    lens, prompts = [], []
    anti, selftie = 0.0, 0.0
    for batch in loader:
        y = {k: v.to(device) for k, v in batch["y"].items()}
        z = {k: v.to(device) for k, v in batch["z"].items()}
        with torch.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
            h_y = model.encode(y["input_ids"], y["attention_mask"])
            h_z = model.encode(z["input_ids"], z["attention_mask"])
            for k, o in enumerate(OBJECTIVES):
                ell = model.logit(h_y.float(), h_z.float(), k).float()
                rev = model.logit(h_z.float(), h_y.float(), k).float()
                same = model.logit(h_y.float(), h_y.float(), k).float()
                p = torch.sigmoid(ell)
                anti = max(anti, float((p + torch.sigmoid(rev) - 1).abs().max()))
                selftie = max(selftie, float((torch.sigmoid(same) - 0.5).abs().max()))
                out[o]["p"] += p.cpu().tolist()
                out[o]["logit"] += ell.cpu().tolist()
                out[o]["y"] += batch["labels"][:, k].tolist()
        lens += (batch["len_y"] - batch["len_z"]).tolist()
        prompts += batch["prompt_sha256"]
    return out, lens, prompts, {"max_antisymmetry_residual": anti,
                                "max_self_tie_residual": selftie}


def metrics(p, y):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-9, 1 - 1e-9)
    y = np.asarray(y, dtype=np.float64)
    nll = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
    pred = (p >= 0.5).astype(np.float64)
    acc = float((pred == y).mean())
    tpr = float(pred[y == 1].mean()) if (y == 1).any() else float("nan")
    tnr = float(1 - pred[y == 0].mean()) if (y == 0).any() else float("nan")
    order = np.argsort(p)
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    n1, n0 = (y == 1).sum(), (y == 0).sum()
    auc = float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else None
    brier = float(((p - y) ** 2).mean())
    bins = np.clip((p * 15).astype(int), 0, 14)
    ece = 0.0
    for b in range(15):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return {"nll": nll, "accuracy": acc, "balanced_accuracy": (tpr + tnr) / 2,
            "roc_auc": auc, "brier": brier, "ece": float(ece),
            "base_rate": float(y.mean()), "n": int(len(y))}


def constant_predictor(y_train, y_eval):
    """The floor every model must clear: predict the training base rate."""
    q = float(np.clip(np.mean(y_train), 1e-9, 1 - 1e-9))
    y = np.asarray(y_eval, dtype=np.float64)
    return {"nll": float(-(y * math.log(q) + (1 - y) * math.log(1 - q)).mean()),
            "accuracy": float(max(y.mean(), 1 - y.mean())),
            "balanced_accuracy": 0.5, "roc_auc": 0.5,
            "brier": float(((q - y) ** 2).mean()), "predicted_probability": q}


def cluster_bootstrap(pa, pb, y, prompts, n=2000, seed=0):
    """Paired bootstrap resampling PROMPTS, not rows.

    Rows sharing a prompt are not independent, so a row-level bootstrap would
    understate the interval. Prompt clusters are the resampling unit.
    """
    rng = np.random.default_rng(seed)
    idx = {}
    for i, ph in enumerate(prompts):
        idx.setdefault(ph, []).append(i)
    keys = list(idx)
    clip = lambda v: np.clip(np.asarray(v, dtype=np.float64), 1e-9, 1 - 1e-9)
    pa, pb = clip(pa), clip(pb)
    y = np.asarray(y, dtype=np.float64)
    nll = lambda p, s: float(-(y[s] * np.log(p[s]) + (1 - y[s]) * np.log(1 - p[s])).mean())
    accd = lambda p, s: float(((p[s] >= 0.5).astype(float) == y[s]).mean())
    d_nll, d_acc = [], []
    for _ in range(n):
        pick = rng.choice(len(keys), size=len(keys), replace=True)
        s = np.concatenate([idx[keys[i]] for i in pick])
        d_nll.append(nll(pa, s) - nll(pb, s))
        d_acc.append(accd(pa, s) - accd(pb, s))
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {"delta_nll_gpm_minus_bt_mean": float(np.mean(d_nll)),
            "delta_nll_gpm_minus_bt_ci95": q(d_nll),
            "delta_accuracy_gpm_minus_bt_mean": float(np.mean(d_acc)),
            "delta_accuracy_gpm_minus_bt_ci95": q(d_acc),
            "n_bootstrap": n, "resampling_unit": "prompt"}


def pearson(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def train_one(kind, args, device, splits):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    model, tok, hidden = build(kind, args.encoder, len(OBJECTIVES), args.width,
                               args.bt_width, args.dropout, device)
    mk = lambda rows, shuffle: DataLoader(
        PairData(rows, tok, args.max_len), batch_size=args.batch_size,
        shuffle=shuffle, num_workers=args.workers, drop_last=False,
        collate_fn=lambda b: collate(b, tok, args.max_len),
        generator=torch.Generator().manual_seed(args.seed))
    tr, va, te = (mk(splits["train"], True), mk(splits["validation"], False),
                  mk(splits["test"], False))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    steps = args.epochs * len(tr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                                total_steps=max(1, steps),
                                                pct_start=0.06)
    amp = torch.bfloat16
    t0 = time.time()
    step = 0
    for ep in range(args.epochs):
        model.train()
        for batch in tr:
            y = {k: v.to(device) for k, v in batch["y"].items()}
            z = {k: v.to(device) for k, v in batch["z"].items()}
            lab = batch["labels"].to(device)
            with torch.autocast("cuda", dtype=amp, enabled=device.type == "cuda"):
                h_y = model.encode(y["input_ids"], y["attention_mask"])
                h_z = model.encode(z["input_ids"], z["attention_mask"])
            loss = 0.0
            for k in range(len(OBJECTIVES)):
                ell = model.logit(h_y.float(), h_z.float(), k)
                loss = loss + nn.functional.binary_cross_entropy_with_logits(
                    ell, lab[:, k])
            loss = loss / len(OBJECTIVES)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % args.log_every == 0:
                print(f"  [{kind}] ep{ep} step {step}/{steps} loss {float(loss):.4f}",
                      flush=True)
    train_secs = time.time() - t0

    res = {"kind": kind, "train_seconds": train_secs,
           "head_parameters": head_parameter_count(model),
           "total_parameters": sum(p.numel() for p in model.parameters()),
           "optimizer_steps": step}
    for name, loader in (("validation", va), ("test", te)):
        out, lens, prompts, exact = predict(model, loader, device, amp)
        res[name] = {"exactness": exact, "per_objective": {}}
        for o in OBJECTIVES:
            res[name]["per_objective"][o] = {
                **metrics(out[o]["p"], out[o]["y"]),
                "logit_vs_length_difference_pearson": pearson(out[o]["logit"], lens)}
        res[f"_{name}_raw"] = {"p": {o: out[o]["p"] for o in OBJECTIVES},
                               "y": {o: out[o]["y"] for o in OBJECTIVES},
                               "prompts": prompts}
    res["predicted_cycles_test"] = predicted_cycles(
        model, splits["test"], tok, device, args.max_len, amp, seed=args.seed)
    if kind == "gpm":
        with torch.no_grad():
            h = torch.randn(8, hidden, device=device)
            res["not_scalar_decomposable_random_inputs"] = assert_not_scalar_decomposable(
                model, h, 0)
        res["provenance"] = GPMProvenance(
            encoder_name=args.encoder, adaptation_mode="full_finetune",
            n_objectives=len(OBJECTIVES), hidden_size=hidden, head_width=args.width,
            label_source="PKU-SafeRLHF released better_response_id/safer_response_id",
        ).to_dict()
    if args.save_checkpoint:
        ck = args.out_dir / f"ckpt_{kind}_seed{args.seed}"
        ck.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(),
                    "kind": kind, "seed": args.seed,
                    "encoder": args.encoder, "hidden": hidden,
                    "width": args.width, "bt_width": args.bt_width,
                    "dropout": args.dropout, "objectives": list(OBJECTIVES)},
                   ck / "model.pt")
        tok.save_pretrained(ck)
        res["checkpoint"] = str(ck)
    del model
    torch.cuda.empty_cache()
    return res


@torch.no_grad()
def predicted_cycles(model, rows, tok, device, max_len, amp_dtype, max_prompts=400,
                     max_responses=6, seed=0):
    """Cycles the MODEL predicts, on held-out prompts -- never confused with observed ones.

    SafeRLHF's annotation graph is a near-perfect matching, so the number of
    three-cycles a human actually reported is 0 and no subset of it can be used
    to test cyclic preference. A model, however, is defined on every pair, so it
    can be asked about triples the annotators never compared. That is a
    *prediction*, reported separately and never as evidence about people.
    """
    import collections
    by_prompt = collections.defaultdict(dict)
    for r in rows:
        by_prompt[r["prompt_sha256"]].setdefault("prompt", r["prompt"])
        for side in ("0", "1"):
            by_prompt[r["prompt_sha256"]].setdefault("resp", {})[
                r[f"response_{side}_sha256"]] = r[f"response_{side}"]
    keys = sorted(k for k, v in by_prompt.items() if len(v["resp"]) >= 3)
    rng = np.random.default_rng(seed)
    if len(keys) > max_prompts:
        keys = [keys[i] for i in sorted(rng.choice(len(keys), max_prompts, replace=False))]
    model.eval()
    out = {o: {"triples": 0, "cycles": 0, "prompts_with_a_cycle": 0}
           for o in OBJECTIVES}
    resid = 0.0
    for key in keys:
        blk = by_prompt[key]
        ids = sorted(blk["resp"])[:max_responses]
        texts = [blk["resp"][i] for i in ids]
        enc = tok([blk["prompt"]] * len(texts), texts, padding=True,
                  truncation="longest_first", max_length=max_len, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
            h = model.encode(enc["input_ids"], enc["attention_mask"]).float()
        n = len(ids)
        for k, o in enumerate(OBJECTIVES):
            L = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    if i != j:
                        L[i, j] = float(model.logit(h[i:i + 1], h[j:j + 1], k))
            found = False
            for i in range(n):
                for j in range(i + 1, n):
                    for m in range(j + 1, n):
                        out[o]["triples"] += 1
                        resid = max(resid, abs(L[i, j] + L[j, m] + L[m, i]))
                        s3 = [(L[i, j] > 0), (L[j, m] > 0), (L[m, i] > 0)]
                        if all(s3) or not any(s3):
                            out[o]["cycles"] += 1
                            found = True
            if found:
                out[o]["prompts_with_a_cycle"] += 1
    for o in OBJECTIVES:
        t = out[o]["triples"]
        out[o]["cycle_rate"] = (out[o]["cycles"] / t) if t else None
    return {"per_objective": out, "n_prompts_scored": len(keys),
            "max_cyclic_logit_residual_on_real_encodings": resid,
            "note": ("PREDICTED, not observed. The human annotation graph contains "
                     "zero triangles, so these triples were never compared by people.")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--encoder", default="roberta-base")
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--bt-width", type=int, default=None,
                    help="head width for the BT baseline; defaults to --width. Set it "
                         "larger to give the scalar model a matched parameter count.")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--limit-train", type=int, default=None)
    ap.add_argument("--models", nargs="+", default=["gpm", "bt"])
    ap.add_argument("--save-checkpoint", action="store_true",
                    help="persist the trained weights. Required for anything that "
                         "must SCORE with the frozen oracle later: without it the "
                         "reported ensemble and the used ensemble are different "
                         "models, because GPU training is not bit-reproducible.")
    ap.add_argument("--save-predictions", action="store_true",
                    help="write per-example probabilities for validation and test so "
                         "an ensemble can be calibrated and combined afterwards "
                         "WITHOUT retraining. Calibration must see validation only.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = {n: read_jsonl(args.splits_dir / f"saferlhf_{n}.jsonl")
              for n in ("train", "validation", "test")}
    if args.limit_train:
        splits["train"] = splits["train"][:args.limit_train]
    manifest = json.loads((args.splits_dir / "split_manifest.json").read_text())

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for kind in args.models:
        print(f"=== training {kind} ===", flush=True)
        results[kind] = train_one(kind, args, device, splits)

    report = {"config": {k: str(v) for k, v in vars(args).items()},
              "split_manifest_prompt_hashes": {
                  n: manifest["splits"][n]["prompt_set_sha256"]
                  for n in ("train", "validation", "test")},
              "observed_three_cycles": {
                  n: {o: manifest["splits"][n]["graph"]["per_objective"][o]["three_cycles"]
                      for o in OBJECTIVES} for n in ("train", "validation", "test")},
              "models": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                         for k, v in results.items()}}

    if "gpm" in results and "bt" in results:
        cmp_block = {}
        for split in ("validation", "test"):
            g, b = results["gpm"][f"_{split}_raw"], results["bt"][f"_{split}_raw"]
            cmp_block[split] = {}
            for o in OBJECTIVES:
                cmp_block[split][o] = cluster_bootstrap(
                    g["p"][o], b["p"][o], g["y"][o], g["prompts"], seed=args.seed)
                tr_y = [1.0 - float(r[FIELD[o]]) for r in splits["train"]]
                cmp_block[split][o]["constant_predictor"] = constant_predictor(
                    tr_y, g["y"][o])
        report["gpm_vs_bt"] = cmp_block
    if args.save_predictions:
        import numpy as _np
        for kind, r in results.items():
            for split in ("validation", "test"):
                raw = r[f"_{split}_raw"]
                _np.savez_compressed(
                    args.out_dir / f"pred_{kind}_seed{args.seed}_{split}.npz",
                    prompts=_np.array(raw["prompts"]),
                    **{f"p_{o}": _np.asarray(raw["p"][o], dtype=_np.float64)
                       for o in OBJECTIVES},
                    **{f"y_{o}": _np.asarray(raw["y"][o], dtype=_np.float64)
                       for o in OBJECTIVES})
        print(f"wrote per-seed predictions for seed {args.seed}", flush=True)

    (args.out_dir / f"gpm_vs_bt_seed{args.seed}.json").write_text(
        json.dumps(report, indent=2, default=str))
    print(json.dumps(report["models"], indent=2, default=str)[:4000])
    if "gpm_vs_bt" in report:
        print(json.dumps(report["gpm_vs_bt"], indent=2)[:3000])


if __name__ == "__main__":
    main()
