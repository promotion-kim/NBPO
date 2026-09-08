#!/usr/bin/env python3
"""Score generated pools with the frozen preference ensembles, and diagnose them.

Every response is encoded **once** per (model, seed); all pair scores are head
evaluations on cached encodings. Scoring 16 responses pairwise would otherwise
cost 128 encoder passes per prompt instead of 16.

Two tensors come out per (model, geometry):

    A_policy[k, x, i, j] = P_k(learner_i > comparator_j | x) - 1/2
    A_ref[k, x, i, j]    = P_k(comparator_i > comparator_j | x) - 1/2

``A_ref`` is the reference-as-learner tensor the disagreement point is measured
from. Both are built from the **calibrated ensemble mean**, so the temperature
fitted on validation is applied before averaging, exactly as it will be
downstream.

The diagnostics answer one question -- which pool geometry gives a more stable
finite-pool target -- and they are computed on validation prompts and compute
only. No downstream policy performance is consulted.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

OBJECTIVES = ("helpfulness", "harmlessness")
EPS = 1e-9


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    return np.log(p) - np.log1p(-p)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def load_model(ckpt_dir: Path, device):
    from transformers import AutoModel, AutoTokenizer
    from mnpo_scripts.gpm import AntiSymmetricGPM
    from scripts.experiments.iclr2027_table1_v2.train_saferlhf_gpm import ScalarBT
    blob = torch.load(ckpt_dir / "model.pt", map_location="cpu", weights_only=False)
    tok = AutoTokenizer.from_pretrained(ckpt_dir)
    enc = AutoModel.from_pretrained(blob["encoder"])
    hidden, kind = blob["hidden"], blob["kind"]
    if kind == "gpm":
        model = AntiSymmetricGPM(enc, hidden, len(OBJECTIVES), width=blob["width"],
                                 dropout=0.0)
    else:
        model = ScalarBT(enc, hidden, len(OBJECTIVES),
                         width=blob["bt_width"] or blob["width"], dropout=0.0)
    model.load_state_dict(blob["state_dict"])
    return model.to(device).eval(), tok, kind


@torch.no_grad()
def encode_all(model, tok, prompt, responses, device, max_len, batch=32):
    hs = []
    for i in range(0, len(responses), batch):
        chunk = responses[i:i + batch]
        enc = tok([prompt] * len(chunk), chunk, padding=True,
                  truncation="longest_first", max_length=max_len,
                  return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            hs.append(model.encode(enc["input_ids"], enc["attention_mask"]).float())
    return torch.cat(hs, dim=0)


@torch.no_grad()
def pair_matrix(model, h_rows, h_cols, k):
    """``P_k(row_i > col_j)`` for every pair, from cached encodings."""
    n, m = h_rows.shape[0], h_cols.shape[0]
    out = torch.empty(n, m, dtype=torch.float32, device=h_rows.device)
    for i in range(n):
        rep = h_rows[i:i + 1].expand(m, -1)
        out[i] = torch.sigmoid(model.logit(rep, h_cols, k))
    return out.double().cpu().numpy()


def entropy(p):
    p = np.clip(p, EPS, 1 - EPS)
    return float(np.mean(-(p * np.log(p) + (1 - p) * np.log(1 - p))))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool-dir", type=Path, required=True)
    ap.add_argument("--ensemble-dir", type=Path, required=True)
    ap.add_argument("--calibration", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[41, 42, 43])
    ap.add_argument("--geometries", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--max-len", type=int, default=384)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cal = json.loads(args.calibration.read_text())["calibration"]
    rows = read_jsonl(args.pool_dir / "pool_responses.jsonl")
    by_prompt = defaultdict(lambda: {"learner": {}, "comparator": {}})
    prompt_text = {}
    for r in rows:
        by_prompt[r["prompt_id"]][r["role"]][r["sample_index"]] = r
        prompt_text[r["prompt_id"]] = r["prompt"]
    prompt_ids = sorted(by_prompt)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    n_max = max(args.geometries)
    # P[model][seed][kind] -> arrays indexed [k, x, i, j]
    acc = {m: {"policy": [], "ref": []} for m in ("gpm", "bt")}
    timings = {}

    for model_kind in ("gpm", "bt"):
        for seed in args.seeds:
            t0 = time.time()
            model, tok, kind = load_model(
                args.ensemble_dir / f"ckpt_{model_kind}_seed{seed}", device)
            pol = np.empty((len(OBJECTIVES), len(prompt_ids), n_max, n_max))
            ref = np.empty((len(OBJECTIVES), len(prompt_ids), n_max, n_max))
            for xi, pid in enumerate(prompt_ids):
                blk = by_prompt[pid]
                learners = [blk["learner"][i]["response"] for i in range(n_max)]
                comps = [blk["comparator"][i]["response"] for i in range(n_max)]
                h_l = encode_all(model, tok, prompt_text[pid], learners, device,
                                 args.max_len)
                h_c = encode_all(model, tok, prompt_text[pid], comps, device,
                                 args.max_len)
                for k, o in enumerate(OBJECTIVES):
                    T = cal[model_kind][str(seed)][o]["temperature"]
                    pol[k, xi] = sigmoid(logit(pair_matrix(model, h_l, h_c, k)) / T)
                    ref[k, xi] = sigmoid(logit(pair_matrix(model, h_c, h_c, k)) / T)
                if (xi + 1) % 50 == 0:
                    print(f"    {model_kind} seed {seed}: {xi+1}/{len(prompt_ids)} "
                          f"prompts [{time.time()-t0:.0f}s]", flush=True)
            acc[model_kind]["policy"].append(pol)
            acc[model_kind]["ref"].append(ref)
            timings[f"{model_kind}_seed{seed}_seconds"] = time.time() - t0
            del model
            torch.cuda.empty_cache()
            print(f"  {model_kind} seed {seed} done in "
                  f"{timings[f'{model_kind}_seed{seed}_seconds']:.0f}s", flush=True)

    out = {"prompt_ids": prompt_ids, "objectives": list(OBJECTIVES),
           "timings_seconds": timings, "geometries": {}}
    for model_kind in ("gpm", "bt"):
        for name in ("policy", "ref"):
            stack = np.stack(acc[model_kind][name])          # (seed, K, X, n, n)
            np.savez_compressed(
                args.out_dir / f"probs_{model_kind}_{name}.npz",
                p=stack.astype(np.float32))
    print(f"\nwrote probability tensors to {args.out_dir}")
    (args.out_dir / "scoring_manifest.json").write_text(json.dumps({
        "n_prompts": len(prompt_ids), "n_max_pool": n_max,
        "seeds": args.seeds, "models": ["gpm", "bt"],
        "calibration_source": str(args.calibration),
        "calibration_applied_before_averaging": True,
        "encoding_note": ("each response is encoded once per (model, seed); all "
                          "pair scores are head evaluations on cached encodings"),
        "timings_seconds": timings,
    }, indent=2))


if __name__ == "__main__":
    main()
