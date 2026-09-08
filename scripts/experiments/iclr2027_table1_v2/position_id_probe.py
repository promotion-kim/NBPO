#!/usr/bin/env python3
"""Is the zero-step mismatch left padding without position ids?

The collator left-pads prompts (`precompute_trainer.py` flips the prompt so the
padding lands on the left) and the forward passes only `input_ids` and
`attention_mask`. With no `position_ids`, the model numbers positions from the
start of the PADDED sequence, so every real token's RoPE position is shifted by
however much left padding that row received -- which depends on the longest
prompt in its batch, and therefore on batch composition.

If that is the mechanism, supplying `position_ids` computed from the attention
mask should make the same response score the same regardless of who it was
batched with, up to floating-point noise. If it does not, the mechanism is
elsewhere and this rules the hypothesis out.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def stats(e):
    e = np.abs(np.asarray(e, float))
    return {"rms": float(np.sqrt((e ** 2).mean())), "median": float(np.median(e)),
            "p95": float(np.percentile(e, 95)), "max": float(e.max())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--precomputed", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-rows", type=int, default=24)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from mnpo_scripts.precompute import concatenated_inputs, get_batch_logps
    from mnpo_scripts.precompute_trainer import PreferenceDataCollatorWithPadding

    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    ds = load_from_disk(args.precomputed)[args.split]
    lens = np.array([len(c) + len(r) for c, r in zip(ds["chosen"], ds["rejected"])])
    order = np.argsort(lens)
    pick = [int(i) for i in list(order[: args.n_rows // 2]) + list(order[-(args.n_rows // 2):])]
    rows = ds.select(pick)
    feats = [{"prompt": rows[i]["prompt"], "chosen": rows[i]["chosen"],
              "rejected": rows[i]["rejected"]} for i in range(len(rows))]

    coll = PreferenceDataCollatorWithPadding(
        tokenizer=tok, max_length=2048, max_prompt_length=1024,
        label_pad_token_id=-100, padding_value=0, truncation_mode="keep_end",
        is_encoder_decoder=False, max_target_length=None)

    m = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, use_cache=False).to(dev).eval()

    def forward(batch_feats, with_positions):
        b = coll(batch_feats)
        b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
        cb = concatenated_inputs(b)
        ids = cb["concatenated_input_ids"]
        mask = cb["concatenated_attention_mask"]
        labels = cb["concatenated_labels"]
        kw = {}
        if with_positions:
            # the standard left-padding correction: real tokens are numbered
            # from zero, pad positions are clamped and masked out anyway
            pos = (mask.cumsum(-1) - 1).clamp(min=0)
            kw["position_ids"] = pos
        with torch.no_grad():
            logits = m(input_ids=ids, attention_mask=mask, **kw).logits
        lp = get_batch_logps(logits, labels, average_log_prob=False)
        n = b["chosen_labels"].shape[0]
        return lp[:n].float().cpu().numpy(), lp[n:].float().cpu().numpy()

    def run(chunk, with_positions):
        C, R = [], []
        for i in range(0, len(feats), chunk):
            c, r = forward(feats[i:i + chunk], with_positions)
            C += list(c); R += list(r)
        return np.asarray(C), np.asarray(R)

    report = {"model": args.model, "n_rows": len(feats),
              "hypothesis": "left padding without position_ids makes a response's "
                            "score depend on its batch neighbours",
              "left_padding_confirmed": True, "results": {}}

    for with_pos in (False, True):
        a_c, a_r = run(1, with_pos)              # alone
        b_c, b_r = run(8, with_pos)              # batched with padding neighbours
        key = "with_position_ids" if with_pos else "production_no_position_ids"
        report["results"][key] = {
            "response_logp_alone_vs_batched": stats(np.concatenate([a_c - b_c,
                                                                    a_r - b_r])),
            "pair_difference_alone_vs_batched": stats((a_c - a_r) - (b_c - b_r)),
        }
        s = report["results"][key]
        print(f"{key:28s} resp rms {s['response_logp_alone_vs_batched']['rms']:.5f} "
              f"max {s['response_logp_alone_vs_batched']['max']:.5f} | "
              f"pair rms {s['pair_difference_alone_vs_batched']['rms']:.5f} "
              f"max {s['pair_difference_alone_vs_batched']['max']:.5f}", flush=True)

    a = report["results"]["production_no_position_ids"]["response_logp_alone_vs_batched"]["rms"]
    b = report["results"]["with_position_ids"]["response_logp_alone_vs_batched"]["rms"]
    report["batch_sensitivity_reduction_factor"] = (a / b) if b > 0 else None
    report["verdict"] = (
        "position ids explain the batch sensitivity" if b < 0.05 * max(a, 1e-9)
        else "position ids do NOT explain it; look elsewhere")
    print(f"\nreduction factor: {report['batch_sensitivity_reduction_factor']}")
    print(report["verdict"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
