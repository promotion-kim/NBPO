#!/usr/bin/env python3
"""Where does the zero-step mismatch enter? One factor at a time.

At theta = theta_t the quantity

    h_ab = [log pi(a) - log pi_t(a)] - [log pi(b) - log pi_t(b)]

is identically zero: the same weights on the same text. A measured h RMS of 0.35
therefore means the two sides were NOT computed the same way, and this walks the
ladder of possible differences until the first one that produces the error.

Conditions, each changing exactly one thing from the production path:

``repeat``      identical model, identical collated batch, forward run twice --
                bounds the noise floor of the computation itself
``train_mode``  model.train() instead of eval() -- stochastic layers
``alone``       each pair collated on its own instead of with batch neighbours
``padded``      each pair collated next to the longest response in the probe
``fp32``        weights upcast, same inputs -- separates arithmetic precision
``cache``       the value precompute stored for the same row

Everything reuses precompute's own collator and concatenated_forward, so the
baseline IS the production path rather than a reimplementation of it. Response
log-probabilities and pair differences are reported separately, because the pair
difference is what the loss sees and cancellations there are not guaranteed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def summarize(err):
    e = np.abs(np.asarray(err, dtype=np.float64))
    if e.size == 0:
        return {}
    return {"rms": float(np.sqrt((e ** 2).mean())), "median": float(np.median(e)),
            "p95": float(np.percentile(e, 95)), "max": float(e.max()), "n": int(e.size)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--precomputed", required=True,
                    help="a scored/precomputed dir whose history0 columns are pi_t")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n-rows", type=int, default=24)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--max-prompt-length", type=int, default=1024)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from mnpo_scripts.precompute import concatenated_forward
    from mnpo_scripts.precompute_trainer import PreferenceDataCollatorWithPadding

    torch.use_deterministic_algorithms(False)
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model)
    # Llama ships without a pad token; training resolved it to the EOS id
    # (128009 in the trainer's own log). Mirror that here rather than inventing
    # one, because the pad id changes what the collator writes into the padded
    # positions and so changes the forward.
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    print(f"pad_token_id={tok.pad_token_id} eos_token_id={tok.eos_token_id}", flush=True)
    ds = load_from_disk(args.precomputed)[args.split]

    # a probe spanning short and long responses, chosen by length so the
    # length-dependence of any error is visible rather than averaged away
    lens = np.array([len(c) + len(r) for c, r in zip(ds["chosen"], ds["rejected"])])
    order = np.argsort(lens)
    pick = list(order[: args.n_rows // 2]) + list(order[-(args.n_rows // 2):])
    rows = ds.select([int(i) for i in pick])

    coll = PreferenceDataCollatorWithPadding(
        tokenizer=tok, max_length=args.max_length,
        max_prompt_length=args.max_prompt_length, label_pad_token_id=-100,
        padding_value=0, truncation_mode="keep_end", is_encoder_decoder=False,
        max_target_length=None)

    feats = [{"prompt": rows[i]["prompt"], "chosen": rows[i]["chosen"],
              "rejected": rows[i]["rejected"]} for i in range(len(rows))]

    def logps(model, batch_feats, train_mode=False):
        model.train() if train_mode else model.eval()
        b = coll(batch_feats)
        b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
        with torch.no_grad():
            c, r = concatenated_forward(model, b, average_log_prob=False)
        return c.float().cpu().numpy(), r.float().cpu().numpy()

    def run_all(model, mode, chunk, train_mode=False):
        C, R = [], []
        for i in range(0, len(feats), chunk):
            c, r = logps(model, feats[i:i + chunk], train_mode)
            C += list(c); R += list(r)
        return np.asarray(C), np.asarray(R)

    print("loading bf16 model", flush=True)
    m = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, use_cache=False).to(dev).eval()

    report = {"model": args.model, "n_probe_rows": len(feats),
              "pad_token_id": tok.pad_token_id, "eos_token_id": tok.eos_token_id,
              "response_char_lengths": [int(x) for x in lens[pick]],
              "torch": torch.__version__,
              "attn_impl": getattr(m.config, "_attn_implementation", "?"),
              "conditions": {}}

    base_c, base_r = run_all(m, "base", 2)                       # production: chunk 2
    def record(name, c, r, note=""):
        dc = c - base_c; dr = r - base_r
        dpair = (c - r) - (base_c - base_r)
        report["conditions"][name] = {
            "note": note,
            "response_logp_error": summarize(np.concatenate([dc, dr])),
            "pair_difference_error": summarize(dpair)}
        s = report["conditions"][name]
        print(f"{name:12s} resp rms {s['response_logp_error']['rms']:.5f} "
              f"max {s['response_logp_error']['max']:.5f} | "
              f"pair rms {s['pair_difference_error']['rms']:.5f} "
              f"max {s['pair_difference_error']['max']:.5f}  {note}", flush=True)

    c, r = run_all(m, "repeat", 2)
    record("repeat", c, r, "same model, same batching, run twice")
    c, r = run_all(m, "train_mode", 2, train_mode=True)
    record("train_mode", c, r, "model.train(): stochastic layers")
    c, r = run_all(m, "alone", 1)
    record("alone", c, r, "batch size 1: no padding neighbours")
    c, r = run_all(m, "chunk8", 8)
    record("chunk8", c, r, "batch size 8: longer padding")
    c, r = run_all(m, "chunk_all", len(feats))
    record("chunk_all", c, r, "one batch: padded to the longest probe row")

    del m
    torch.cuda.empty_cache()
    print("loading fp32 model", flush=True)
    m32 = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, use_cache=False).to(dev).eval()
    c, r = run_all(m32, "fp32", 2)
    record("fp32", c, r, "float32 weights, same inputs and batching")
    del m32
    torch.cuda.empty_cache()

    # against what precompute actually stored for these very rows
    cache_c = np.asarray([rows[i]["history0_chosen_logps"] for i in range(len(rows))])
    cache_r = np.asarray([rows[i]["history0_rejected_logps"] for i in range(len(rows))])
    dpair = (base_c - base_r) - (cache_c - cache_r)
    report["conditions"]["cache"] = {
        "note": "production path here vs the value precompute stored for the same row",
        "response_logp_error": summarize(np.concatenate([base_c - cache_c,
                                                         base_r - cache_r])),
        "pair_difference_error": summarize(dpair)}
    s = report["conditions"]["cache"]
    print(f"{'cache':12s} resp rms {s['response_logp_error']['rms']:.5f} "
          f"max {s['response_logp_error']['max']:.5f} | "
          f"pair rms {s['pair_difference_error']['rms']:.5f} "
          f"max {s['pair_difference_error']['max']:.5f}", flush=True)

    # is the error length-dependent?
    L = np.asarray([len(rows[i]["chosen"]) for i in range(len(rows))], float)
    e = np.abs(base_c - cache_c)
    if L.std() > 0 and e.std() > 0:
        report["cache_error_vs_response_length_pearson"] = float(np.corrcoef(L, e)[0, 1])
        print(f"cache error vs response length: r = "
              f"{report['cache_error_vs_response_length_pearson']:+.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
