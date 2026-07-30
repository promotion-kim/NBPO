"""Decisive test of whether the SafeRLHF value function is useful for best-of-K:
within each prompt, does the VF's predicted value rank responses in the same
order as the true Beaver score? Reports mean within-prompt Spearman correlation
per head. A best-of-K selector can only beat base if this is clearly positive.
"""
import argparse, json, statistics as st
import torch
from datasets import load_from_disk
from scipy.stats import spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer
from robust_multi_objective_decoding.multi_objective_value_function import MultiHeadValueFunction
from robust_multi_objective_decoding.utils.load_utils import load_base_vf_module_state_dict_from_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--base_model", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--labels", nargs="+", default=["rewards_harmless", "rewards_helpful"])
    ap.add_argument("--hidden_dim", type=int, default=2048)
    ap.add_argument("--vocab_size", type=int, default=128256)
    ap.add_argument("--num_prompts", type=int, default=150)
    args = ap.parse_args()
    dev = "cuda"

    tok = AutoTokenizer.from_pretrained(args.base_model, cache_dir=args.cache_dir, padding_side="right")
    tok.truncation_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16,
                                                cache_dir=args.cache_dir, attn_implementation="eager")
    vf = MultiHeadValueFunction(base_model=base, base_model_hidden_dim=args.hidden_dim,
                                num_heads=len(args.labels), token_vocab_size=args.vocab_size,
                                torch_dtype=torch.bfloat16, lora_config=None).to(dev).eval()
    vf.load_state_dict(load_base_vf_module_state_dict_from_checkpoint(args.checkpoint))

    ds = load_from_disk(args.data)
    groups = {}
    for r in ds:
        groups.setdefault(r["prompt"], []).append(r)
    prompts = [p for p in groups if len(groups[p]) >= 4][:args.num_prompts]

    def vf_vals(prompt, responses):
        strs = [prompt + "\n" + y for y in responses]
        enc = tok(strs, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(dev)
        with torch.inference_mode():
            out = vf(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            out = out[0] if isinstance(out, tuple) else out          # [b, heads, seq]
        last = enc["attention_mask"].sum(1) - 1
        return out[torch.arange(out.shape[0]), :, last].float().cpu()  # [b, heads]

    per_head = {i: [] for i in range(len(args.labels))}
    for p in prompts:
        rows = groups[p]
        vals = vf_vals(p, [r["response"] for r in rows])
        for hi, lab in enumerate(args.labels):
            true = [r[lab] for r in rows]
            rho, _ = spearmanr(vals[:, hi].tolist(), true)
            if rho == rho:                                            # not nan
                per_head[hi].append(rho)
    for hi, lab in enumerate(args.labels):
        v = per_head[hi]
        print(f"{lab}: mean within-prompt Spearman = {st.mean(v):.3f} "
              f"(median {st.median(v):.3f}, n={len(v)} prompts, frac>0 {sum(x>0 for x in v)/len(v):.2f})")


if __name__ == "__main__":
    main()
