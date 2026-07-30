"""Empirically identify which ArmoRM objective each RMOD value-function head
tracks. For a sample of (prompt, response) pairs, compute each of the 5 VF
heads' terminal value (last-token prediction, ~= predicted reward per CD-FUDGE)
and save them. Correlated offline against ArmoRM attribute rewards.
Run with the RMOD venv (venv_rmod) on a GPU.
"""
import argparse, json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from robust_multi_objective_decoding.multi_objective_value_function import MultiHeadValueFunction
from robust_multi_objective_decoding.utils.load_utils import load_base_vf_module_state_dict_from_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses_json", required=True, help="[{prompt, all_generated_responses:[...]}]")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--base_model", default="openai-community/gpt2-large")
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--num_heads", type=int, default=5)
    ap.add_argument("--max_samples", type=int, default=300)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = "cuda"

    tok = AutoTokenizer.from_pretrained(args.base_model, cache_dir=args.cache_dir, padding_side="right")
    tok.truncation_side = "left"   # keep the response (end); gpt2 caps at 1024 positions
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16,
                                                cache_dir=args.cache_dir, attn_implementation="eager")
    vf = MultiHeadValueFunction(base_model=base, base_model_hidden_dim=1280, num_heads=args.num_heads,
                                token_vocab_size=50257,
                                torch_dtype=torch.bfloat16, lora_config=None).to(dev).eval()
    vf.load_state_dict(load_base_vf_module_state_dict_from_checkpoint(args.checkpoint))

    rows = json.load(open(args.responses_json))[:args.max_samples]
    out = []
    bs = 16
    texts = [(r["prompt"], r["all_generated_responses"][0]) for r in rows]
    for i in range(0, len(texts), bs):
        block = texts[i:i + bs]
        strs = [p + "\n" + y for p, y in block]
        enc = tok(strs, return_tensors="pt", padding=True, truncation=True, max_length=1000).to(dev)
        with torch.inference_mode():
            vals = vf(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            vals = vals[0] if isinstance(vals, tuple) else vals   # [b, heads, seq]
        b = vals.shape[0]
        last = enc["attention_mask"].sum(1) - 1                    # last real token idx, [b]
        term = vals[torch.arange(b), :, last].float().cpu().tolist()  # [b, heads]
        for (p, y), hv in zip(block, term):
            out.append({"prompt": p, "head_values": hv})
    json.dump(out, open(args.out, "w"))
    print(f"[vf-probe] wrote {len(out)} rows x {args.num_heads} heads -> {args.out}")


if __name__ == "__main__":
    main()
