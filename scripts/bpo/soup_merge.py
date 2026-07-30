#!/usr/bin/env python3
"""Rewarded Soups: uniform average of per-objective expert weights.
Usage: soup_merge.py --experts d1,d2,... --base BASE --output OUT"""
import argparse, torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    dirs = args.experts.split(",")
    avg = None
    for i, d in enumerate(dirs):
        sd = AutoModelForCausalLM.from_pretrained(d, torch_dtype=torch.float32).state_dict()
        if avg is None:
            avg = {k: v.clone() for k, v in sd.items()}
        else:
            for k in avg:
                avg[k] += sd[k]
        print(f"[soup] loaded {d} ({i+1}/{len(dirs)})")
    for k in avg:
        avg[k] /= len(dirs)
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16)
    model.load_state_dict({k: v.to(torch.bfloat16) for k, v in avg.items()})
    model.save_pretrained(args.output, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.base).save_pretrained(args.output)
    print(f"[soup] wrote merged model of {len(dirs)} experts -> {args.output}")


if __name__ == "__main__":
    main()
