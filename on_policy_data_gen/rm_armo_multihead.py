"""Score responses with several ArmoRM attribute heads in a SINGLE forward pass
and emit one per-objective jsonl (Skywork-style `all_rm_scores`) each, so the
RONPO multi-objective pipeline can consume the 5 ArmoRM heads as 5 objectives
without running ArmoRM once per head.

  python -m on_policy_data_gen.rm_armo_multihead \
    --input_file merged.json --output_dir scored/ --split train \
    --indices 4,6,7,8,9 --names conciseness,instruction_following,truthfulness,honesty,helpfulness \
    --cache_dir $HF_HUB_CACHE  [--negate_indices 4]   # conciseness = -verbosity

Input rows need `prompt` and `all_generated_responses`. Output files are
`<output_dir>/<split>_<name>.jsonl` with `all_rm_scores` per response.
"""
import argparse, json, os
import torch


def _patch_transformers_compat():
    # ArmoRM's custom modeling imports symbols removed in newer transformers.
    try:
        import transformers.models.llama.modeling_llama as lm
        if not hasattr(lm, "LLAMA_INPUTS_DOCSTRING"):
            lm.LLAMA_INPUTS_DOCSTRING = ""
    except Exception:
        pass
    try:
        import transformers.generation as generation
        if not hasattr(generation, "GenerationMixin"):
            from transformers.generation.utils import GenerationMixin
            generation.GenerationMixin = GenerationMixin
    except Exception:
        pass


_patch_transformers_compat()
del _patch_transformers_compat

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "RLHFlow/ArmoRM-Llama3-8B-v0.1"
MAX_SEQ_LENGTH = 4096


def load_rows(path):
    if path.endswith(".jsonl"):
        return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    return json.load(open(path, encoding="utf-8"))


def truncate_user(conv, max_chars):
    if max_chars <= 0:
        return conv
    return [{**m, "content": m["content"][:max_chars]} if m["role"] == "user" else m for m in conv]


def score_block(convs, tok, model, device, indices):
    enc = tok.apply_chat_template(convs, return_tensors="pt", padding=True,
                                  truncation=True, max_length=MAX_SEQ_LENGTH)
    ids = enc["input_ids"] if not isinstance(enc, torch.Tensor) else enc
    with torch.inference_mode():
        rewards = model(ids.to(device)).rewards          # [n, n_attributes]
    return rewards[:, indices].float().cpu().tolist()     # [n, k]


def score_all(convs, tok, model, device, indices, batch_size):
    out = []
    for i in range(0, len(convs), batch_size):
        block = convs[i:i + batch_size]
        try:
            out.extend(score_block(block, tok, model, device, indices))
        except Exception as exc:
            if "Token pattern not found" not in str(exc):
                raise
            for conv in block:                            # gating truncated: retry per-item
                for mc in (4096, 2048, 1024, 512, 256, 0):
                    try:
                        out.extend(score_block([truncate_user(conv, mc)], tok, model, device, indices))
                        break
                    except Exception as e2:
                        if "Token pattern not found" not in str(e2):
                            raise
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_file", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--indices", required=True, help="comma-separated ArmoRM attribute indices")
    ap.add_argument("--names", required=True, help="comma-separated objective names, same order")
    ap.add_argument("--negate_indices", default="", help="comma-separated indices whose sign is flipped")
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--sample_batch_size", type=int, default=32)
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_index", type=int, default=0)
    args = ap.parse_args()

    indices = [int(x) for x in args.indices.split(",")]
    names = args.names.split(",")
    assert len(indices) == len(names), "indices and names length mismatch"
    negate = {int(x) for x in args.negate_indices.split(",") if x != ""}
    sign = [-1.0 if idx in negate else 1.0 for idx in indices]
    device = "cuda"

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=args.cache_dir, use_fast=True,
                                        local_files_only=args.local_files_only)
    tok.truncation_side = "right"
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, trust_remote_code=True,
        cache_dir=args.cache_dir, local_files_only=args.local_files_only).to(device).eval()

    rows = load_rows(args.input_file)
    rows = [r for i, r in enumerate(rows) if i % args.num_shards == args.shard_index]
    os.makedirs(args.output_dir, exist_ok=True)
    suffix = "" if args.num_shards == 1 else f".shard{args.shard_index}"
    outs = {n: open(os.path.join(args.output_dir, f"{args.split}_{n}.jsonl{suffix}"), "w", encoding="utf-8")
            for n in names}
    for block in tqdm([rows[i:i + args.sample_batch_size] for i in range(0, len(rows), args.sample_batch_size)]):
        convs, counts = [], []
        for r in block:
            resp = r["all_generated_responses"]
            counts.append(len(resp))
            for y in resp:
                convs.append([{"role": "user", "content": r["prompt"]},
                              {"role": "assistant", "content": y}])
        mat = score_all(convs, tok, model, device, indices, args.batch_size) if convs else []
        off = 0
        for r, c in zip(block, counts):
            rowscores = mat[off:off + c]; off += c
            for j, n in enumerate(names):
                rec = {"prompt": r["prompt"], "all_generated_responses": r["all_generated_responses"],
                       "all_rm_scores": [sign[j] * rowscores[i][j] for i in range(c)]}
                outs[n].write(json.dumps(rec, ensure_ascii=False) + "\n")
    for f in outs.values():
        f.close()
    print(f"[armo-multihead] wrote {len(names)} files to {args.output_dir} for split={args.split}")


if __name__ == "__main__":
    main()
