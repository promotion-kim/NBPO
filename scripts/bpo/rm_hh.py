#!/usr/bin/env python3
"""Score HH-RLHF responses with a reward model.
  --kind reward : Ray2333 gpt2-large helpful/harmless RM, scalar reward = logits[:,0]
  --kind humor  : humor classifier, reward = P(humorous) = softmax(logits)[:,humor_idx]
Input rows need `prompt` + `all_generated_responses`; output rows carry `all_rm_scores`
(one per response), matching the Beaver/ArmoRM scorer schema so the rest of the
multi-objective pipeline is unchanged."""
import argparse, json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_file", required=True)
    ap.add_argument("--output_file", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--kind", choices=["reward", "humor"], required=True)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_length", type=int, default=1024)
    args = ap.parse_args()
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16).to(dev).eval()
    model.config.pad_token_id = tok.pad_token_id
    # cap to the model's positional limit (e.g. DistilBERT humor classifier = 512)
    mpe = getattr(model.config, "max_position_embeddings", None)
    if mpe:
        args.max_length = min(args.max_length, int(mpe))
    humor_idx = 1
    if args.kind == "humor":
        lab = {str(v).lower(): int(k) for k, v in (model.config.id2label or {}).items()}
        humor_idx = next((i for n, i in lab.items() if "humor" in n and "no" not in n), 1)

    rows = [json.loads(l) for l in open(args.input_file) if l.strip()]
    with open(args.output_file, "w") as out:
        for r in rows:
            prompt = str(r["prompt"]); resps = [str(x) for x in r["all_generated_responses"]]
            texts = ["\n\nHuman: " + prompt + "\n\nAssistant: " + x if args.kind == "reward" else x
                     for x in resps]
            scores = []
            for i in range(0, len(texts), args.batch_size):
                enc = tok(texts[i:i + args.batch_size], return_tensors="pt", padding=True,
                          truncation=True, max_length=args.max_length).to(dev)
                with torch.no_grad():
                    logits = model(**enc).logits
                if args.kind == "reward":
                    scores += logits[:, 0].float().cpu().tolist()
                else:
                    scores += torch.softmax(logits, -1)[:, humor_idx].float().cpu().tolist()
            out.write(json.dumps({"prompt": prompt, "all_generated_responses": resps,
                                  "all_rm_scores": scores}, ensure_ascii=False) + "\n")
    print(f"[rm_hh:{args.kind}] wrote {len(rows)} rows -> {args.output_file}")


if __name__ == "__main__":
    main()
