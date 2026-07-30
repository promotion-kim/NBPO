"""Build a minimal MultiObjectiveDataset-format dataset (prompt + placeholder
response + 5 dummy label columns) from our UltraFeedback test prompts, so
RMOD's eval.py can decode over them. Only `prompt` is used for generation.
"""
import argparse, json
from datasets import Dataset
from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts_jsonl", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--chat_template_model", default=None)
    args = ap.parse_args()
    tokenizer = None
    if args.chat_template_model:
        tokenizer = AutoTokenizer.from_pretrained(args.chat_template_model)
    prompts, seen = [], set()
    for line in open(args.prompts_jsonl, encoding="utf-8"):
        r = json.loads(line)
        p = r["prompt"] if isinstance(r.get("prompt"), str) else str(r.get("prompt"))
        if p and p not in seen:
            seen.add(p)
            if tokenizer:
                p = tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                if tokenizer.bos_token and p.startswith(tokenizer.bos_token):
                    p = p[len(tokenizer.bos_token):]
            prompts.append(p)
    # response is unused for decoding but must survive the >1-word filter
    rows = {"prompt": prompts, "response": ["placeholder response text unused"] * len(prompts)}
    for j in range(5):
        rows[f"obj{j}"] = [0.0] * len(prompts)
    Dataset.from_dict(rows).save_to_disk(args.output_dir)
    print(f"[uf-eval-ds] {len(prompts)} prompts -> {args.output_dir}")


if __name__ == "__main__":
    main()
