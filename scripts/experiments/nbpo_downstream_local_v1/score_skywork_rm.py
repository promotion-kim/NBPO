#!/usr/bin/env python3
"""Skywork-RM proxy win rate against the base policy, on identical prompts.

This is an evaluation-only scalar reward model. It never touches training, the
teacher, or checkpoint selection, and its number is a PROXY: the official
AlpacaEval-2 and Arena-Hard scores use a GPT-4 judge with length control and a
specific reference, none of which this reproduces. The metric is named for what
it is wherever it is reported.

Following the model card: the RM's own tokenizer and chat template, no system
prompt, sequence classification with a single logit, bf16, eval and no_grad.
A tie is exact score equality -- no epsilon is introduced to manufacture ties.
Identical response strings reuse one cached score, so a candidate that copies the
base is scored as a tie rather than as a coin flip.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rm", required=True)
    ap.add_argument("--responses-dir", required=True)
    ap.add_argument("--base-label", default="base")
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--benchmarks", nargs="+", default=["alpaca_eval", "arena_hard"])
    ap.add_argument("--max-length", type=int, default=16384)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.rm)
    # the evaluation environment pins transformers 4.55 for vLLM compatibility,
    # where the argument is torch_dtype; 5.x renamed it to dtype
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.rm, torch_dtype=torch.bfloat16, num_labels=1)
    except TypeError:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.rm, dtype=torch.bfloat16, num_labels=1)
    model = model.cuda().eval()

    cache: dict[tuple[str, str], float] = {}

    @torch.no_grad()
    def score(prompt: str, answer: str) -> float | None:
        key = (prompt, answer)
        if key in cache:
            return cache[key]
        # the card's format: user/assistant only, no system message
        conv = [{"role": "user", "content": prompt},
                {"role": "assistant", "content": answer}]
        text = tok.apply_chat_template(conv, tokenize=False)
        ids = tok(text, return_tensors="pt", truncation=False)
        if ids["input_ids"].shape[1] > args.max_length:
            cache[key] = None          # reported as uncovered, never truncated silently
            return None
        ids = {k: v.cuda() for k, v in ids.items()}
        v = float(model(**ids).logits[0][0].item())
        cache[key] = v
        return v

    R = Path(args.responses_dir)
    report = {"reward_model": args.rm, "base_label": args.base_label,
              "metric": ("proxy win rate = mean[1(r_cand > r_base) + 0.5*1(r_cand == "
                         "r_base)]; exact equality is the only tie"),
              "not_official": ("this is not AlpacaEval-2 LC and not the official "
                               "Arena-Hard score; both need a GPT-4 judge"),
              "results": {}}

    for bench in args.benchmarks:
        base_rows = {r["uid"]: r for r in
                     (json.loads(l) for l in (R / args.base_label / f"{bench}.jsonl").open())}
        for lbl in args.labels:
            cand_rows = {r["uid"]: r for r in
                         (json.loads(l) for l in (R / lbl / f"{bench}.jsonl").open())}
            uids = [u for u in base_rows if u in cand_rows]
            wins = ties = 0.0
            scored = 0
            uncovered = 0
            per = []
            for u in uids:
                b, c = base_rows[u], cand_rows[u]
                rb = score(b["prompt"], b["output"])
                rc = score(c["prompt"], c["output"])
                if rb is None or rc is None:
                    uncovered += 1
                    continue
                scored += 1
                if rc > rb:
                    wins += 1
                elif rc == rb:
                    ties += 1
                per.append({"uid": u, "r_base": rb, "r_cand": rc,
                            "n_tok_base": b["n_output_tokens"],
                            "n_tok_cand": c["n_output_tokens"]})
            wr = (wins + 0.5 * ties) / scored if scored else None
            report["results"].setdefault(bench, {})[lbl] = {
                "n_planned": len(base_rows), "n_common": len(uids),
                "n_scored": scored, "n_uncovered_over_max_length": uncovered,
                "wins": wins, "ties": ties, "proxy_win_rate": wr}
            (args.out.parent / f"rm_scores_{bench}_{lbl}.jsonl").write_text(
                "\n".join(json.dumps(x) for x in per) + "\n")
            print(f"{bench:>12} {lbl:>12}  n={scored:>4}  proxy WR={wr:.4f}  "
                  f"ties={int(ties)}  uncovered={uncovered}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
