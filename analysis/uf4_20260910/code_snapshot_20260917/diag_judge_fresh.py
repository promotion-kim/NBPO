"""Judge one arm's fresh responses against the same four reference occurrences.

Per prompt: one fresh response against four references, four rubrics, both
orders -- 32 verdicts per prompt, 6,400 per arm. Contract, template, parser and
missing-data rule are imported from the bank driver so the two panels are
scored identically; only the left-hand candidate differs.

One draw per prompt does not estimate a stochastic policy's game value, so this
driver reports direct wins only and writes no surplus.
"""
from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

from diag_judge_bank import (CRITERIA, TEMPLATE, VERDICT, digest, load_texts)

ROOT = Path("/work/uf4_20260910")
SEED_NAMESPACE = "20260914-uf4-freshjudge:"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--fresh", default=None, help="defaults to the fresh dir for this arm")
    ap.add_argument("--panel", default=str(ROOT / "analysis/diag_20260914/panel_dev200.json"))
    ap.add_argument("--out", default=str(ROOT / "analysis/diag_20260914/fresh_verdicts"))
    ap.add_argument("--pool", default="dev_v1")
    ap.add_argument("--judge", default=str(ROOT / "assets/Qwen3-14B"))
    ap.add_argument("--judge-revision", default="40c069824f4251a91eefaf281ebe4c544efd3e18")
    ap.add_argument("--rubrics", default=str(ROOT / "audit/v1/rubrics.json"))
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--max-model-len", type=int, default=8192)
    args = ap.parse_args()

    fresh_path = Path(args.fresh or (ROOT / "analysis/diag_20260914/fresh" / ("%s.jsonl" % args.arm)))
    fresh = {}
    with fresh_path.open() as stream:
        for line in stream:
            r = json.loads(line)
            fresh[r["prompt_id"]] = r

    panel = json.loads(Path(args.panel).read_text())
    ids = [p for p in panel["panel_prompt_ids"] if p in fresh]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    done = out_dir / ("complete_%s.json" % args.arm)
    if done.exists():
        print(json.dumps({"skipped": "already judged", "path": str(done)}))
        return 0

    wanted = set()
    for pid in ids:
        wanted.update(panel["reference_occurrences"][pid])
    texts, instructions = load_texts(args.pool, wanted)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(args.judge, local_files_only=True)
    rubrics = json.loads(Path(args.rubrics).read_text())["rubrics"]
    llm = LLM(model=args.judge, tokenizer=args.judge, tensor_parallel_size=1, dtype="bfloat16",
              max_model_len=args.max_model_len, gpu_memory_utilization=0.90, max_num_seqs=64,
              generation_config="vllm", seed=20260914, trust_remote_code=False)

    requests, params, meta = [], [], []
    truncated = 0
    for pid in ids:
        instruction = instructions[pid]
        left = fresh[pid]["response"]
        for j, rid in enumerate(panel["reference_occurrences"][pid]):
            for criterion in CRITERIA:
                for order in (0, 1):
                    a, b = ((left, texts[rid]["response"]) if order == 0
                            else (texts[rid]["response"], left))
                    text = TEMPLATE.format(rubric=rubrics[criterion], instruction=instruction,
                                           response_a=a, response_b=b)
                    tids = tok.apply_chat_template([{"role": "user", "content": text}],
                                                   tokenize=True, add_generation_prompt=True,
                                                   enable_thinking=False)
                    budget = args.max_model_len - args.max_tokens
                    if len(tids) > budget:
                        tids = tids[:budget]
                        truncated += 1
                    seed = int(digest(SEED_NAMESPACE + "%s:%s:R%d:%s:%d"
                                      % (args.arm, pid, j, criterion, order))[:16], 16) % (2**63 - 1)
                    requests.append({"prompt_token_ids": tids})
                    params.append(SamplingParams(n=1, temperature=args.temperature,
                                                 max_tokens=args.max_tokens, seed=seed))
                    meta.append({"prompt_id": pid, "left": "FRESH", "right": "R%d" % j,
                                 "kind": "fresh_reference", "right_candidate": rid,
                                 "left_sha256": fresh[pid]["response_sha256"],
                                 "right_sha256": texts[rid]["response_sha256"],
                                 "criterion": criterion, "order": order,
                                 "left_first": order == 0, "seed": seed})

    print(json.dumps({"phase": "start", "arm": args.arm, "prompts": len(ids),
                      "verdicts": len(requests), "truncated": truncated}), flush=True)
    started = time.monotonic()
    generated = llm.generate(requests, params, use_tqdm=False)
    elapsed = time.monotonic() - started

    counts = {"ok": 0, "no_verdict": 0, "multiple_verdicts": 0}
    dest = out_dir / ("%s.jsonl" % args.arm)
    tmp = dest.with_suffix(".jsonl.tmp")
    with tmp.open("w") as stream:
        for m, result in zip(meta, generated):
            text = result.outputs[0].text
            found = VERDICT.findall(text)
            if len(found) != 1:
                value, status = None, ("no_verdict" if not found else "multiple_verdicts")
            else:
                verdict = found[0].upper()
                if verdict == "TIE":
                    value = 0.5
                elif m["left_first"]:
                    value = 1.0 if verdict == "A" else 0.0
                else:
                    value = 0.0 if verdict == "A" else 1.0
                status = "ok"
            counts[status] += 1
            stream.write(json.dumps({**m, "value_for_left": value, "status": status,
                                     "raw": text[-300:]}, ensure_ascii=False) + "\n")
    tmp.replace(dest)
    summary = {"arm": args.arm, "prompts": len(ids), "verdicts": len(meta),
               "truncated": truncated, **counts,
               "judge": args.judge, "judge_revision": args.judge_revision,
               "judge_thinking_mode": "disabled (enable_thinking=False)",
               "rubrics_sha256": digest(Path(args.rubrics).read_text()),
               "decoding": {"temperature": args.temperature, "max_tokens": args.max_tokens},
               "seed_namespace": SEED_NAMESPACE,
               "seconds": round(elapsed, 1),
               "verdicts_per_s": round(len(meta) / max(elapsed, 1e-9), 2),
               "surplus": "n/a: one stochastic draw per prompt is not a policy game value"}
    done.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
