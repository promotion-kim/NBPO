"""Independent judging of the frozen 200-prompt diagnostic bank.

Per prompt: the eight learner occurrences against four reference occurrences
(32 ordered-pair sets) plus the six unordered reference pairs, each at four
rubrics and both presentation orders -- 304 rubric-order verdicts per prompt,
60,800 over the panel.

The judge contract is copied from code/judge_final_eval.py and not re-derived:
same template, same rubric file, temperature 0, 256 new tokens, thinking off,
ties 0.5, an unparsed verdict stays missing and keeps its place in the
denominator. Nothing here refits a target or selects a weight.

Output is one file per prompt, so the 10-prompt pilot's verdicts are part of the
panel rather than a throwaway, and a shard can resume without recomputing.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
import time
from pathlib import Path

ROOT = Path("/work/uf4_20260910")
CRITERIA = ("instruction_following", "truthfulness", "honesty", "helpfulness")
SEED_NAMESPACE = "20260914-uf4-bankdiag:"
VERDICT = re.compile(r"\[\[(A|B|TIE)\]\]", re.I)

TEMPLATE = """You are comparing two responses to the same user instruction.

{rubric}

Ignore which response is longer unless the instruction asked for a particular length. Ignore the order in which they are shown.

### INSTRUCTION
{instruction}

### RESPONSE A
{response_a}

### RESPONSE B
{response_b}

Answer with one line of reasoning, then your verdict on its own line in exactly this form: [[A]] if A is better, [[B]] if B is better, or [[TIE]] if they are equally good."""


def digest(t):
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def load_texts(pool, wanted):
    """candidate_id -> {prompt, response, response_sha256} for the ids we need."""
    out = {}
    prompts = {}
    for path in sorted(glob.glob(str(ROOT / "pools" / pool / "shard*/chunk*.jsonl"))):
        with open(path) as stream:
            for line in stream:
                r = json.loads(line)
                if r["candidate_id"] in wanted:
                    out[r["candidate_id"]] = {"response": r["response"],
                                              "response_sha256": r["response_sha256"]}
                    prompts[r["prompt_id"]] = r["prompt"]
    return out, prompts


def pairs_for(learners, references):
    """Canonical (left, right, kind) pairs: left wins are what we record."""
    out = []
    for i, y in enumerate(learners):
        for j, z in enumerate(references):
            out.append((("L%d" % i, y), ("R%d" % j, z), "learner_reference"))
    for a in range(len(references)):
        for b in range(a + 1, len(references)):
            out.append((("R%d" % a, references[a]), ("R%d" % b, references[b]),
                        "reference_reference"))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", default=str(ROOT / "analysis/diag_20260914/panel_dev200.json"))
    ap.add_argument("--out", default=str(ROOT / "analysis/diag_20260914/bank4"))
    ap.add_argument("--pool", default="dev_v1")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--first", type=int, default=0, help="only the first N panel prompts")
    ap.add_argument("--judge", default=str(ROOT / "assets/Qwen3-14B"))
    ap.add_argument("--judge-revision", default="40c069824f4251a91eefaf281ebe4c544efd3e18")
    ap.add_argument("--rubrics", default=str(ROOT / "audit/v1/rubrics.json"))
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--max-model-len", type=int, default=8192)
    args = ap.parse_args()

    panel = json.loads(Path(args.panel).read_text())
    ids = panel["panel_prompt_ids"]
    if args.first:
        ids = ids[:args.first]
    mine = [p for k, p in enumerate(ids) if k % args.nshards == args.shard]

    out_dir = Path(args.out) / "verdicts"
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [p for p in mine if not (out_dir / ("%s.done.json" % p)).exists()]
    print(json.dumps({"phase": "plan", "shard": args.shard, "of": args.nshards,
                      "assigned": len(mine), "todo": len(todo)}), flush=True)
    if not todo:
        return 0

    wanted = set()
    for pid in todo:
        wanted.update(panel["learner_occurrences"][pid])
        wanted.update(panel["reference_occurrences"][pid])
    texts, instructions = load_texts(args.pool, wanted)
    missing = sorted(wanted - set(texts))
    if missing:
        raise SystemExit("missing %d candidate texts, first %s" % (len(missing), missing[:2]))

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(args.judge, local_files_only=True)
    rubrics = json.loads(Path(args.rubrics).read_text())["rubrics"]
    llm = LLM(model=args.judge, tokenizer=args.judge, tensor_parallel_size=1, dtype="bfloat16",
              max_model_len=args.max_model_len, gpu_memory_utilization=0.90, max_num_seqs=64,
              generation_config="vllm", seed=20260914, trust_remote_code=False)

    started = time.monotonic()
    totals = {"ok": 0, "no_verdict": 0, "multiple_verdicts": 0, "truncated": 0, "verdicts": 0}
    for n, pid in enumerate(todo, 1):
        learners = panel["learner_occurrences"][pid]
        references = panel["reference_occurrences"][pid]
        instruction = instructions[pid]
        requests, params, meta = [], [], []
        for (lname, lid), (rname, rid), kind in pairs_for(learners, references):
            for criterion in CRITERIA:
                for order in (0, 1):
                    first, second = ((lid, rid) if order == 0 else (rid, lid))
                    text = TEMPLATE.format(rubric=rubrics[criterion], instruction=instruction,
                                           response_a=texts[first]["response"],
                                           response_b=texts[second]["response"])
                    tids = tok.apply_chat_template([{"role": "user", "content": text}],
                                                   tokenize=True, add_generation_prompt=True,
                                                   enable_thinking=False)
                    budget = args.max_model_len - args.max_tokens
                    truncated = len(tids) > budget
                    if truncated:
                        tids = tids[:budget]
                        totals["truncated"] += 1
                    seed = int(digest(SEED_NAMESPACE + "%s:%s:%s:%s:%d"
                                      % (pid, lname, rname, criterion, order))[:16], 16) % (2**63 - 1)
                    requests.append({"prompt_token_ids": tids})
                    params.append(SamplingParams(n=1, temperature=args.temperature,
                                                 max_tokens=args.max_tokens, seed=seed))
                    meta.append({"prompt_id": pid, "left": lname, "right": rname, "kind": kind,
                                 "left_candidate": lid, "right_candidate": rid,
                                 "left_sha256": texts[lid]["response_sha256"],
                                 "right_sha256": texts[rid]["response_sha256"],
                                 "criterion": criterion, "order": order,
                                 "left_first": order == 0, "seed": seed,
                                 "truncated": truncated})
        generated = llm.generate(requests, params, use_tqdm=False)
        rows = []
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
            totals[status] += 1
            totals["verdicts"] += 1
            rows.append({**m, "value_for_left": value, "status": status, "raw": text[-300:]})
        tmp = out_dir / ("%s.jsonl.tmp" % pid)
        with tmp.open("w") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        tmp.replace(out_dir / ("%s.jsonl" % pid))
        (out_dir / ("%s.done.json" % pid)).write_text(json.dumps({
            "prompt_id": pid, "verdicts": len(rows),
            "ok": sum(1 for r in rows if r["status"] == "ok"),
            "judge": args.judge, "judge_revision": args.judge_revision,
            "judge_thinking_mode": "disabled (enable_thinking=False)",
            "rubrics_sha256": digest(Path(args.rubrics).read_text()),
            "decoding": {"temperature": args.temperature, "max_tokens": args.max_tokens},
            "seed_namespace": SEED_NAMESPACE,
            "learner_occurrences": learners, "reference_occurrences": references,
        }, indent=2) + "\n")
        if n % 5 == 0 or n == len(todo):
            rate = totals["verdicts"] / max(time.monotonic() - started, 1e-9)
            print(json.dumps({"phase": "progress", "shard": args.shard, "prompts_done": n,
                              "of": len(todo), "verdicts": totals["verdicts"],
                              "verdicts_per_s": round(rate, 2),
                              "ok_rate": round(totals["ok"] / max(totals["verdicts"], 1), 4),
                              "eta_min_this_shard": round((len(todo) - n) * 304 / max(rate, 1e-9) / 60, 1)}),
                  flush=True)
    elapsed = time.monotonic() - started
    summary = {"phase": "done", "shard": args.shard, "of": args.nshards,
               "first": args.first, "prompts": len(todo),
               "seconds": round(elapsed, 1), **totals,
               "verdicts_per_s": round(totals["verdicts"] / max(elapsed, 1e-9), 2)}
    tag = "first%d" % args.first if args.first else "shard%dof%d" % (args.shard, args.nshards)
    done = Path(args.out) / ("complete_%s.json" % tag)
    done.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
