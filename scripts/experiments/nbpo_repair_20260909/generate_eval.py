"""Matched greedy benchmark generation with raw tokens and complete coverage."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from scripts.experiments.nbpo_downstream_local_v1.generate_panel import build
from scripts.experiments.nbpo_repair_20260909.common import (
    digest, file_hash, object_hash, read_jsonl, write_json, write_jsonl,
)


def items_for(bench, root):
    if bench == "xstest":
        with (root / "data/xstest_prompts.csv").open() as stream:
            return [(f"xstest:{r['id']}", r["prompt"], r) for r in csv.DictReader(stream)]
    if bench == "saferlhf":
        return [(f"saferlhf:{r['prompt_id']}", r["prompt"], {"split": r["split"]})
                for s in ("dev", "test") for r in read_jsonl(root / "splits" / f"{s}.jsonl")]
    return build(bench, root / "data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--tokenizer", default="/work/models/bases/Llama-3.1-8B-Instruct")
    ap.add_argument("--benchmarks", nargs="+", default=["ifeval", "gsm8k", "harmbench", "xstest", "alpaca_eval", "arena_hard", "saferlhf"])
    ap.add_argument("--max-model-len", type=int, default=16384)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    end = json.loads((Path(args.tokenizer) / "generation_config.json").read_text())["eos_token_id"]
    end = end if isinstance(end, list) else [end]
    out = args.root / "responses" / args.label
    out.mkdir(parents=True, exist_ok=True)
    model_files = {p.name: file_hash(p) for p in sorted(Path(args.model).glob("*.safetensors"))}
    if not model_files:
        raise ValueError("Evaluation requires a concrete exported full-policy checkpoint")
    settings = {"label": args.label, "model": args.model, "model_weights": model_files,
        "tokenizer": args.tokenizer, "tokenizer_sha256": file_hash(Path(args.tokenizer) / "tokenizer.json"),
        "chat_template_sha256": digest(tok.chat_template), "native_user_only": True,
        "temperature": 0.0, "top_p": 1.0, "top_k": -1, "repetition_penalty": 1.0,
        "max_new_tokens": 2048, "seed": 20260909, "terminal_ids": end,
        "max_model_len": args.max_model_len, "benchmarks": args.benchmarks,
        "source_sha256": file_hash(__file__), "benchmark_builder_sha256": file_hash(Path(__file__).parents[1] / "nbpo_downstream_local_v1/generate_panel.py")}
    settings_path = out / "settings.json"
    if settings_path.exists():
        if json.loads(settings_path.read_text()) != settings:
            raise ValueError("Existing response run has different settings")
    else:
        write_json(settings_path, settings)
    llm = None
    for bench in args.benchmarks:
        target = out / f"{bench}.jsonl"
        summary = out / f"{bench}.manifest.json"
        if target.exists():
            if not summary.exists() or json.loads(summary.read_text())["response_file_sha256"] != file_hash(target):
                raise ValueError("Existing responses incomplete/corrupted")
            continue
        items = items_for(bench, args.root)
        if not items or len({uid for uid, _, _ in items}) != len(items):
            raise ValueError("Empty/duplicate benchmark IDs")
        pids = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=True, add_generation_prompt=True) for _, p, _ in items]
        if any(len(ids) + 2048 > args.max_model_len for ids in pids):
            raise ValueError(f"Benchmark {bench} exceeds context: increase shared context for all arms, do not truncate prompts")
        if llm is None:
            llm = LLM(model=args.model, tokenizer=args.tokenizer, dtype="bfloat16",
                max_model_len=args.max_model_len, gpu_memory_utilization=0.85,
                generation_config="vllm", seed=20260909, tensor_parallel_size=1)
        sp = SamplingParams(temperature=0.0, top_p=1.0, top_k=-1, repetition_penalty=1.0,
            max_tokens=2048, stop_token_ids=end, seed=20260909, n=1)
        started = time.monotonic()
        outputs = llm.generate([{"prompt_token_ids": ids} for ids in pids], sp, use_tqdm=False)
        if len(outputs) != len(items) or any(len(g.outputs) != 1 for g in outputs):
            raise ValueError("Generation coverage mismatch")
        rows = []
        for (uid, prompt, meta), prompt_ids, output in zip(items, pids, outputs):
            g = output.outputs[0]
            ids = list(g.token_ids)
            if not ids or g.finish_reason not in ("stop", "length"):
                raise ValueError("Invalid generated event")
            if g.finish_reason == "stop" and ids[-1] not in end:
                raise ValueError("Sampled terminal token absent from event")
            rows.append({"uid": uid, "prompt": prompt, "output": g.text,
                "prompt_sha256": digest(prompt), "prompt_token_ids": prompt_ids,
                "output_token_ids": ids, "n_output_tokens": len(ids),
                "finish_reason": g.finish_reason, "stop_reason": g.stop_reason,
                "meta": meta, "model_sha256": object_hash(model_files)})
        write_jsonl(target, rows)
        write_json(summary, {"n_planned": len(items), "n_generated": len(rows), "n_failed": 0,
            "seconds": time.monotonic() - started, "response_file_sha256": file_hash(target),
            "settings_sha256": file_hash(settings_path), "prompt_sha256": object_hash([(u, p) for u, p, _ in items]),
            "tokens": sum(r["n_output_tokens"] for r in rows),
            "capped": sum(r["finish_reason"] == "length" for r in rows),
            "empty_text": sum(not r["output"].strip() for r in rows)})
        print(f"{args.label}/{bench}: {len(rows)} responses in {time.monotonic()-started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
