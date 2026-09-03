#!/usr/bin/env python3
"""Decode a candidate checkpoint on the held-out monitoring prompts, synchronously.

Replaces the legacy ``scripts/bpo/decode_eval.sh`` in the NBPO path: that
script takes ``ROOT arm:seed:gpu``, hard-codes internal paths and a model
directory, launches detached tmux jobs, and returns before anything finishes.
This CLI decodes directly from ``--model-checkpoint``, waits for every seed,
validates every output file, and writes a manifest that cryptographically binds
the responses to the checkpoint that produced them:

- content fingerprint of the candidate checkpoint
  (``mnpo_scripts.precompute_provenance.checkpoint_fingerprint``: every weight
  shard, index, config and tokenizer file);
- sha256 of the exact prompt file and the exact sorted prompt IDs;
- generation seeds and decode parameters;
- per-seed output paths and their sha256.

``run_nbpo_stage.py`` derives the monitoring responses ONLY from this manifest
in real mode and re-verifies each binding before the Algorithm-1 gate.

Backends: ``vllm`` (real) and ``mock`` (deterministic, LLM-free; used by the
orchestration tests -- it still fingerprints whatever checkpoint it is given).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from mnpo_scripts.precompute_provenance import checkpoint_fingerprint, sha256_file_hex
from scripts.nbpo.nbpo_common import SCHEMA_VERSION, stable_hash_int

MANIFEST_SCHEMA = 1


def load_prompts(path: Path):
    """(prompt_id, text) pairs from a jsonl of {prompt[, prompt_id]}; ids default to line index."""
    rows = []
    for i, line in enumerate(path.read_text().splitlines()):
        if not line.strip():
            continue
        d = json.loads(line)
        rows.append((str(d.get("prompt_id", i)), d["prompt"]))
    if len({pid for pid, _ in rows}) != len(rows):
        raise ValueError(f"duplicate prompt ids in {path}")
    return rows


class MockDecoder:
    def __init__(self, checkpoint: str):
        self.tag = checkpoint_fingerprint(checkpoint)[:8]

    def generate(self, prompts, seed, temperature, top_p, max_new_tokens):
        return [f"[mock {self.tag} seed={seed}] "
                f"{stable_hash_int(self.tag, str(seed), text) % 100000}: {text[:40]}"
                for text in prompts]


class VllmDecoder:
    def __init__(self, checkpoint: str, tp: int, max_model_len: int):
        from vllm import LLM, SamplingParams  # lazy

        self._SP = SamplingParams
        self.llm = LLM(model=checkpoint, dtype="bfloat16", max_model_len=max_model_len,
                       gpu_memory_utilization=0.85, tensor_parallel_size=tp,
                       trust_remote_code=False)
        self.tok = self.llm.get_tokenizer()

    def generate(self, prompts, seed, temperature, top_p, max_new_tokens):
        rendered = [self.tok.apply_chat_template([{"role": "user", "content": p}],
                                                 tokenize=False, add_generation_prompt=True)
                    for p in prompts]
        outs = self.llm.generate(rendered, self._SP(n=1, temperature=temperature, top_p=top_p,
                                                    max_tokens=max_new_tokens, seed=seed),
                                 use_tqdm=False)
        return [o.outputs[0].text for o in outs]


def decode_candidate(model_checkpoint: Path, prompts_file: Path, seeds, temperature: float,
                     top_p: float, max_new_tokens: int, output_dir: Path, manifest: Path,
                     backend: str = "vllm", tp: int = 1, max_model_len: int = 2048,
                     tag: str = "policy") -> dict:
    fingerprint = checkpoint_fingerprint(str(model_checkpoint))
    prompts = load_prompts(prompts_file)
    prompt_ids = sorted(pid for pid, _ in prompts)
    texts = [t for _, t in prompts]
    dec = (MockDecoder(str(model_checkpoint)) if backend == "mock"
           else VllmDecoder(str(model_checkpoint), tp, max_model_len))
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for seed in seeds:                       # synchronous: every seed finishes here
        gens = dec.generate(texts, seed, temperature, top_p, max_new_tokens)
        if len(gens) != len(texts):
            raise RuntimeError(f"seed {seed}: decoder returned {len(gens)} of {len(texts)} outputs")
        rows = [{"prompt_id": pid, "prompt": text, "generated_text": g, "seed": f"s{seed}"}
                for (pid, text), g in zip(prompts, gens)]
        for r in rows:
            if not isinstance(r["generated_text"], str):
                raise RuntimeError(f"seed {seed}: non-string generation for prompt {r['prompt_id']}")
        path = output_dir / f"{tag}_s{seed}.json"
        path.write_text(json.dumps(rows, ensure_ascii=False))
        # validate the file we just wrote: exact prompt-id set, one row per prompt
        back = json.loads(path.read_text())
        if sorted(r["prompt_id"] for r in back) != prompt_ids:
            raise RuntimeError(f"seed {seed}: written file does not cover the prompt set exactly")
        outputs[f"s{seed}"] = {"path": str(path), "sha256": sha256_file_hex(str(path)),
                               "n_rows": len(back)}
    man = {
        "schema_version": MANIFEST_SCHEMA,
        "nbpo_schema": SCHEMA_VERSION,
        "candidate_checkpoint": str(model_checkpoint),
        "candidate_fingerprint": fingerprint,
        "prompts_file": str(prompts_file),
        "prompts_file_sha256": sha256_file_hex(str(prompts_file)),
        "prompt_ids": prompt_ids,
        "n_prompts": len(prompt_ids),
        "seeds": [int(s) for s in seeds],
        "decode_params": {"temperature": temperature, "top_p": top_p,
                          "max_new_tokens": max_new_tokens, "backend": backend,
                          "tensor_parallel_size": tp, "max_model_len": max_model_len},
        "outputs": outputs,
        "synchronous": True,
    }
    man["manifest_sha256_of_outputs"] = hashlib.sha256(
        json.dumps(outputs, sort_keys=True).encode()).hexdigest()
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(man, indent=2, ensure_ascii=False) + "\n")
    return man


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-checkpoint", type=Path, required=True)
    ap.add_argument("--prompts-file", type=Path, required=True)
    ap.add_argument("--seeds", required=True, help="comma-separated generation seeds")
    ap.add_argument("--temperature", type=float, required=True)
    ap.add_argument("--top-p", type=float, required=True)
    ap.add_argument("--max-new-tokens", type=int, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--backend", choices=["vllm", "mock"], default="vllm")
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--tag", default="policy")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    man = decode_candidate(a.model_checkpoint, a.prompts_file, seeds, a.temperature, a.top_p,
                           a.max_new_tokens, a.output_dir, a.manifest, backend=a.backend,
                           tp=a.tensor_parallel_size, max_model_len=a.max_model_len, tag=a.tag)
    print(json.dumps({"candidate_fingerprint": man["candidate_fingerprint"],
                      "n_prompts": man["n_prompts"], "seeds": man["seeds"],
                      "manifest": str(a.manifest)}, indent=1))


if __name__ == "__main__":
    main()
