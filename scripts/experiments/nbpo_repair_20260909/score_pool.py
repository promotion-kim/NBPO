"""Score all three independent pools with the frozen calibrated GPM ensemble."""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from scripts.experiments.nbpo_repair_20260909.common import file_hash, read_jsonl, write_json, write_jsonl


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--ensemble", type=Path, default=Path("/work/iclr27_table1_v2/models/saferlhf_ensemble_ckpt"))
    ap.add_argument("--encoder", required=True, help="Pinned local roberta-base configuration; all weights loaded strictly from saved GPM state")
    args = ap.parse_args()
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    from mnpo_scripts.gpm import AntiSymmetricGPM
    pool_dir = args.root / "pools" / f"shard{args.shard}"
    if not (pool_dir / "complete.json").exists():
        raise ValueError("Generation shard is incomplete")
    by_prompt = defaultdict(dict)
    input_files = {}
    for file in sorted(pool_dir.glob("chunk*.jsonl")):
        expected = json.loads(file.with_suffix(".manifest.json").read_text())["sha256"]
        if file_hash(file) != expected:
            raise ValueError("Pool token artifact was changed")
        input_files[str(file)] = expected
        for row in read_jsonl(file):
            key = (row["role"], row["sample_index"])
            if key in by_prompt[row["prompt_id"]]:
                raise ValueError("Duplicate occurrence ID")
            by_prompt[row["prompt_id"]][key] = row
    pids = sorted(by_prompt)
    out = args.root / "scores" / f"shard{args.shard}"
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    roles = ("learner", "comparator", "reference_learner")
    for block in by_prompt.values():
        if set(block) != {(r, i) for r in roles for i in range(8)}:
            raise ValueError("Pool must contain exactly8+8+8 occurrences")
    calibration_path = args.ensemble / "saferlhf_ensemble.json"
    calibration = json.loads(calibration_path.read_text())["calibration"]["gpm"]
    probabilities = np.empty((3, 2, len(pids), 8, 8), dtype=np.float64)
    references = np.empty_like(probabilities)
    truncations = []
    models, timings = [], {}
    tokenizer_hashes = None
    device = torch.device("cuda")
    for si, seed in enumerate((41, 42, 43)):
        started = time.monotonic()
        ckpt = args.ensemble / f"ckpt_gpm_seed{seed}"
        blob = torch.load(ckpt / "model.pt", map_location="cpu", weights_only=False)
        if blob["kind"] != "gpm":
            raise ValueError("Unexpected teacher kind")
        tok = AutoTokenizer.from_pretrained(ckpt, local_files_only=True)
        if tok.padding_side != "right":
            raise ValueError("Frozen teacher last-token pooling requires right padding")
        this_hashes = {name: file_hash(ckpt / name) for name in ("tokenizer.json", "vocab.json", "merges.txt", "special_tokens_map.json")}
        if tokenizer_hashes is not None and this_hashes != tokenizer_hashes:
            raise ValueError("Teacher seed tokenizers differ; single-seed truncation ledger is not applicable")
        tokenizer_hashes = this_hashes
        encoder_config = AutoConfig.from_pretrained(args.encoder, local_files_only=True)
        if encoder_config.hidden_size != blob["hidden"] or encoder_config.model_type != "roberta":
            raise ValueError("Frozen encoder configuration mismatches checkpoint")
        enc = AutoModel.from_config(encoder_config)
        model = AntiSymmetricGPM(enc, blob["hidden"], 2, width=blob["width"], dropout=0.0)
        model.load_state_dict(blob["state_dict"], strict=True)
        model = model.to(device).eval()
        capacity = model.encoder.config.max_position_embeddings
        if capacity < 386:
            raise ValueError("Teacher encoder lacks required384-token context capacity")
        models.append({"seed": seed, "checkpoint_sha256": file_hash(ckpt / "model.pt"),
                       "encoder": args.encoder, "max_position_embeddings": capacity})
        for xi, pid in enumerate(pids):
            block = by_prompt[pid]
            rows = [block[(role, i)] for role in roles for i in range(8)]
            prompts, responses = [r["prompt"] for r in rows], [r["response"] for r in rows]
            batch = tok(prompts, responses, padding=True, truncation="longest_first", max_length=384, return_tensors="pt")
            if si == 0:
                plen = [len(v) for v in tok(prompts, add_special_tokens=False)["input_ids"]]
                rlen = [len(v) for v in tok(responses, add_special_tokens=False)["input_ids"]]
                for i, row in enumerate(rows):
                    sides = batch.sequence_ids(i)
                    kept_p, kept_r = sides.count(0), sides.count(1)
                    truncations.append({"candidate_id": row["candidate_id"], "prompt_id": pid,
                        "role": row["role"], "policy_response_tokens": row["n_tokens"],
                        "teacher_prompt_tokens": plen[i], "teacher_response_tokens": rlen[i],
                        "teacher_prompt_tokens_kept": kept_p, "teacher_response_tokens_kept": kept_r,
                        "prompt_truncated": kept_p < plen[i], "response_truncated": kept_r < rlen[i]})
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = model.encode(batch["input_ids"].to(device), batch["attention_mask"].to(device)).float()
            # Encoder follows the original frozen teacher precision; pair-head
            # probabilities and calibration are explicitly FP32/FP64.
            hl, hc, hr = h[:8], h[8:16], h[16:]
            for ki, name in enumerate(("helpfulness", "harmlessness")):
                temperature = float(calibration[str(seed)][name]["temperature"])
                if not np.isfinite(temperature) or not temperature > 0:
                    raise ValueError("Invalid frozen calibration temperature")
                cols = hc.unsqueeze(0).expand(8, 8, -1).reshape(64, -1)
                for tensor, left in ((probabilities, hl), (references, hr)):
                    lhs = left.unsqueeze(1).expand(8, 8, -1).reshape(64, -1)
                    raw = torch.sigmoid(model.logit(lhs, cols, ki)).double().cpu().numpy().reshape(8, 8)
                    if not np.isfinite(raw).all():
                        raise ValueError("Nonfinite frozen teacher probabilities")
                    raw = np.clip(raw, 1e-9, 1 - 1e-9)
                    logit = np.log(raw) - np.log1p(-raw)
                    tensor[si, ki, xi] = 1 / (1 + np.exp(-logit / temperature))
            if (xi + 1) % 100 == 0:
                print(json.dumps({"shard": args.shard, "seed": seed, "prompts": xi + 1,
                                  "seconds": round(time.monotonic() - started, 2)}), flush=True)
        timings[str(seed)] = time.monotonic() - started
        del model, enc, blob
        torch.cuda.empty_cache()
    tensor_path = out / "probabilities.npz"
    np.savez_compressed(tensor_path, policy=probabilities, reference=references)
    write_jsonl(out / "teacher_truncation.jsonl", truncations)
    write_json(out / "manifest.json", {
        "schema_version": "nbpo-repair-score-v1", "prompt_ids": pids,
        "prompt_splits": {pid: by_prompt[pid][("learner", 0)]["split"] for pid in pids},
        "objectives": ["helpfulness", "harmlessness"], "seeds": [41, 42, 43],
        "reference_construction": "independent_reference_learner8_against_comparator8",
        "reference_tensor_not_symmetrized": True,
        "calibration_sha256": file_hash(calibration_path), "calibration_applied_before_seed_average": True,
        "teacher_models": models, "pool_files": input_files, "probabilities_sha256": file_hash(tensor_path),
        "teacher_tokenizer_hashes": tokenizer_hashes,
        "encoder_config_sha256": file_hash(Path(args.encoder) / "config.json"),
        "encoder_initialization": "Pinned architecture, strict load of all saved trained GPM/encoder weights; no new pretrained model weights used",
        "shape": list(probabilities.shape), "max_teacher_tokens": 384,
        "truncation": {"n_occurrences": len(truncations),
            "prompt_fraction": sum(r["prompt_truncated"] for r in truncations) / len(truncations),
            "response_fraction": sum(r["response_truncated"] for r in truncations) / len(truncations)},
        "seconds_per_seed": timings, "source_sha256": file_hash(__file__),
    })
    print(f"Scored shard {args.shard}: {len(pids)} prompts, {sum(timings.values()):.1f}s", flush=True)


if __name__ == "__main__":
    main()
