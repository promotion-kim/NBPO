#!/usr/bin/env python3
"""What each sweep arm actually consumed: updates, epochs, tokens, GPU time.

"Epochs" was quoted from a nominal effective batch. This reads the realized
numbers instead -- per-device batch and accumulation from the run config, wall
time from the trainer state, and response-token counts from the dataset the arm
trained on -- because arms with different response lengths do not cost the same
per update, so equal update budgets are not equal compute.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+", required=True,
                    help="label=<run_config.yaml>=<arm dir>=<precomputed dir>")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from datasets import load_from_disk
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    rows = []
    for spec in args.arms:
        label, cfg_path, arm_dir, data = spec.split("=")
        cfg = yaml.safe_load(Path(cfg_path).read_text())
        st = json.loads((Path(arm_dir) / "trainer_state.json").read_text())
        # train_runtime lives in train_results.json, not the trainer state
        tr = Path(arm_dir) / "train_results.json"
        if tr.exists():
            st.update(json.loads(tr.read_text()))
        ds = load_from_disk(data)["train"]
        n_rows = ds.num_rows
        n_prompts = len(set(ds["prompt_id"]))
        # response tokens per row: both sides of the pair are forwarded
        sample = ds.select(range(min(400, n_rows)))
        tl = [len(tok(c, add_special_tokens=False)["input_ids"])
              + len(tok(r, add_special_tokens=False)["input_ids"])
              for c, r in zip(sample["chosen"], sample["rejected"])]
        mean_tokens = sum(tl) / len(tl)
        eff = int(cfg["per_device_train_batch_size"]) * int(cfg["gradient_accumulation_steps"])
        updates = int(cfg["max_steps"])
        examples = updates * eff
        rows.append({
            "arm": label, "train_prompts": n_prompts, "train_pair_rows": n_rows,
            "per_device_batch": int(cfg["per_device_train_batch_size"]),
            "grad_accum": int(cfg["gradient_accumulation_steps"]),
            "effective_batch": eff, "optimizer_updates": updates,
            "examples_consumed": examples,
            "exposure_epochs": examples / n_rows,
            "mean_response_tokens_per_pair": mean_tokens,
            "approx_response_tokens_consumed": examples * mean_tokens,
            "train_runtime_seconds": st.get("train_runtime"),
            "gpu_hours": (st.get("train_runtime") or 0) / 3600.0,
            "online_reference": bool(cfg.get("nbpo_online_reference", False)),
            "learning_rate": cfg["learning_rate"], "seed": cfg["seed"],
            "solver_artifact_sha256": cfg.get("nbpo_expected_solver_artifact_sha256"),
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2) + "\n")
    hdr = (f"{'arm':>18} {'prompts':>8} {'rows':>7} {'eff.b':>6} {'upd':>5} "
           f"{'epochs':>7} {'tok/pair':>9} {'Mtokens':>8} {'gpu h':>6} {'onlineref':>10}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['arm']:>18} {r['train_prompts']:>8} {r['train_pair_rows']:>7} "
              f"{r['effective_batch']:>6} {r['optimizer_updates']:>5} "
              f"{r['exposure_epochs']:>7.2f} {r['mean_response_tokens_per_pair']:>9.1f} "
              f"{r['approx_response_tokens_consumed']/1e6:>8.1f} {r['gpu_hours']:>6.2f} "
              f"{str(r['online_reference']):>10}")


if __name__ == "__main__":
    main()
