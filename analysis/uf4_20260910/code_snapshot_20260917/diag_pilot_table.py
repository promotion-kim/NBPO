"""Fit and realization metrics for the two projection-pilot checkpoints.

For each pilot arm this scores the frozen train and development panels under the
pilot checkpoint and under the proximal centre pi_t, forms the pair log-ratio
h over every prompt's 28 unordered pairs, and reports the quantities the pilot
table declares: train and development nMSE against the same zero-predictor
denominator, the pool KL to the exact target, target and prediction RMS, sign
accuracy at a frozen zero threshold, and total variation.

nMSE is E[(h-h*)^2] / E[h*^2] on the same pair measure, so the zero predictor
scores exactly 1 and the numbers are comparable between the arms and between
the splits. Fresh win rates are not computed here; they come from the same
independent judge the rest of the panel uses.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch

from diag_neural_pools import load_pool_rows, sequence_logprobs

ROOT = Path("/work/uf4_20260910")
DIAG = ROOT / "analysis/diag_20260914"
PAIRS = [(i, j) for i in range(8) for j in range(i + 1, 8)]
FLOOR = 1e-12
ARMS = [("sampled", "diag_proj_sampled_s42"), ("all", "diag_proj_all_s42")]


def panel_candidates(panel_file, pool):
    """prompt -> the eight learner candidate ids, from the frozen panel or the pool."""
    panel = json.loads(Path(panel_file).read_text())
    ids = panel["panel_prompt_ids"]
    if "learner_occurrences" in panel:
        return ids, {pid: panel["learner_occurrences"][pid] for pid in ids}
    out = {}
    for path in sorted(glob.glob(str(ROOT / "pools" / pool / "shard*/chunk*.jsonl"))):
        with open(path) as stream:
            for line in stream:
                r = json.loads(line)
                if r["prompt_id"] in set(ids) and r["role"] == "learner":
                    out.setdefault(r["prompt_id"], {})[r["sample_index"]] = r["candidate_id"]
    return ids, {pid: [out[pid][k] for k in sorted(out[pid])] for pid in ids if pid in out}


def pair_vector(values):
    return np.array([values[i] - values[j] for i, j in PAIRS])


def metrics(ids, candidates, deltas_arm, deltas_base, target, index, pi_star, pi_t):
    h_list, hstar_list, kl, tv = [], [], [], []
    for pid in ids:
        if pid not in candidates or pid not in index:
            continue
        cs = candidates[pid]
        delta = np.array([deltas_arm[c] - deltas_base[c] for c in cs], dtype=np.float64)
        k = index[pid]
        star = np.clip(pi_star[k], FLOOR, None)
        src = np.clip(pi_t[k], FLOOR, None)
        h_list.append(pair_vector(delta))
        hstar_list.append(pair_vector(np.log(star) - np.log(src)))
        logits = np.log(src) + delta
        logits -= logits.max()
        p = np.exp(logits)
        p /= p.sum()
        kl.append(float(np.sum(star * (np.log(star) - np.log(np.clip(p, FLOOR, None))))))
        tv.append(float(0.5 * np.abs(p - star).sum()))
    h = np.concatenate(h_list)
    hs = np.concatenate(hstar_list)
    return {
        "prompts": len(h_list), "pairs": int(h.size),
        "nmse": float(np.mean((h - hs) ** 2) / np.mean(hs ** 2)),
        "zero_predictor_mse": float(np.mean(hs ** 2)),
        "mse": float(np.mean((h - hs) ** 2)),
        "target_rms": float(np.sqrt(np.mean(hs ** 2))),
        "prediction_rms": float(np.sqrt(np.mean(h ** 2))),
        "sign_accuracy": float(np.mean(np.sign(h) == np.sign(hs))),
        "pearson": float(np.corrcoef(h, hs)[0, 1]),
        "pool_kl_star_to_fitted": float(np.mean(kl)) if kl else None,
        "tv_star_to_fitted": float(np.mean(tv)) if tv else None,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="/work/models/bases/Llama-3.1-8B-Instruct")
    ap.add_argument("--target", default="nash_v1")
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()

    dev_ids, dev_cands = panel_candidates(DIAG / "panel_dev200.json", "dev_v1")
    train_ids, train_cands = panel_candidates(DIAG / "panel_train200.json", "v1")
    rows = {}
    rows["dev"] = load_pool_rows("dev_v1", {c for v in dev_cands.values() for c in v})
    rows["train"] = load_pool_rows("v1", {c for v in train_cands.values() for c in v})

    from transformers import AutoModelForCausalLM
    device = "cuda"

    def score(path):
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16,
                                                     attn_implementation="sdpa",
                                                     local_files_only=True).to(device).eval()
        out = {}
        for split, cands in (("dev", dev_cands), ("train", train_cands)):
            order = [c for pid in (dev_ids if split == "dev" else train_ids)
                     if pid in cands for c in cands[pid]]
            values = sequence_logprobs(model, rows[split], order, device, args.batch)
            out[split] = dict(zip(order, values))
        del model
        torch.cuda.empty_cache()
        return out

    base = score(args.base)
    index, pi_star, pi_t = {}, None, None
    meta = json.loads((ROOT / "targets" / args.target / "dev/tensor/meta.json").read_text())
    dev_index = {pid: k for k, pid in enumerate(meta["prompt_ids"])}
    dev_star = np.load(ROOT / "targets" / args.target / "dev/solver/pi_star.npz")["pi"]
    dev_src = np.load(ROOT / "targets" / args.target / "dev/solver/pi_t.npz")["pi"]
    tmeta = json.loads((ROOT / "targets" / args.target / "train/tensor/meta.json").read_text())
    tr_index = {pid: k for k, pid in enumerate(tmeta["prompt_ids"])}
    tr_star = np.load(ROOT / "targets" / args.target / "train/solver/pi_star.npz")["pi"]
    tr_src = np.load(ROOT / "targets" / args.target / "train/solver/pi_t.npz")["pi"]

    table = {}
    for label, arm in ARMS:
        arm_dir = ROOT / "arms" / arm
        if not (arm_dir / "config.json").exists():
            print(json.dumps({"skipped": arm, "reason": "checkpoint not exported yet"}))
            continue
        scored = score(str(arm_dir))
        dev = metrics(dev_ids, dev_cands, scored["dev"], base["dev"], args.target,
                      dev_index, dev_star, dev_src)
        train = metrics(train_ids, train_cands, scored["train"], base["train"], args.target,
                        tr_index, tr_star, tr_src)
        state = json.loads((arm_dir / "trainer_state.json").read_text())
        results = json.loads((arm_dir / "all_results.json").read_text())
        table[label] = {
            "arm": arm,
            "steps": state.get("global_step"),
            "train_nmse": train["nmse"], "dev_nmse": dev["nmse"],
            "pool_kl": dev["pool_kl_star_to_fitted"],
            "fresh_wmin": None,
            "tokens": None,
            "gpu_hours": round(results.get("train_runtime", 0.0) * 4 / 3600.0, 3),
            "train_detail": train, "dev_detail": dev,
            "train_runtime_s": results.get("train_runtime"),
            "train_samples": results.get("train_samples"),
        }
        print(json.dumps({label: {k: table[label][k] for k in
                                  ("steps", "train_nmse", "dev_nmse", "pool_kl", "gpu_hours")}}),
              flush=True)

    out = DIAG / "projection/pilot_table.json"
    existing = json.loads(out.read_text()) if out.exists() else {}
    existing.update(table)
    out.write_text(json.dumps(existing, indent=2) + "\n")
    print(json.dumps({"written": str(out), "arms": sorted(existing)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
