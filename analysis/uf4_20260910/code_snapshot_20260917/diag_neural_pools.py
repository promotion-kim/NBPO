"""Importance-reweight the learner pool by a fitted checkpoint, on the frozen panel.

For every panel prompt and each of its eight learner occurrences this computes
the sequence log-likelihood under the fitted policy and under the proximal
centre pi_t, using the pool's own cached training tokenization -- the same
input_ids, attention mask and label mask the trainer consumed, summed over
non-masked response tokens -- and then

    p_neural(i) proportional to p_t(i) exp{log pi_theta(y_i) - log pi_t(y_i)},

normalized by log-sum-exp. This is an importance diagnostic on the sampled
support, not the policy's distribution over responses. Raw-likelihood
normalization and length-normalized scores are deliberately not used.

Also reports KL(p* || p_neural), KL(p* || p_t), total variation and the
log-ratio scale, which are the projection columns of the pilot table.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/work/uf4_20260910")
DIAG = ROOT / "analysis/diag_20260914"


def load_pool_rows(pool, wanted):
    """candidate_id -> the cached training tokenization of that occurrence."""
    out = {}
    for path in sorted(glob.glob(str(ROOT / "pools" / pool / "shard*/chunk*.jsonl"))):
        with open(path) as stream:
            for line in stream:
                r = json.loads(line)
                if r["candidate_id"] in wanted:
                    out[r["candidate_id"]] = {"input_ids": r["input_ids"],
                                              "labels": r["labels"],
                                              "attention_mask": r["attention_mask"],
                                              "n_tokens": r["n_tokens"]}
    return out


@torch.no_grad()
def sequence_logprobs(model, rows, order, device, batch=4):
    """Sum of log p(token) over non-masked label positions, one value per row."""
    values = []
    for start in range(0, len(order), batch):
        chunk = order[start:start + batch]
        width = max(len(rows[c]["input_ids"]) for c in chunk)
        ids = torch.zeros(len(chunk), width, dtype=torch.long)
        att = torch.zeros(len(chunk), width, dtype=torch.long)
        lab = torch.full((len(chunk), width), -100, dtype=torch.long)
        for k, cid in enumerate(chunk):
            r = rows[cid]
            n = len(r["input_ids"])
            ids[k, :n] = torch.tensor(r["input_ids"], dtype=torch.long)
            att[k, :n] = torch.tensor(r["attention_mask"], dtype=torch.long)
            lab[k, :n] = torch.tensor(r["labels"], dtype=torch.long)
        ids, att, lab = ids.to(device), att.to(device), lab.to(device)
        logits = model(input_ids=ids, attention_mask=att).logits.float()
        logp = torch.log_softmax(logits[:, :-1], dim=-1)
        target = lab[:, 1:]
        mask = target != -100
        gathered = torch.gather(logp, 2, target.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        values.extend((gathered * mask).sum(dim=1).tolist())
    return values


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="+", default=["nbpo_mse_s42", "util_mse_s42"])
    ap.add_argument("--targets", nargs="+", default=["nash_v1", "util_l1matched_v1"],
                    help="target set whose p* pairs with each arm, same order")
    ap.add_argument("--base", default="/work/models/bases/Llama-3.1-8B-Instruct")
    ap.add_argument("--pool", default="dev_v1")
    ap.add_argument("--out", default=str(DIAG / "neural_pools.json"))
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()

    panel = json.loads((DIAG / "panel_dev200.json").read_text())
    ids = panel["panel_prompt_ids"]
    candidates = {pid: panel["learner_occurrences"][pid] for pid in ids}
    wanted = {c for v in candidates.values() for c in v}
    rows = load_pool_rows(args.pool, wanted)
    missing = sorted(wanted - set(rows))
    if missing:
        raise SystemExit("missing cached tokenization for %d occurrences" % len(missing))
    order = [c for pid in ids for c in candidates[pid]]

    from transformers import AutoModelForCausalLM
    device = "cuda"

    def score(path):
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16,
                                                     attn_implementation="sdpa",
                                                     local_files_only=True).to(device).eval()
        out = sequence_logprobs(model, rows, order, device, args.batch)
        del model
        torch.cuda.empty_cache()
        return dict(zip(order, out))

    base_logp = score(args.base)
    payload, diagnostics, deltas = {}, {}, {}
    for arm, target in zip(args.arms, args.targets):
        arm_logp = score(str(ROOT / "arms" / arm))
        meta = json.loads((ROOT / "targets" / target / "dev/tensor/meta.json").read_text())
        index = {pid: k for k, pid in enumerate(meta["prompt_ids"])}
        pi_star = np.load(ROOT / "targets" / target / "dev/solver/pi_star.npz")["pi"]
        pi_t = np.load(ROOT / "targets" / target / "dev/solver/pi_t.npz")["pi"]

        table, kl_star_neural, kl_star_t, tv, ratio_rms = {}, [], [], [], []
        for pid in ids:
            if pid not in index:
                continue
            cs = candidates[pid]
            delta = np.array([arm_logp[c] - base_logp[c] for c in cs], dtype=np.float64)
            pt = pi_t[index[pid]]
            logits = np.log(np.clip(pt, 1e-12, None)) + delta
            logits -= logits.max()
            p = np.exp(logits)
            p /= p.sum()
            table[pid] = p.tolist()
            ps = np.clip(pi_star[index[pid]], 1e-12, None)
            kl_star_neural.append(float(np.sum(ps * (np.log(ps) - np.log(np.clip(p, 1e-12, None))))))
            kl_star_t.append(float(np.sum(ps * (np.log(ps) - np.log(np.clip(pt, 1e-12, None))))))
            tv.append(float(0.5 * np.abs(p - ps).sum()))
            ratio_rms.append(float(np.sqrt(np.mean(delta ** 2))))
        payload[arm] = table
        deltas[arm] = {pid: [float(arm_logp[c] - base_logp[c]) for c in candidates[pid]]
                       for pid in ids}
        diagnostics[arm] = {
            "target_set": target, "prompts": len(table),
            "pool_kl_star_to_neural_mean": float(np.mean(kl_star_neural)),
            "pool_kl_star_to_source_mean": float(np.mean(kl_star_t)),
            "tv_star_to_neural_mean": float(np.mean(tv)),
            "logratio_rms_mean": float(np.mean(ratio_rms)),
            "note": ("importance reweighting on the sampled support; not the policy's "
                     "response distribution"),
        }
        print(json.dumps({arm: diagnostics[arm]}), flush=True)

    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    (DIAG / "pool_log_ratios.json").write_text(json.dumps(deltas) + "\n")
    (DIAG / "neural_pool_diagnostics.json").write_text(json.dumps(
        {"base": args.base, "pool": args.pool,
         "panel_sha256": panel["panel_sha256"], "arms": diagnostics}, indent=2) + "\n")
    print(json.dumps({"written": args.out, "arms": list(payload)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
