"""Did the trained policy move the finite pool toward the solver target?

The trainer already reports the pairwise regression diagnostics. This asks the
question those cannot answer: the pool distribution the policy actually induces,
p_theta(i) proportional to p_t(i) * exp(log pi_theta - log pi_t), against the
solver target p_star. Forward KL(p_star || p_theta) is the quantity weighted
behaviour cloning minimises on the training pools, so its value on held-out
prompts is the direct test of whether the neural realization transfers.

Every candidate is scored once, on exactly the token ids and label masks the
training rows carry, and the reference is the frozen base model on those same
ids -- no cache is trusted here.
"""
import argparse, json, math
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM

from mnpo_scripts.response_logps import response_logps


def unique_candidates(dataset):
    """One entry per sampled response, with its solver mass and pool centre."""
    out = {}
    for row in dataset:
        for side, other in (("chosen", "a"), ("rejected", "b")):
            rid = row[f"{side}_response_id"]
            if rid in out:
                continue
            out[rid] = {
                "prompt_id": row["prompt_id"],
                "candidate_index": row[f"{side}_candidate_index"],
                "input_ids": row[f"{side}_input_ids"],
                "attention_mask": row[f"{side}_attention_mask"],
                "labels": row[f"{side}_labels"],
                "p_star": row[f"nbpo_weight_{other}"],
                "p_t": row[f"nbpo_center_{other}"],
            }
    return out


@torch.no_grad()
def score(model, items, device, batch=8):
    values = []
    for start in range(0, len(items), batch):
        chunk = items[start:start + batch]
        width = max(len(c["input_ids"]) for c in chunk)
        ids = torch.zeros(len(chunk), width, dtype=torch.long)
        mask = torch.zeros(len(chunk), width, dtype=torch.long)
        labels = torch.full((len(chunk), width), -100, dtype=torch.long)
        for r, c in enumerate(chunk):
            n = len(c["input_ids"])
            ids[r, :n] = torch.tensor(c["input_ids"])
            mask[r, :n] = torch.tensor(c["attention_mask"])
            labels[r, :n] = torch.tensor(c["labels"])
        ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
        logits = model(input_ids=ids, attention_mask=mask).logits
        values.append(response_logps(logits, labels, label_pad_token_id=-100).float().cpu())
    return torch.cat(values)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--reference", default="/work/models/bases/Llama-3.1-8B-Instruct")
    ap.add_argument("--dataset", default="/work/nbpo_repair_20260909/datasets/nash_repair_v2")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", default="/work/nbpo_repair_20260909/analysis_claude")
    args = ap.parse_args()

    device = "cuda"
    dataset = load_from_disk(args.dataset)[args.split]
    table = unique_candidates(dataset)
    keys = sorted(table)
    items = [table[k] for k in keys]

    policy = AutoModelForCausalLM.from_pretrained(args.policy, dtype=torch.bfloat16).to(device).eval()
    logp_policy = score(policy, items, device)
    del policy
    torch.cuda.empty_cache()
    reference = AutoModelForCausalLM.from_pretrained(args.reference, dtype=torch.bfloat16).to(device).eval()
    logp_reference = score(reference, items, device)
    del reference
    torch.cuda.empty_cache()

    delta = (logp_policy - logp_reference).double()
    groups = {}
    for index, key in enumerate(keys):
        groups.setdefault(items[index]["prompt_id"], []).append(index)

    rows = []
    for prompt_id, indices in groups.items():
        indices = sorted(indices, key=lambda i: items[i]["candidate_index"])
        p_star = torch.tensor([items[i]["p_star"] for i in indices], dtype=torch.float64)
        p_t = torch.tensor([items[i]["p_t"] for i in indices], dtype=torch.float64)
        d = delta[indices]
        length = torch.tensor([sum(1 for v in items[i]["labels"] if v != -100) for i in indices],
                              dtype=torch.float64)
        # the induced pool law: p_theta propto p_t * exp(delta)
        logits = torch.log(p_t) + d
        p_theta = torch.softmax(logits, 0).double()
        eps = 1e-12
        rows.append({
            "prompt_id": prompt_id,
            "kl_target_to_policy": float((p_star * (torch.log(p_star + eps) - torch.log(p_theta + eps))).sum()),
            "kl_target_to_center": float((p_star * (torch.log(p_star + eps) - torch.log(p_t + eps))).sum()),
            "kl_policy_to_center": float((p_theta * (torch.log(p_theta + eps) - torch.log(p_t + eps))).sum()),
            "top1_match": int(torch.argmax(p_theta) == torch.argmax(p_star)),
            "mass_on_target_argmax": float(p_theta[torch.argmax(p_star)]),
            "delta": [float(v) for v in d],
            "p_star": [float(v) for v in p_star],
            "p_theta": [float(v) for v in p_theta],
            "length": [float(v) for v in length],
        })

    def stack(field):
        return torch.tensor([r[field] for r in rows], dtype=torch.float64)

    d_all, p_all, l_all = stack("delta"), stack("p_star"), stack("length")
    g_all = torch.log(p_all + 1e-12) - math.log(0.125)
    centre = lambda t: t - t.mean(1, keepdim=True)
    dc, gc, lc = centre(d_all).flatten(), centre(g_all).flatten(), centre(l_all).flatten()

    def corr(a, b):
        if a.std() == 0 or b.std() == 0:
            return None
        return float(((a - a.mean()) * (b - b.mean())).mean() / (a.std(0, unbiased=False) * b.std(0, unbiased=False)))

    r_dg, r_dl, r_gl = corr(dc, gc), corr(dc, lc), corr(gc, lc)
    partial = None
    if None not in (r_dg, r_dl, r_gl) and abs(r_dl) < 1 and abs(r_gl) < 1:
        partial = (r_dg - r_dl * r_gl) / math.sqrt((1 - r_dl ** 2) * (1 - r_gl ** 2))

    summary = {
        "label": args.label, "policy": args.policy, "split": args.split,
        "n_prompts": len(rows), "n_candidates_scored": len(items),
        "delta_mean": float(d_all.mean()), "delta_std": float(d_all.std()),
        "delta_within_prompt_std": float(dc.std()),
        "corr_delta_vs_g": r_dg, "corr_delta_vs_length": r_dl, "corr_g_vs_length": r_gl,
        "partial_corr_delta_vs_g_given_length": partial,
        "kl_target_to_policy_mean": float(stack("kl_target_to_policy").mean()) if rows else None,
        "kl_target_to_center_mean": float(torch.tensor([r["kl_target_to_center"] for r in rows]).mean()),
        "kl_policy_to_center_mean": float(torch.tensor([r["kl_policy_to_center"] for r in rows]).mean()),
        "top1_match_rate": float(sum(r["top1_match"] for r in rows) / len(rows)),
        "mass_on_target_argmax_mean": float(torch.tensor([r["mass_on_target_argmax"] for r in rows]).mean()),
        "center_mass_on_target_argmax": 0.125,
    }
    summary["kl_improvement_vs_center"] = (
        summary["kl_target_to_center_mean"] - summary["kl_target_to_policy_mean"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"pool_drift_{args.label}_{args.split}.json").write_text(json.dumps(summary, indent=2))
    with (out / f"pool_drift_{args.label}_{args.split}.jsonl").open("w") as handle:
        for r in rows:
            handle.write(json.dumps(r) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
