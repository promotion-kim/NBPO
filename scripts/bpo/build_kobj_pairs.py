#!/usr/bin/env python3
"""Build K-objective pairs from N per-objective scored files, in the schema that
build_os_ronpo_targets + the RONPO loss consume (objective_names,
normalized_objective_scores, chosen/rejected_index, ronpo validation fields).
Each scored file: rows with prompt_id, all_generated_responses, all_rm_scores.
Usage: build_kobj_pairs.py --scored name=f.jsonl ... --train-output ... --test-output ..."""
import argparse, hashlib, itertools, json, math
from pathlib import Path


def sigmoid(v): return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, v))))
def minmax(vals):
    lo, hi = min(vals), max(vals)
    return [0.5] * len(vals) if hi == lo else [(v - lo) / (hi - lo) for v in vals]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", nargs="+", required=True, help="name=path.jsonl per objective")
    ap.add_argument("--train-output", type=Path, required=True)
    ap.add_argument("--test-output", type=Path, required=True)
    ap.add_argument("--pairs-per-prompt", type=int, default=6)
    ap.add_argument("--internal-test-prompts", type=int, default=64)
    ap.add_argument("--scale", type=float, default=8.0)
    ap.add_argument("--split-salt", default="hh-k3")
    args = ap.parse_args()
    OBJ = [s.split("=")[0] for s in args.scored]
    src = {}
    for s in args.scored:
        name, path = s.split("=", 1)
        rows = [json.loads(l) for l in open(path) if l.strip()]
        # join on prompt_id when present, else on row index (files are decode-aligned)
        src[name] = {str(r.get("prompt_id", i)): {**r, "prompt_id": r.get("prompt_id", i)}
                     for i, r in enumerate(rows)}
    pids = sorted(set.intersection(*[set(src[o]) for o in OBJ]))
    itest = set(sorted(pids, key=lambda v: hashlib.sha256(
        f"{args.split_salt}|{v}".encode()).hexdigest())[:args.internal_test_prompts])
    out = {"train": [], "test": []}
    for pid in pids:
        base = src[OBJ[0]][pid]
        resps = [str(x) for x in base["all_generated_responses"]]
        nR = len(resps)
        if nR < 2 or any(len(src[o][pid]["all_generated_responses"]) != nR for o in OBJ):
            continue
        raw = {o: [float(x) for x in src[o][pid]["all_rm_scores"]] for o in OBJ}
        norm = {o: minmax(raw[o]) for o in OBJ}
        avg = [sum(norm[o][i] for o in OBJ) / len(OBJ) for i in range(nR)]
        win = [sum(sigmoid(args.scale * (avg[i] - avg[j])) for j in range(nR)) / nR for i in range(nR)]
        pairs = sorted(itertools.combinations(range(nR), 2),
                       key=lambda p: hashlib.sha256(f"{args.split_salt}|{pid}|{p[0]}|{p[1]}".encode()).hexdigest())
        for (l, r) in pairs[:args.pairs_per_prompt]:
            chosen, rejected = (r, l) if (avg[r], -r) > (avg[l], -l) else (l, r)
            o0 = OBJ[0]
            gap = (sigmoid(args.scale * (norm[o0][chosen] - norm[o0][rejected]))
                   - sigmoid(args.scale * (norm[o0][rejected] - norm[o0][rejected])))
            out["test" if pid in itest else "train"].append({
                "prompt_id": pid, "prompt": base["prompt"], "all_generated_responses": resps,
                "objective_names": OBJ, "normalized_objective_scores": norm, "objective_scores": raw,
                "avg_objective_scores": avg, "chosen": resps[chosen], "rejected": resps[rejected],
                "chosen_index": chosen, "rejected_index": rejected,
                "homogeneous_oracle_preference_scale": args.scale,
                "ronpo_target": gap, "ronpo_objective_index": 0, "ronpo_objective_gap": gap,
                "ronpo_adversary_response_index": rejected, "ronpo_weight": 1.0,
                "chosen_probs": win[chosen], "rejected_probs": win[rejected],
                "avg_oracle_chosen_score": avg[chosen], "avg_oracle_rejected_score": avg[rejected],
            })
    for split, rows in out.items():
        p = args.train_output if split == "train" else args.test_output
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"objectives": OBJ, "prompts": len(pids),
                      "train": len(out["train"]), "test": len(out["test"])}))


if __name__ == "__main__":
    main()
