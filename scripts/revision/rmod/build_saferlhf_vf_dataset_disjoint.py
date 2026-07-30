"""Build a FAIR (out-of-distribution) SafeRLHF VF training set from the training
pool (p4 train_pool), whose prompts are disjoint from the fresh_default_test
eval panel. Each shard file carries prompt_id + all_generated_responses +
all_rm_scores; helpfulness and harmlessness live in separate shard sets and are
joined by (prompt_id, response index). Excludes any prompt overlapping the eval
panel so the VF never sees an eval prompt. Fixes the in-distribution leak of the
first VF (trained on the eval panel itself).
"""
import argparse, glob, json
from datasets import Dataset


def load_shards(paths):
    out = {}
    for p in paths:
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            out[r["prompt_id"]] = (r["prompt"], r["all_generated_responses"], r["all_rm_scores"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores_dir", required=True)
    ap.add_argument("--eval_panel", required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()
    H = load_shards(sorted(glob.glob(f"{args.scores_dir}/helpfulness_shard_*.jsonl")))
    S = load_shards(sorted(glob.glob(f"{args.scores_dir}/harmlessness_shard_*.jsonl")))
    eval_prompts = {json.loads(l)["prompt"] for l in open(args.eval_panel, encoding="utf-8")}
    cols = {"prompt": [], "response": [], "rewards_helpful": [], "rewards_harmless": []}
    kept = skipped = 0
    for pid, (prompt, resps, hsc) in H.items():
        if pid not in S or prompt in eval_prompts:
            skipped += 1
            continue
        _, sresps, ssc = S[pid]
        for j, resp in enumerate(resps):
            if j >= len(ssc) or not str(resp).strip():
                continue
            cols["prompt"].append(prompt)
            cols["response"].append(str(resp))
            cols["rewards_helpful"].append(float(hsc[j]))
            cols["rewards_harmless"].append(float(ssc[j]))
        kept += 1
    ds = Dataset.from_dict(cols)
    ds.save_to_disk(args.output_dir)
    print(f"[disjoint-vf-data] {len(ds)} rows from {kept} prompts ({skipped} skipped) -> {args.output_dir}")


if __name__ == "__main__":
    main()
