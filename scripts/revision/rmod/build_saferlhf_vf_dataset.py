"""Build the RMOD value-function training dataset for the SafeRLHF setting from
the existing Beaver-scored evaluation pools (p7 stage-3 + p8 stage-4): rows
(prompt, response, rewards_helpful, rewards_harmless) in the
MultiObjectiveDataset on-disk format. Each pool holds ~10 policy responses per
prompt (response_pool.jsonl) with per-response Beaver scores in the score files
(all_rm_scores aligned to response_model_names). Note: these are eval-panel
prompts, which favors RMOD at decode time; we accept and disclose this.
"""
import argparse, json
from datasets import Dataset


def load_pool(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        out[r["prompt_id"]] = (r["prompt"], r["response_model_names"], r["all_generated_responses"])
    return out


def load_scores(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        out[r["prompt_id"]] = dict(zip(r["response_model_names"], r["all_rm_scores"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", nargs="+", required=True,
                    help="pool.jsonl:helpfulness.jsonl:harmlessness.jsonl triples")
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()
    rows = {"prompt": [], "response": [], "rewards_helpful": [], "rewards_harmless": []}
    for triple in args.pools:
        pool_p, help_p, harm_p = triple.split(":")
        pool, H, S = load_pool(pool_p), load_scores(help_p), load_scores(harm_p)
        for pid, (prompt, names, resps) in pool.items():
            if pid not in H or pid not in S:
                continue
            for name, resp in zip(names, resps):
                if name not in H[pid] or name not in S[pid] or not str(resp).strip():
                    continue
                rows["prompt"].append(prompt)
                rows["response"].append(str(resp))
                rows["rewards_helpful"].append(float(H[pid][name]))
                rows["rewards_harmless"].append(float(S[pid][name]))
    ds = Dataset.from_dict(rows)
    ds.save_to_disk(args.output_dir)
    print(f"[vf-data] {len(ds)} rows -> {args.output_dir}")


if __name__ == "__main__":
    main()
