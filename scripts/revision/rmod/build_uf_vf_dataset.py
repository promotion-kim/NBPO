"""Build a gemma-token-matched 5-head value-function training set for the UF
setting from the scored multi-response pool (5 ArmoRM-objective jsonl files,
each with all_generated_responses + all_rm_scores aligned per prompt). Emits a
MultiObjectiveDataset with columns prompt, response, obj0..obj4. This replaces
the public gpt2-large VF whose tokenizer mismatch broke within-prompt ranking.
"""
import argparse, json
from datasets import Dataset

OBJS = ["instruction_following", "truthfulness", "honesty", "helpfulness", "safety"]


def load(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        out[r["prompt"]] = (r["all_generated_responses"], r["all_rm_scores"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored_dir", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()
    tabs = {o: load(f"{args.scored_dir}/{args.split}_{o}.jsonl") for o in OBJS}
    cols = {"prompt": [], "response": [], **{f"obj{i}": [] for i in range(5)}}
    ref = tabs[OBJS[0]]
    for p, (resps, _) in ref.items():
        if any(p not in tabs[o] for o in OBJS):
            continue
        for j, resp in enumerate(resps):
            if not str(resp).strip():
                continue
            cols["prompt"].append(p)
            cols["response"].append(str(resp))
            for i, o in enumerate(OBJS):
                cols[f"obj{i}"].append(float(tabs[o][p][1][j]))
    ds = Dataset.from_dict(cols)
    ds.save_to_disk(args.output_dir)
    print(f"[uf-vf-data] {len(ds)} rows -> {args.output_dir}")


if __name__ == "__main__":
    main()
