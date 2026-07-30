"""Convert the Robust-Decoding HH_gemma-2-2b-it CSV (4 responses/prompt, with
rewards_harmless + rewards_helpful) into the two per-objective scored jsonl
files that build_multi_objective_dataset consumes, so RONPO can train on the
genuinely-conflicting helpful-vs-harmless objectives (RMOD Fig-3c setting).

The gemma chat template is stripped back to the raw user text (all user turns
joined) so the downstream pipeline re-applies the template consistently.
"""
import argparse, json, re
import pandas as pd

USER_RE = re.compile(r"<start_of_turn>user\n(.*?)<end_of_turn>", re.DOTALL)


def raw_prompt(templated: str) -> str:
    turns = USER_RE.findall(templated)
    return "\n\n".join(t.strip() for t in turns) if turns else templated.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out_harmless", required=True)
    ap.add_argument("--out_helpful", required=True)
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    df["rp"] = df["prompts"].map(raw_prompt)
    n = 0
    fh = open(args.out_harmless, "w", encoding="utf-8")
    fp = open(args.out_helpful, "w", encoding="utf-8")
    for rp, g in df.groupby("rp", sort=False):
        resp = [str(x) for x in g["responses"].tolist()]
        if len(resp) < 2:
            continue
        harm = [float(x) for x in g["rewards_harmless"].tolist()]
        helpf = [float(x) for x in g["rewards_helpful"].tolist()]
        fh.write(json.dumps({"prompt": rp, "all_generated_responses": resp, "all_rm_scores": harm}, ensure_ascii=False) + "\n")
        fp.write(json.dumps({"prompt": rp, "all_generated_responses": resp, "all_rm_scores": helpf}, ensure_ascii=False) + "\n")
        n += 1
    fh.close(); fp.close()
    print(f"[hh] {n} prompts -> {args.out_harmless} / {args.out_helpful}")


if __name__ == "__main__":
    main()
