"""Correlate each VF head's terminal value with ArmoRM attribute rewards over
the same prompts, to recover the true head->objective mapping and gauge VF
quality. Prints a [head x attribute] Pearson matrix and the best attribute per
head.
"""
import argparse, json
import numpy as np

ARMO = ["helpsteer-helpfulness", "helpsteer-correctness", "helpsteer-coherence",
        "helpsteer-complexity", "helpsteer-verbosity", "ultrafeedback-overall_score",
        "ultrafeedback-instruction_following", "ultrafeedback-truthfulness",
        "ultrafeedback-honesty", "ultrafeedback-helpfulness", "beavertails-is_safe"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vf_json", required=True, help="output of vf_head_probe")
    ap.add_argument("--armo_dir", required=True, help="dir with base128_<name>.jsonl per attribute")
    ap.add_argument("--attr_names", required=True, help="comma-separated attribute names present in armo_dir")
    args = ap.parse_args()

    vf = {r["prompt"]: r["head_values"] for r in json.load(open(args.vf_json))}
    attrs = args.attr_names.split(",")
    armo = {a: {} for a in attrs}
    for a in attrs:
        for line in open(f"{args.armo_dir}/base128_{a}.jsonl"):
            r = json.loads(line)
            armo[a][r["prompt"]] = float(np.mean(r["all_rm_scores"]))

    prompts = [p for p in vf if all(p in armo[a] for a in attrs)]
    print(f"aligned prompts: {len(prompts)}")
    H = np.array([vf[p] for p in prompts])              # [n, nheads]
    nH = H.shape[1]
    A = np.array([[armo[a][p] for a in attrs] for p in prompts])  # [n, nattr]

    print("\nPearson corr  (rows=VF head, cols=ArmoRM attribute):")
    print("head\\attr  " + "  ".join(f"{a[:8]:>8}" for a in attrs))
    corr = np.zeros((nH, len(attrs)))
    for h in range(nH):
        for j in range(len(attrs)):
            c = np.corrcoef(H[:, h], A[:, j])[0, 1]
            corr[h, j] = c
        best = int(np.argmax(corr[h]))
        print(f"head{h}    " + "  ".join(f"{corr[h,j]:+8.3f}" for j in range(len(attrs))) +
              f"   -> best: {attrs[best]} ({corr[h,best]:+.3f})")
    print("\nper-head max |corr|:", [round(float(np.max(np.abs(corr[h]))), 3) for h in range(nH)])


if __name__ == "__main__":
    main()
