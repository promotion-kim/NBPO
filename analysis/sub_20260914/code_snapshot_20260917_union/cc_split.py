"""Complete-case splits for the solver, and the record of what they drop.

The solver refuses to run unless every prompt in the split has scores, and the
scorer excludes a prompt when any of its 92 scheduled pairs never resolved in
both presentation orders. That left 1,956 of 2,000 train prompts and 488 of 500
dev prompts, so the solver's split has to be the scored subset rather than the
declared one.

This is prompt-level complete-case analysis, which the scorer already declared:
no invalid verdict becomes a tie or a loss, nothing is imputed, and the dropped
prompts are counted here rather than quietly missing. The test split is
untouched, so the final panel is still the declared 1,000 prompts.
"""
import json
import glob
from pathlib import Path

import numpy as np

UF = Path("/work/uf4_20260910")
SRC = UF / "splits/safe_v1"
DST = UF / "splits/safe_v1cc"


def scored(name, shards=4):
    ids = set()
    for shard in range(shards):
        for path in (UF / "scores" / name / ("shard%d" % shard)).glob("chunk*.npz"):
            a = np.load(path, allow_pickle=True)
            ids.update(str(x) for x in a["prompt_ids"])
    return ids


def main():
    DST.mkdir(parents=True, exist_ok=True)
    report = {}
    for split, score_name in (("policy_train", "safe_v1"), ("policy_dev", "safe_dev_v1")):
        keep = scored(score_name)
        rows = [json.loads(l) for l in (SRC / ("%s.jsonl" % split)).open() if l.strip()]
        kept = [r for r in rows if r["prompt_id"] in keep]
        dropped = [r["prompt_id"] for r in rows if r["prompt_id"] not in keep]
        dest = DST / ("%s.jsonl" % split)
        with dest.open("w") as stream:
            for r in kept:
                stream.write(json.dumps(r, ensure_ascii=False) + "\n")
        report[split] = {"declared": len(rows), "kept": len(kept),
                         "dropped": len(dropped), "dropped_ids": dropped[:8],
                         "coverage": len(kept) / len(rows)}
        print("  %-12s declared %4d  kept %4d  dropped %3d  (%.2f%% coverage)"
              % (split, len(rows), len(kept), len(dropped),
                 100 * len(kept) / len(rows)))
    # the test split is copied unchanged: nothing has been judged on it yet
    (DST / "final_eval.jsonl").write_bytes((SRC / "final_eval.jsonl").read_bytes())
    report["final_eval"] = "copied unchanged; the declared 1,000-prompt test panel"
    report["reason"] = ("the solver requires scores for every split prompt; a prompt is "
                        "excluded by the scorer when any scheduled pair never resolved in "
                        "both presentation orders")
    (DST / "freeze.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
