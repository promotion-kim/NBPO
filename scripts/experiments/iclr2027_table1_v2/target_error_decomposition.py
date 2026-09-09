#!/usr/bin/env python3
"""Where does the held-out error live? Decompose it by prompt, length and target size.

The arms fit their training rows and do not transfer, so the useful question is
what distinguishes a row the policy gets right from one it does not. This splits
the held-out squared error by response length, by target magnitude, and by
prompt, and reports how much of the variance is between prompts rather than
within them -- a large between-prompt share means the failure is prompt-level,
which is where a per-prompt target would be expected to fail.

It deliberately does not assume a deterministic label is easy: the canonical
target is exact, and the question is whether it is PREDICTABLE from the prompt
and responses, which is a different property.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def bucket_stats(err, key, edges, names):
    out = []
    idx = np.digitize(key, edges)
    for b, name in enumerate(names):
        m = idx == b
        if m.sum() < 5:
            continue
        out.append({"bucket": name, "n": int(m.sum()),
                    "mean_squared_error": float((err[m] ** 2).mean()),
                    "rms_error": float(np.sqrt((err[m] ** 2).mean()))})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scored", required=True, help="held-out scored dir")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--eta", type=float, default=1.0)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from datasets import load_from_disk
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    d = load_from_disk(str(Path(args.scored) / "precomputed"))["train"].to_dict()
    h = (np.asarray(d["reference_chosen_logps"], float)
         - np.asarray(d["reference_rejected_logps"], float)
         - np.asarray(d["history0_chosen_logps"], float)
         + np.asarray(d["history0_rejected_logps"], float))
    T = args.eta * np.asarray(d["nbpo_weighted_z"], float)
    err = h - T
    pid = np.asarray(d["prompt_id"])

    n_tok = np.asarray([len(tok(c, add_special_tokens=False)["input_ids"])
                        + len(tok(r, add_special_tokens=False)["input_ids"])
                        for c, r in zip(d["chosen"], d["rejected"])], float)
    p_tok = np.asarray([len(tok(p, add_special_tokens=False)["input_ids"])
                        for p in d["prompt"]], float)
    truncated = (p_tok + n_tok) > args.max_length

    # between- vs within-prompt variance of the error
    uniq = {p: i for i, p in enumerate(sorted(set(pid)))}
    g = np.asarray([uniq[p] for p in pid])
    means = np.array([err[g == i].mean() for i in range(len(uniq))])
    n_per = np.array([(g == i).sum() for i in range(len(uniq))])
    between = float(np.average((means - err.mean()) ** 2, weights=n_per))
    total = float(err.var())

    report = {
        "n_pairs": int(err.size), "n_prompts": len(uniq),
        "overall_rms_error": float(np.sqrt((err ** 2).mean())),
        "target_rms": float(np.sqrt((T ** 2).mean())),
        "h_rms": float(np.sqrt((h ** 2).mean())),
        "variance_decomposition": {
            "between_prompt": between, "total": total,
            "between_prompt_share": between / total if total else None,
            "note": ("share of the error variance that is a per-prompt offset rather "
                     "than variation within a prompt; a large share means the policy "
                     "is wrong about whole prompts, not about particular pairs")},
        "by_response_tokens": bucket_stats(
            err, n_tok, [128, 256, 512, 1024],
            ["<128", "128-256", "256-512", "512-1024", ">1024"]),
        "by_target_magnitude": bucket_stats(
            err, np.abs(T), [0.5, 1.5, 3.0, 5.0],
            ["|T|<0.5", "0.5-1.5", "1.5-3", "3-5", ">5"]),
        "truncated_rows": {
            "n": int(truncated.sum()),
            "rms_error": float(np.sqrt((err[truncated] ** 2).mean()))
            if truncated.any() else None,
            "rms_error_untruncated": float(np.sqrt((err[~truncated] ** 2).mean()))},
        "correlation_of_abs_error_with": {
            "response_tokens": float(np.corrcoef(np.abs(err), n_tok)[0, 1]),
            "target_magnitude": float(np.corrcoef(np.abs(err), np.abs(T))[0, 1]),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
