#!/usr/bin/env python3
"""Publish the Section-7 policy checkpoints, diagnostic status and all.

Every one of these arms FAILED the pre-registered gate. They are released anyway,
because each is the artifact behind a reported number: an arm's normalized MSE,
sign agreement, Pearson and Spearman are properties of that checkpoint, and a
negative result rests on its checkpoints exactly as a positive one does.

What ships with each arm is what makes its number checkable: the run config it
was trained under, its gate record, and the target estimator it was trained on.
Nothing here is a recommended model, and the card says so in the first line.

The token is read from HF_TOKEN and is never written to disk or into the upload.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

CARD = """---
license: cc-by-nc-4.0
base_model:
- meta-llama/Llama-3.1-8B-Instruct
tags:
- diagnostic
- negative-result
- preference-optimization
library_name: transformers
---

# NBPO Section-7 policy checkpoints (diagnostic; the gate did not pass)

**None of these is a recommended model.** Every arm here failed the
pre-registered neural-realization gate. They are published because each one is
the artifact behind a reported number, and a negative result should be as
checkable as a positive one.

## The gate, and what happened

A policy is asked to realize the finite-pool Eq. (26) target: the pairwise
log-ratio change

    h = [log pi(y|x) - log pi(z|x)] - [log pi_t(y|x) - log pi_t(z|x)]

should track `eta * z`. The gate required held-out `normalized MSE < 0.90`, sign
agreement `> 0.65`, and positive Pearson and Spearman. No arm cleared it; the
best held-out normalized MSE was slightly above 1.0, which is what predicting
nothing scores.

{table}

`nMSE` is `MSE / Var(target)` on held-out prompts, each arm scored at its own
training eta. An arm that never moved would score 1.0.

## Target estimators

| estimator | what it integrates | max attainable $r^2$ |
|---|---|---|
| `sampled` | nothing -- one comparator draw and two Bernoulli flips | 0.142 |
| `rao_blackwell` | the Bernoulli flips only | 0.983 |
| `canonical` | comparator and flips: the exact finite-pool expectation | 1.000 |

The three are unbiased for the same quantity and differ only in variance.

## What the diagnostics established

The trainer is not broken: on 8 prompts and 224 pairs the same code reduces the
loss 87% below its `h = 0` baseline. Longer training helps monotonically
(held-out Pearson +0.025 at 300 steps, +0.089 at 1200). Relaxing gradient
clipping 100x barely moves the result. One real defect was found and is recorded
rather than fixed away: at `learning_rate = 0` the online log-probabilities do
not reproduce the cached ones, so `h` carries a spurious component of RMS ~0.35
in the TRAINING signal. Evaluation is unaffected -- both of its terms come from
the same cached path.

## Contents

Each `arms/<label>/` holds the checkpoint, its `run_config.yaml`, and
`gate.json` with the full metric record including both exact MSE identities.

## Licence

CC BY-NC 4.0, inherited from the PKU-SafeRLHF supervision. Base model
[meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct);
preference supervision from
[promotion/nbpo-saferlhf-preference-models](https://huggingface.co/promotion/nbpo-saferlhf-preference-models).
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+", required=True,
                   help="label=<checkpoint dir>=<gate.json>=<run_config.yaml>=<estimator>")
    ap.add_argument("--repo-id", default="promotion/nbpo-policy-diagnostics")
    ap.add_argument("--manifest", type=Path, required=True)
    args = ap.parse_args()

    from huggingface_hub import HfApi
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is not set; refusing to attempt an upload")
    api = HfApi(token=token)
    api.create_repo(args.repo_id, repo_type="model", private=False, exist_ok=True)

    rows, entries = [], {}
    for spec in args.arms:
        label, ckpt, gate, cfg, est = spec.split("=")
        g = json.loads(Path(gate).read_text())
        t = g["splits"]["test"]["metrics"]
        v = g["splits"]["validation"]["metrics"]
        rows.append(f"| `{label}` | {est} | {g['eta']} | {v['normalized_mse_var']:.3f} | "
                    f"{t['normalized_mse_var']:.3f} | {t['sign_agreement']:.3f} | "
                    f"{t['pearson']:+.3f} | {t['spearman']:+.3f} |")
        entries[label] = {"estimator": est, "eta": g["eta"],
                          "checkpoint": ckpt,
                          "validation": v, "test": t,
                          "gate_passed": bool(g["splits"]["test"]["gate"]["all_pass"])}
        api.upload_folder(folder_path=ckpt, repo_id=args.repo_id,
                          path_in_repo=f"arms/{label}", repo_type="model",
                          # the Trainer writes its own README whose base_model is a
                          # LOCAL path; the Hub rejects that, and the repo card is
                          # written at the root anyway
                          ignore_patterns=["optimizer*", "scheduler*", "rng_state*",
                                           "README.md", "training_args.bin",
                                           "trainer_state.json"])
        for src, dst in ((gate, "gate.json"), (cfg, "run_config.yaml")):
            api.upload_file(path_or_fileobj=src, repo_id=args.repo_id,
                            path_in_repo=f"arms/{label}/{dst}", repo_type="model")
        print(f"uploaded {label}", flush=True)

    header = ("| arm | target estimator | eta | val nMSE | test nMSE | test sign | "
              "test Pearson | test Spearman |\n|---|---|---|---|---|---|---|---|")
    card = CARD.format(table=header + "\n" + "\n".join(rows))
    Path("/tmp/_policy_card.md").write_text(card)
    api.upload_file(path_or_fileobj="/tmp/_policy_card.md", repo_id=args.repo_id,
                    path_in_repo="README.md", repo_type="model")
    info = api.model_info(args.repo_id)
    out = {"repo_id": args.repo_id, "url": f"https://huggingface.co/{args.repo_id}",
           "revision": info.sha, "public": not info.private,
           "status": "diagnostic -- every arm failed the pre-registered gate",
           "arms": entries}
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "arms"}, indent=2))


if __name__ == "__main__":
    main()
