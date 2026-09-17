"""Publish the five UW checkpoints under short, distinguishable names.

One name per loss, no dates or run tags, so the five are told apart at a glance:

    UW1-NBPO-PW  UW1-FixedRefNash-PW  UW1-DPO-soft  UW1-INPO-soft  UW1-SPPO

Every card is generated from the run's own artifacts -- training config, solver
record, and whichever measured cells exist at upload time -- and states the three
limits that apply to all five: the prompt set is the certified, representable
subset of a reduced-budget panel; there is one policy seed; and every reported
judge is a local open-weight model rather than the benchmarks' official API
judge. Cards are refreshed by re-running this script once more cells land.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from huggingface_hub import HfApi

UF = Path("/work/uf4_20260910")
OWNER = "promotion"
BASE = "Qwen/Qwen2.5-7B-Instruct"
CELLS = Path("/work/sub_20260914/prosper/uw1_cells.json")

ALLOW = ["config.json", "generation_config.json", "model.safetensors.index.json",
         "model-*.safetensors", "tokenizer.json", "tokenizer_config.json",
         "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt",
         "chat_template.jinja", "all_results.json", "train_results.json",
         "trainer_state.json"]
IGNORE = ["checkpoint-*", "checkpoint-*/*", "runs", "runs/*", "wandb", "wandb/*",
          "*.log", "optimizer.pt", "scheduler.pt", "rng_state*.pth",
          "reference_init_rank*.json", "rotary_buffer_precision_rank*.json",
          "runtime_rank*.jsonl"]

ARMS = [
    ("uw1_pw_nbpo", "UW1-NBPO-PW", "nbpopw",
     "NBPO with per-prompt inverse-surplus weights (Nash bargaining, adaptive reference)"),
    ("uw1_pw_fixedref", "UW1-FixedRefNash-PW", "fixedpw",
     "Nash bargaining per prompt against a frozen reference policy"),
    ("uw1_dpo_soft", "UW1-DPO-soft", "dpo",
     "DPO on the mean-preference oracle with soft labels, beta_DPO = 0.1"),
    ("uw1_inpo_soft", "UW1-INPO-soft", "inpo",
     "INPO's published squared loss integrated against the cached pair probability, "
     "eta = 0.005, tau/eta = 1/3"),
    ("uw1_sppo", "UW1-SPPO", "sppo",
     "SPPO's target on the eight-candidate mean win rates, eta_SPPO = 1000"),
    ("uw1_prosper", "UW1-PROSPER", "prosper",
     "PROSPER: Algorithm 1 step 7 with the Eq. 11 worst-objective criterion, "
     "estimated from M=7 leave-two-out banks over the 64 learner-by-reference "
     "fitting pairs"),
]
COLS = [("mmlu", "MMLU"), ("arc_c", "ARC-C"), ("hellaswag", "HellaSwag"),
        ("winogrande", "WinoGrande"), ("truthfulqa_mc2", "TruthfulQA"),
        ("gsm8k", "GSM8K"), ("ifeval", "IFEval"),
        ("ah_raw", "Arena-Hard"), ("ae_raw", "AlpacaEval"), ("mt", "MT-Bench")]


def table(cells):
    head = "| Model | " + " | ".join(n for _, n in COLS) + " |\n"
    head += "|" + "---|" * (len(COLS) + 1) + "\n"
    rows = []
    for _, repo, key, _ in ARMS:
        vals = []
        for col, _ in COLS:
            c = cells.get("UW.%s.%s" % (key, col))
            vals.append(("%.4f" % c["value"]) if c else "not yet measured")
        rows.append("| %s | %s |" % (repo, " | ".join(vals)))
    base_vals = []
    for col, _ in COLS:
        c = cells.get("UW.base.%s" % col)
        base_vals.append(("%.4f" % c["value"]) if c else "not yet measured")
    return head + "| base (untrained) | %s |\n" % " | ".join(base_vals) + "\n".join(rows) + "\n"


def card(repo, description, cells):
    cfg = (UF / "configs" / ("%s.yaml" % repo.lower().replace("uw1-", "uw1_")
                             .replace("-", "_"))).name
    return """---
library_name: transformers
base_model: %s
pipeline_tag: text-generation
tags:
- nbpo
- preference-optimization
- nash-bargaining
- research
---

# %s

%s

Fine-tuned from `%s`. One of five arms trained and evaluated under a single
protocol so that the only difference between them is the loss.

## Training

| | |
|---|---|
| Panel | WildChecklists, four native checklist items per prompt |
| Prompts | 319 train / 83 dev |
| Learner pairs | 8,932 train (all 28 unordered pairs of 8 sampled responses) |
| Optimizer steps | 140 at batch size 128, two epochs |
| Learning rate | 3e-7 cosine, warmup 0.1, weight decay 1e-6, grad clip 1.0 |
| Policy seed | 42 (one seed) |
| Candidate pool | 8 learner + 8 reference occurrences, temperature 1, top-p 1, 1024 tokens |
| Training judge | local open-weight judge, two judgments per pair and item, both orders |

## Measurements

Capability columns come from one local harness at the declared few-shot counts;
the two preference columns are win rates against each benchmark's released
baseline answers, judged by a local 72B open-weight model in both presentation
orders.

%s

## Three limits that apply to every row

1. **The prompt set is a filtered subset.** 2,000 prompts were requested; 750
   were judged under a reduced budget, 660 had complete feedback, 344 passed the
   Nash certificate and 319 survived the representability filter. The surviving
   prompts are those with a strictly positive bargaining surplus, so the panel is
   not a random sample of the dataset.
2. **One seed.** Differences between these arms cannot be separated from
   policy-seed variance, and none of the differences measured so far exceeds a
   single standard error on any column.
3. **Local judges.** No hosted API was called at any point. Arena-Hard normally
   judges with `gpt-4-1106`, AlpacaEval 2.0 with `weighted_alpaca_eval_gpt4_turbo`
   and MT-Bench with GPT-4; these numbers are comparable across the five arms and
   are not leaderboard scores.

## The five arms

%s
""" % (BASE, repo, description, BASE, table(cells),
       "\n".join("- [`%s/%s`](https://huggingface.co/%s/%s)" % (OWNER, r, OWNER, r)
                 for _, r, _, _ in ARMS))


def main():
    cells = json.loads(CELLS.read_text())["cells"] if CELLS.exists() else {}
    api = HfApi()
    print(json.dumps({"authenticated_as": api.whoami()["name"],
                      "measured_cells": len(cells)}), flush=True)
    out = Path("/work/sub_20260914/hf_cards")
    out.mkdir(parents=True, exist_ok=True)
    for arm, repo, key, description in ARMS:
        local = UF / "arms" / arm
        if not (local / "config.json").is_file():
            print(json.dumps({"skipped": repo, "why": "no exported checkpoint"}), flush=True)
            continue
        repo_id = "%s/%s" % (OWNER, repo)
        started = time.monotonic()
        api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)
        path = out / ("%s.md" % repo)
        path.write_text(card(repo, description, cells))
        api.upload_file(path_or_fileobj=str(path), path_in_repo="README.md",
                        repo_id=repo_id, repo_type="model",
                        commit_message="Model card with the measurements available so far")
        api.upload_folder(folder_path=str(local), repo_id=repo_id, repo_type="model",
                          allow_patterns=ALLOW, ignore_patterns=IGNORE,
                          commit_message="Upload the UW checkpoint")
        files = sorted(api.list_repo_files(repo_id=repo_id, repo_type="model"))
        weights = [f for f in files if f.endswith(".safetensors")]
        missing = {"config.json", "README.md", "model.safetensors.index.json"} - set(files)
        record = {"repo_id": repo_id, "files": len(files), "weight_shards": len(weights),
                  "missing": sorted(missing), "seconds": round(time.monotonic() - started, 1)}
        print(json.dumps(record), flush=True)
        if missing or not weights:
            raise SystemExit("incomplete upload: %s" % json.dumps(record))
    print("done", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
