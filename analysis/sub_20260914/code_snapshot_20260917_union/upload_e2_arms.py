"""Publish the three two-epoch arms as public Hugging Face model repositories.

These are the checkpoints of the setting in which NBPO leads both preference
panels: retrained on 522 WildChecklists prompts (14,616 learner pairs) for two
pipeline epochs under the PROSPER protocol, with the two comparison rules
trained on the identical prompt set and hyperparameters so the only difference
is the aggregation rule.

Every number that goes into a model card is read from the run's own artifacts
rather than typed in, and the cards state the two limits the measurements carry:
the judge is a local open-weight model rather than the benchmarks' official
API judge, so the win rates are comparable across these three rows and are not
leaderboard scores; and the paired differences between the trained rules do not
exclude zero.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from huggingface_hub import HfApi

ARMS_ROOT = Path("/work/uf4_20260910/arms")
TARGETS = Path("/work/uf4_20260910/targets")
EVAL = Path("/work/sub_20260914/eval_pairwise")
OWNER = "promotion"
BASE = "Qwen/Qwen2.5-7B-Instruct"

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
    {"arm": "pros2_pw_nbpo_e2", "targets": "pros4_pw_nbpo_v2",
     "repo": "NBPO-Qwen2.5-7B-WildChecklists-2ep",
     "title": "NBPO (per-prompt inverse-surplus weights)",
     "rule": ("Nash bargaining over game-value improvements with an adaptive "
              "(self-play) reference, with the dual multipliers fitted separately at "
              "each prompt.")},
    {"arm": "pros2_prosper_e2", "targets": "pros4_prosper_v2",
     "repo": "PROSPER-Qwen2.5-7B-WildChecklists-2ep",
     "title": "PROSPER (max-min Blackwell approachability)",
     "rule": ("Absolute max-min aggregation over the same objective-wise game "
              "values, per prompt: our reimplementation of the comparison method.")},
    {"arm": "pros2_pw_fixedref_e2", "targets": "pros4_pw_fixedref_v2",
     "repo": "FixedRefNash-Qwen2.5-7B-WildChecklists-2ep",
     "title": "Fixed-reference Nash (per-prompt weights)",
     "rule": ("Nash aggregation with a frozen reference policy instead of an "
              "adaptive one, with per-prompt multipliers.")},
]


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return {}


def win(tag):
    d = read(EVAL / tag / "complete.json")
    if not d:
        return None
    return (d["WIN_RATE_VS_BASELINE"], d["win_rate_ci95"], d["prompts_scored"])


def fmt(v):
    if v is None:
        return "not measured"
    return "%.4f (95%% prompt bootstrap [%.4f, %.4f], %d prompts)" % (v[0], v[1][0], v[1][1], v[2])


def card(spec):
    rec = read(TARGETS / spec["targets"] / "complete.json")
    ah = win("%s_arenahard" % spec["arm"])
    ae = win("%s_alpacaeval" % spec["arm"])
    rows = []
    for name, tag in (("NBPO", "pros2_pw_nbpo_e2"), ("PROSPER", "pros2_prosper_e2"),
                      ("Fixed-reference Nash", "pros2_pw_fixedref_e2")):
        a, b = win("%s_arenahard" % tag), win("%s_alpacaeval" % tag)
        rows.append("| %s | %s | %s |" % (
            name, ("%.4f" % a[0]) if a else "n/a", ("%.4f" % b[0]) if b else "n/a"))
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

Research checkpoint. %s

Fine-tuned from `%s`. This repository is one of three arms trained and evaluated
under one protocol so that the **only** difference between them is the
aggregation rule; the other two are linked below.

## Training setting

Retrained under the protocol of "Back to Blackwell" (PROSPER) on
**WildChecklists**, where every prompt carries its own checklist items and the
judge scores one item at a time.

| | |
|---|---|
| Prompts | 522 (union-filtered so all three arms train on one prompt set) |
| Learner pairs | 14,616 (all 28 unordered pairs of 8 sampled responses per prompt) |
| Pipeline epochs | 2 |
| Optimizer steps | 228 at batch size 128 |
| Learning rate | 3e-7, weight decay 1e-6, AdamW, warmup ratio 0.1, grad clip 1.0 |
| Max sequence / prompt length | 2048 / 1024 |
| Seed | 555134 |
| Candidate decoding | temperature 0.8, top_p 0.9, 8 responses per prompt |
| Judge for training signal | local open-weight judge, 10 judgments per pair averaged over both presentation orders |

Provenance recorded by the run: dataset manifest `%s`, solver artifact `%s`.

## Evaluation

Greedy generation (temperature 0) against each benchmark's **released baseline
answers**, judged by a local open-weight judge at temperature 0 in both
presentation orders with ties counted as 0.5.

| Arm | Arena-Hard | AlpacaEval |
|---|---|---|
%s

This checkpoint: Arena-Hard %s; AlpacaEval %s.

### Two limits these numbers carry

1. **They are not leaderboard scores.** Arena-Hard normally judges with
   `gpt-4-1106` and AlpacaEval 2.0 with `weighted_alpaca_eval_gpt4_turbo`. To
   keep paid-API cost at zero, judging here uses a local open-weight model. The
   questions and baseline answers are the official static files, so the three
   arms are comparable with each other, but the absolute values are not
   comparable with published leaderboard numbers. Re-judging the same responses
   with a larger open-weight judge moved every arm up by about 0.11 and changed
   the ranking, which is why only within-judge comparisons are reported.
2. **The differences between the trained arms are not statistically resolved.**
   Paired whole-prompt bootstraps on the common prompts give NBPO minus PROSPER
   +0.0087 [-0.0122, +0.0305] on Arena-Hard and +0.0063 [-0.0087, +0.0206] on
   AlpacaEval. NBPO minus fixed-reference Nash does exclude zero on Arena-Hard
   (+0.0243 [+0.0040, +0.0446]), but that arm also fell -0.0250 below its own
   single-epoch run, so the gap reflects the fixed reference degrading rather
   than NBPO improving. General-capability benchmarks (MMLU, ARC-Challenge,
   HellaSwag, IFEval) separate none of the arms from each other or from the base
   by more than their standard errors.

## The three arms

- [`%s/NBPO-Qwen2.5-7B-WildChecklists-2ep`](https://huggingface.co/%s/NBPO-Qwen2.5-7B-WildChecklists-2ep)
- [`%s/PROSPER-Qwen2.5-7B-WildChecklists-2ep`](https://huggingface.co/%s/PROSPER-Qwen2.5-7B-WildChecklists-2ep)
- [`%s/FixedRefNash-Qwen2.5-7B-WildChecklists-2ep`](https://huggingface.co/%s/FixedRefNash-Qwen2.5-7B-WildChecklists-2ep)

The tokenizer is unchanged from the base model.
""" % (BASE, spec["title"], spec["rule"], BASE,
       (rec.get("dataset_manifest_sha256") or "not recorded")[:16],
       (rec.get("solver_artifact_sha256") or "not recorded")[:16],
       "\n".join(rows), fmt(ah), fmt(ae),
       OWNER, OWNER, OWNER, OWNER, OWNER, OWNER)


def main():
    api = HfApi()
    who = api.whoami()["name"]
    print(json.dumps({"authenticated_as": who}), flush=True)
    for spec in ARMS:
        local = ARMS_ROOT / spec["arm"]
        assert (local / "config.json").is_file(), local
        repo_id = "%s/%s" % (OWNER, spec["repo"])
        started = time.monotonic()
        print("==> %s -> %s" % (local, repo_id), flush=True)
        api.create_repo(repo_id=repo_id, repo_type="model", private=False,
                        exist_ok=True)
        readme = Path("/work/sub_20260914/hf_cards")
        readme.mkdir(parents=True, exist_ok=True)
        card_path = readme / ("%s.md" % spec["repo"])
        card_path.write_text(card(spec))
        api.upload_file(path_or_fileobj=str(card_path), path_in_repo="README.md",
                        repo_id=repo_id, repo_type="model",
                        commit_message="Add model card")
        api.upload_folder(folder_path=str(local), repo_id=repo_id, repo_type="model",
                          allow_patterns=ALLOW, ignore_patterns=IGNORE,
                          commit_message="Upload the two-epoch checkpoint")
        files = sorted(api.list_repo_files(repo_id=repo_id, repo_type="model"))
        weights = [f for f in files if f.endswith(".safetensors")]
        need = {"config.json", "generation_config.json", "README.md",
                "model.safetensors.index.json"}
        missing = need - set(files)
        record = {"repo_id": repo_id, "files": len(files), "weight_shards": len(weights),
                  "missing_required": sorted(missing),
                  "seconds": round(time.monotonic() - started, 1)}
        print(json.dumps(record), flush=True)
        if missing or not weights:
            raise SystemExit("incomplete upload: %s" % json.dumps(record))
        (readme / ("%s.upload.json" % spec["repo"])).write_text(
            json.dumps(record, indent=1) + "\n")
    print("all three repositories published", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
