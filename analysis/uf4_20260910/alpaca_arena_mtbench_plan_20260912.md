# AlpacaEval-2, Arena-Hard-v2 and MT-Bench at zero API cost

Requested 2026-09-12 12:45 KST: add these three to Table 3
(`tab:general-capability`) and run them with no paid API, at the front of the
queue behind the running experiment.

## What zero API cost forces

All three official protocols use a GPT-4-class annotator:

| benchmark | official scorer | available here? |
|---|---|---|
| AlpacaEval 2 | GPT-4-turbo annotator, length-controlled WR | no (paid) |
| Arena-Hard-v2 | GPT-4 judge, Bradley-Terry score vs baseline | no (paid) |
| MT-Bench | GPT-4 single-answer grade 1-10, two turns | no (paid) |

So the official numbers cannot be produced under this constraint, and the table
must not imply otherwise. The substitute for the first two is a **pinned local
reward model**, Skywork-Reward-V2-Qwen3-8B, through the already-verified
`evaluate_responses.py --mode skywork` path. That code stamps its own output
with `"not_official": "Not AlpacaEval LC or official Arena-Hard score"`, and the
new column heading carries a dagger pointing at the same caveat.

## AlpacaEval-2 and Arena-Hard-v2: queued

Both prompt sets were already on disk and were checked before queueing:

- `data/alpaca_eval.jsonl`, 804 prompts (805 with the header row counted)
- `data/arena_hard.jsonl`, 749 prompts: 500 `hard_prompt` + 250
  `creative_writing`, which is the Arena-Hard-**v2** composition, and the scorer
  already reports those two subsets separately

Metric: order-free win rate against the same cached base response per prompt,
scored by the local RM, ties on exact scalar equality, 16,384-token RM limit
with over-length pairs recorded as failures rather than dropped.

Pre-checks done before the jobs reach a card, after three queue-wide failures
earlier in this campaign taught the habit:

- `items_for` routes both benchmarks through the shared builder: supported
- context headroom: AlpacaEval max prompt 518 tokens, Arena-Hard max 6,384,
  against a 16,384 window less 2,048 generated: fits
- `assert_matched_protocol` compares decoding and tokenizer fields only, not the
  benchmark list, and every one of those eleven fields on the existing `base`
  label equals the generator's defaults: passes
- fresh `uf4ab_<arm>` labels, because `generate_eval` refuses when an existing
  `settings.json` records a different benchmark list

Eleven arms are queued (all trained arms except maxmin seed 44, which is still
training and will be added when it finishes). Priority 48/49: immediately behind
the running experiment's own generation and judging, ahead of the remaining
cross-play pairs, BT-RM, the DPO arms and PROSPER.

## MT-Bench: not queued yet, and why

`data/mt_bench_question.jsonl` is present with the standard 80 questions, each
carrying two turns. Two things are missing and neither is a detail:

1. **Multi-turn generation.** `generate_eval` builds a single user message per
   prompt. MT-Bench's second turn is conditioned on the model's own first
   answer, so scoring only `turns[0]` would not be MT-Bench, it would be a
   different 80-prompt benchmark wearing the name.
2. **A declared local grading rule.** The official score is an absolute 1-10
   GPT-4 grade. A local substitute has to choose between a local judge grading
   1-10 (comparable across arms only if the rubric and parse are fixed first)
   and a pairwise win rate against the base (comparable, but no longer an
   MT-Bench score at all).

Both are buildable. Reporting them as "MT-Bench" without declaring the
substitution would be the problem, so the column is added and left unmeasured
until the grading rule is declared.
