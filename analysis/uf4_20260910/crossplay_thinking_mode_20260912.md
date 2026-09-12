# The cross-play parse-rate collapse: cause found, runs retired

Date: 2026-09-12 (KST), during the UF-4 main-results campaign.

## What was wrong

Cross-play judging parsed **2.6-6.9 %** of its judgments at the declared
256-token output limit. Raising the limit to 1024 tokens gave 81.8-83.3 % and
2048 tokens gave 88.0-91.1 %. The byte-identical final-evaluation judge -- same
model, same revision, same rubric file, same template, same verdict grammar,
same 256-token limit -- parses **99.4 %** across 190,960 judgments. That gap was
recorded as unexplained after ruling out template, regex, budget, prompt length,
response length and judge-input truncation.

## Cause

`judge_crossplay_pair.py` called

    tok.apply_chat_template([...], add_generation_prompt=True, tokenize=True)

while `judge_final_eval.py` and `run_audit_judge.py` call the same function with
`enable_thinking=False`. The Qwen3 chat template leaves the assistant turn open
unless that flag is passed, so the cross-play judge ran in **thinking mode**: it
spent its output budget on a reasoning block and the verdict never arrived. The
final-eval judge never did.

The manuscript's own declared evaluator contract (Appendix, audit protocol)
reads "Run the local Qwen3-14B judge with thinking disabled ... and a $256$-token
output limit", so this was a departure from a protocol declared in advance, not
a discretionary choice.

## Evidence

Failing outputs are truncated *inside* the reasoning block; successful ones
closed it and went on to a verdict. Counting whether the stored 300-character
tail ever closes `</think>`:

| run | n | parsed | failing tails with no `</think>` |
|---|---|---|---|
| base vs fixedref, 256 tok | 4000 | 127 (3.2 %) | 3813 / 3873 = 98.5 % |
| base vs nbpo, 256 tok | 4000 | 105 (2.6 %) | 3816 / 3895 = 97.9 % |
| fixedref vs nbpo, 256 tok | 4000 | 275 (6.9 %) | 3661 / 3725 = 98.3 % |
| base vs fixedref, 2048 tok | 4000 | 3518 (88.0 %) | 398 / 482 = 82.6 % |
| base vs nbpo, 2048 tok | 4000 | 3544 (88.6 %) | 367 / 456 = 80.5 % |
| fixedref vs nbpo, 2048 tok | 4000 | 3645 (91.1 %) | 285 / 355 = 80.3 % |

A representative failing tail is mid-sentence deliberation
("...Now, the user's instruction was to judge which response better follows the
instruction in terms of form") and a representative succeeding tail ends
`</think>\n\n[[TIE]]`.

This also explains why the dev-selection runs reached 99.4 % at 256 tokens while
cross-play did not: those runs went through `judge_final_eval.py`, which passes
the flag. The earlier hypothesis -- that judging two near-identical trained
policies simply takes more deliberation than judging a policy against the base
-- was never needed.

## What was done

1. `judge_crossplay_pair.py` now passes `enable_thinking=False`, records it in
   the run report, and keeps the declared 256-token limit as its default.
2. All three thinking-mode runs were moved aside under dotted directory names
   (`<pair>.thinkon_t2048_parse88pct` and the 1024/256 variants), which the
   aggregator and the progress counters already skip. Nothing was deleted.
3. `analysis/crossplay_summary.json` was set aside the same way.
4. All six pairs are requeued as `uf4_crossplay_judge_<pair>_think0` at
   priority 64, followed by `uf4_crossplay_aggregate_think0`.
5. Both cross-play exhibits were rewritten to `\pending`. This required a fix
   in `progress/fill_exhibits.py`: it previously skipped the marked regions
   when no summary existed, which left retracted numbers standing in the
   manuscript. Absence of a summary now rewrites the exhibits as unmeasured.

## What this does not affect

Every number in Table 1 (`tab:uf-objectives`), the dev-selection of the
cross-play competitor, and the audit panel came through judges that pass the
flag, at parse rates of 99.38-99.58 %. The main results are untouched.

## Cost

The retired runs cost about 2.5 GPU-hours. They stay in the cost accounting:
they were really spent.

## Result of the re-judgment (measured 2026-09-12 14:33-14:47 KST)

Parse rates at the declared 256-token limit with thinking off, all six pairs:

| pair | parsed | was (2048, thinking on) |
|---|---|---|
| base vs fixedref_mse_s42 | 3968/4000 = 99.2 % | 88.0 % |
| base vs nbpo_mse_s42 | 3974/4000 = 99.3 % | 88.6 % |
| base vs util_mse_s42 | 3978/4000 = 99.5 % | not run |
| nbpo_mse_s42 vs fixedref_mse_s42 | 3977/4000 = 99.4 % | 91.1 % |
| fixedref_mse_s42 vs util_mse_s42 | 3990/4000 = 99.8 % | not run |
| nbpo_mse_s42 vs util_mse_s42 | see summary | not run |

That is the final-eval judge's own range (99.38-99.58 %), which is what a
protocol match should look like. Complete prompts per cell rose from 365-442 to
487-500 of 500, so every cell clears the declared 100-prompt floor with room.

**The retired numbers were shifted, not merely noisy.** Comparing the same
twelve cells under both settings, the corrected win rate moves by $-0.019$ to
$+0.025$, and the movement is not centred on zero: honesty rises by $+0.021$
and $+0.025$ in the two base-vs-arm pairs while instruction following and
truthfulness fall by up to $0.019$. Missing judgments were therefore not
missing at random with respect to the verdict -- the comparisons the judge
deliberated longest over are exactly the ones it failed to return, and those
lean a particular way per rubric. Keeping the 2048-token numbers "because the
parse rate was acceptable" would have kept a biased estimate.
