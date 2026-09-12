# The base HarmBench cell comes from a different campaign

Found 2026-09-12 09:45 KST while checking what blocks the two safety columns of
`tab:general-capability`.

The base row of that table currently reads IFEval 0.7542, GSM8K 0.8666,
HB 0.2813. The first two are UF measurements. The third is 0.28125 from
`/work/nbpo_repair_20260909/analysis_claude/capability.json`, which is the
**SafeRLHF repair campaign**, not the UF campaign.

The appendix already says the UF and historical SafeRLHF base rows "use
different recorded protocols and are not pooled". A single row carrying two
campaigns' protocols is precisely what that sentence warns against, and nothing
in the table marks the cell as borrowed.

Note also that a *third* HarmBench number exists for a "base" label:
`/work/nbpo_downstream_local_v1/scores/harmbench.json` records
`harmful_completion_rate = 0.328125`. So "the base HarmBench value" is not one
number in this repository, which is another reason to measure it here rather
than cite it.

## What is being measured instead

The protocol is the one already recorded in that downstream artifact, not a new
one:

- classifier `cais/HarmBench-Llama-2-13b-cls`, local snapshot `bda705349d`
- metric: direct-request harmful-completion rate over the 320 text behaviours
- classifier generation budget 512 tokens, with truncation counts recorded
- explicitly **not** adversarial robustness: no jailbreak template, no attack
  search. The artifact states this and the paper must repeat it.

Generation and scoring both run through the verified
`evaluate_responses.py --mode harmbench` path, on the ten trained UF arms and
the base, so every cell in the column shares one protocol.

## XSTest is not queued, and why

`evaluate_responses.py` has an `--mode xstest`, but its own docstring describes
the output as "unadjudicated responses + explicitly diagnostic keywords". A
keyword-diagnostic refusal count is not an adjudicated refusal rate, and turning
one into a headline safety number requires declaring an adjudication rule. That
declaration has not been made, so the XS column stays unmeasured rather than
being filled with a keyword proxy presented as a refusal rate.
