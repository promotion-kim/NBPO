# v4 development — final, frozen 2026-09-08

This record closes v4. It is written before any holdout is opened and states the
outcome in the terms the pre-registered rule uses, so that nothing downstream has
to re-derive it.

## Candidates and their evidence

Both candidates are scored against the **same frozen protocol's own controls**;
the pairing is now enforced in code (`assert_same_protocol`) after a real
mispairing was caught while assembling this record — see "A pairing bug" below.

| candidate | protocol sha | dev run | control run | renderings |
|---|---|---|---|---|
| `P2_long` (3 templates t0/t1/t2) | `33962ad4275f…` | `v4_dev/scored_dev_P2_long_t3` | `v4_controls2/P2_long` | 52 800 + 4 800 |
| `P2_balanced` (t0/t1, balanced analysis order) | `a91658156dda…` | `v4_dev/scored_dev_P2_balanced` | `v4_dev/scored_controls_P2_balanced` | 35 200 + 3 200 |

Development set: `judge_v4_dev`, 8 800 same-policy pairs over 4 objectives,
prompt-disjoint from both holdouts and from every training/validation/test split.

## Known-answer eligibility (checked before ranking)

| gate | threshold | P2_long | P2_balanced |
|---|---|---|---|
| final unresolved invalid rate | < 0.002 | **0.0000** pass | **0.0000** pass |
| deterministic degradation accuracy | ≥ 0.90 | **0.960** pass (min over objectives) | **0.800** FAIL (honesty 0.800, truthfulness 0.800) |
| identical-pair confident tie | ≥ 0.90 | **1.000** pass | **1.000** pass |

`P2_long` is the **only eligible candidate and is therefore selected**.
`P2_balanced` is **rejected on known-answer controls**: making the analysis order
symmetric cost the ability to detect a deterministic honesty or truthfulness
edit, and did not reduce position bias (its max |bias| is 0.486, worse than
P2_long's 0.404).

## Selected ≠ admissible: the stability gates fail

`P2_long` is the best candidate and still not usable.

| gate | threshold | helpfulness | honesty | instruction following | truthfulness |
|---|---|---|---|---|---|
| confident semantic swap | ≥ 0.85 | **0.494** | **0.708** | **0.412** | **0.691** |
| order split-half Spearman | ≥ 0.75 | **0.304** | **0.428** | **0.189** | **0.404** |
| signed position bias | — | **+0.364** | +0.037 | **+0.404** | +0.016 |
| contradiction rate | — | 0.514 | 0.315 | 0.586 | 0.329 |

Both stability gates fail on every objective. **P2_long is not an admissible
final judge protocol.** No threshold was lowered and no variant was created to
chase one.

## The decoding problem is solved, and it was only a decoding problem

The v3 budget was 96 new tokens. Measured verdict-token positions under the
repair:

| objective | p50 | p90 | p99 | max |
|---|---|---|---|---|
| helpfulness | 52 | 75 | 98 | 218 |
| honesty | 61 | 94 | 131 | 246 |
| instruction following | 64 | 94 | 131 | 333 |
| truthfulness | 66 | 107 | 151 | 241 |

p99 straddles 96 on three of four objectives and the maximum reaches 333, which
is exactly the 4–21 % invalid rate v3 suffered. At 512 (+1024 retry) the final
unresolved invalid rate is 0.0000 everywhere and first-pass invalid is 0.0005 at
worst. That component is fixed and stays fixed.

## What the failure looks like

The instability is **concentrated, not diffuse**. Helpfulness and
instruction-following carry +0.36/+0.40 position bias and swap consistency below
chance; honesty and truthfulness carry +0.04/+0.02 and reach ≈0.70. A uniform
"the judge is noisy" account does not fit. That is what the Section-1 opaque-ID
factorial is designed to decompose: the **lexical** effect of the identifier
"A" against the **physical** effect of being shown first, which the standard
rendering confounds perfectly.

## A pairing bug, found while writing this record

The `v4_dev/scored_controls_P2_long` directory holds controls judged by the
**2-template cost-parity cut** (sha `ac8fcab2bf6d`), not by the restored
3-template protocol. Scoring the 3-template dev run against them moves
truthfulness deterministic-degradation accuracy from 0.960 to **0.860** — across
a hard eligibility gate, in the direction of a false rejection. `analyze_v4.py`
now derives each results file's sibling manifest and refuses to mix two
protocols; two tests pin it.

## Standing consequences

* `judge_v4_holdout` is **unopened**; `judge_v4_backup_holdout` is **sealed**.
* The 7 500-prompt judgment bank is **not launched** and stays blocked.
* No reward model and no policy has been trained from any v2/v3/v4
  prompted-Qwen label.
* No v2/v3/v4 label may appear in a final paper table, and rows from different
  vintages may never be mixed.
* The prompted-Qwen training-judge route remains **unvalidated**. One bounded
  diagnostic remains (opaque identifiers); if it fails the route is retired for
  the main paper.
