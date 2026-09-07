# The opaque-identifier factorial, and the end of the prompted-judge route

Run 2026-09-08 on `judge_v4_dev` only. Neither holdout was opened.
24 000 renderings in 1 023 s (200 natural pairs x 24 cells, 800 known-answer
controls x 24 cells), Qwen3-32B, protocol `P2_long` byte-identical apart from the
identifier substitution. Artifacts: `opaque_factorial_{report.md,metrics.json}`,
`run_manifest.json`.

## Design

Three factors fully crossed, so that two things the ordinary rendering confounds
*perfectly* come apart exactly:

| factor | levels |
|---|---|
| semantic | which response is the learner `y` |
| physical position | which response occupies the first slot |
| identifier order | `AB` = first identifier on slot 1 (the canonical rendering); `BA` = first identifier on slot 2 |
| identifier scheme | `A`/`B` versus the neutral `K7`/`Q2` |
| template | t0, t1, t2 |

Every cell holds exactly 600 observations — balance re-derived from the written
file, not asserted. Pairs were chosen by the lowest salted hash of the pair id
per objective, so selection is **outcome-blind**: choosing on decisiveness or on
disagreement would bias the very effect being estimated.

Tokenization was verified rather than assumed. `K7` and `Q2` are **two** tokens
each (`[42, 22]`, `[48, 17]`) and `[[K7]]` is four. That is harmless here because
P2 parses a marker out of generated text, but the same identifiers could not be
dropped into P1's restricted single-token scoring, and the fact is recorded so
nobody does that later.

## The bias is lexical, and it is strong enough to beat a known answer

| objective | scheme | position effect [95% CI] | identifier effect [95% CI] |
|---|---|---|---|
| helpfulness | A/B | +0.012 [−0.013, +0.037] | **+0.237 [+0.183, +0.287]** |
| instruction following | A/B | +0.026 [+0.002, +0.051] | **+0.147 [+0.086, +0.207]** |
| honesty | A/B | −0.020 [−0.041, +0.000] | +0.017 [−0.010, +0.046] |
| truthfulness | A/B | −0.016 [−0.046, +0.013] | −0.019 [−0.072, +0.030] |

Under A/B the physical slot does **almost nothing** and the letter does almost
everything. On the two broken criteria the judge is following the identifier "A",
not the position and not the content.

The sharpest evidence is the known-answer control. In the factorial's A/B
condition, deterministic-degradation accuracy falls to **0.480 / 0.680 / 0.980 /
0.640**. Under opaque identifiers on the *same* controls it is **1.000 / 1.000 /
1.000 / 0.940**. The letter "A" is strong enough to override a deterministic,
objective-specific content degradation whose correct answer is not in doubt.

*This is not a re-measurement of the frozen protocol.* The frozen P2-long always
renders `AB`, and it scores 0.960–1.000 on these same controls. The factorial
averages over a labelling the frozen protocol never uses; that is precisely how
it isolates the letter effect.

## Opaque identifiers relocate the bias; they do not remove it

| objective | scheme | position effect | identifier effect | **total slot-cue bias** |
|---|---|---|---|---|
| helpfulness | A/B | +0.012 | +0.237 | **+0.249** |
| helpfulness | opaque | **+0.168 [+0.120, +0.219]** | +0.068 [+0.042, +0.098] | **+0.236** |
| instruction following | A/B | +0.026 | +0.147 | **+0.173** |
| instruction following | opaque | **+0.158 [+0.103, +0.215]** | −0.005 [−0.026, +0.017] | **+0.153** |
| honesty | A/B | −0.020 | +0.017 | −0.003 |
| honesty | opaque | −0.004 | −0.016 | −0.020 |
| truthfulness | A/B | −0.016 | −0.019 | −0.035 |
| truthfulness | opaque | +0.017 | −0.017 | +0.000 |

The total advantage the canonical rendering confers on the first-and-"A" response
is **conserved** across schemes on the two broken criteria: 0.249 → 0.236 and
0.173 → 0.153. Removing the letters does not remove the bias; the judge simply
falls back on the first slot.

Opaque identifiers do make the instrument measurably better on every other axis:
mean |Δ| rises from 0.043 to 0.207 on helpfulness, decisive pairs from 4/50 to
38/50, order split-half Spearman from **−0.232 to +0.356**, and the deterministic
controls are fully restored. It is a real improvement. It is not enough.

## The pre-registered decision, applied without adjustment

Seven conjunctive gates. Both schemes:

| gate | threshold | A/B | opaque |
|---|---|---|---|
| unresolved invalid rate | < 0.002 | **PASS** (≤ 0.001) | **PASS** (0.000) |
| deterministic degradation | ≥ 0.90 | FAIL (0.480) | **PASS** (0.940) |
| identical-pair confident tie | ≥ 0.90 | **PASS** (1.000) | **PASS** (1.000) |
| confident semantic swap | ≥ 0.85 | FAIL (0.000–0.600) | **FAIL (0.448–0.750)** |
| order split-half Spearman | ≥ 0.75 | FAIL (−0.232–0.046) | **FAIL (0.008–0.465)** |
| finite-pool target Pearson | ≥ 0.90 | not evaluated | not evaluated |
| target sign agreement | ≥ 0.85 | not evaluated | not evaluated |

The two downstream target gates were **not run**. They require a separate
full-pool judging pass, and the gates are conjunctive, so two judge-level
failures already decide the outcome. That is a cost decision, stated; it is not a
relaxation, and running them could only have confirmed a decision already made.

> **Decision B. The prompted-Qwen training-judge route is permanently retired
> from the main experiment.** No further prompt variants will be created and no
> threshold will be lowered. `judge_v4_holdout` and the backup holdout are never
> opened; the 7 500-prompt bank is never launched; no reward model and no policy
> is trained from any v2/v3/v4/v5 prompted-Qwen label.

## What the route cost, and what it bought

Four protocol generations established, in order: the swap mapping is correct
(v2); the pool of same-model samples is near-degenerate and both Qwen3-32B and
Llama-3.3-70B tie heavily on it (v2 diagnosis); the 96-token budget was
truncating verdicts and that is fixable (v3 → v4); and now, that what remains is
a slot-cue bias whose channel can be changed but whose magnitude cannot. The
useful residue is the last line: **a prompted LLM judge asked for a pairwise
verdict carries a first-slot advantage of about 0.15–0.25 on helpfulness and
instruction following that is invariant to the identifier scheme.** That is worth
reporting on its own terms, and it is the reason the project now moves to a
learned anti-symmetric preference model trained on released human annotations,
where antisymmetry is a property of the architecture rather than something a
prompt has to be talked into.
