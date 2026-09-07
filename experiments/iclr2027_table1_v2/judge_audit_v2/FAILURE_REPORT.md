# v2 judge protocol — failure report

The v2 hard-verdict protocol did not pass its swap-consistency gate and will not label the final experiment. This is the frozen evidence for that decision. **No v2 label is reused in the v3 bank.**

- git commit: `dc4831dbd89a94ee0d4406f763b5b1fdbf5a693d`
- rubric: `/work/iclr27_table1_v2/code/training_configs/nbpo/objectives/ultrafeedback_v2.yaml` sha256 `934d3062d67cff0d46975c12ffe1330c…`

## First: is the swap mapping itself correct?

Before blaming the judge, the arithmetic that turns two ordered verdicts into one semantic preference was pinned against hand-constructed cases (`tests/test_judge_swap_mapping.py`):

| forward verdict | reverse verdict | p_hat(y > z) |
|---|---|---|
| A | B | 1.0 |
| B | A | 0.0 |
| A | A | 0.5  (first-position contradiction) |
| B | B | 0.5  (second-position contradiction) |
| TIE | TIE | 0.5 |

Reference tensor exactly skew-symmetric with a zero diagonal: **True**.

**the v2 semantic swap mapping is CORRECT; the consistency failure is not a mapping bug.**

Note the mechanically important row: a judge that always picks whichever response is shown first, and a judge that genuinely finds the two equal, produce the *same* tensor entry. The tensor cannot distinguish noise from indifference, which is why a hard-verdict protocol was the wrong instrument for a pool of same-policy samples.

## qwen3_32b

- bank: `/work/iclr27_table1_v2/smoke/verdicts.jsonl` (8800 rows), sha256 `349230dcd2ba3b8f38be94e2af1a536a…`
- judge model: `/work/models/xj_judges/qwen3-32b`
- parser-invalid rate: **0.0000%**, retry rate 0.0000%
- mean |Delta| over all pairs: **0.1798** (maximum possible 0.5)

| objective | calls | pairs | decisive | consistency | contradiction | tie/tie | one-order tie | position bias | mean \|D\| | \|D\|=0 |
|---|---|---|---|---|---|---|---|---|---|---|
| helpfulness | 2200 | 1100 | 659 | 0.645 | 0.355 | 0.185 | 0.216 | +0.1445 | 0.2473 | 39.7% |
| honesty | 2200 | 1100 | 346 | 0.592 | 0.408 | 0.527 | 0.158 | -0.0391 | 0.1327 | 65.5% |
| instruction_following | 2200 | 1100 | 411 | 0.696 | 0.304 | 0.384 | 0.243 | -0.0041 | 0.1907 | 49.7% |
| truthfulness | 2200 | 1100 | 347 | 0.663 | 0.337 | 0.509 | 0.175 | -0.0314 | 0.1484 | 61.5% |

## llama33_70b

- bank: `/work/iclr27_table1_v2/diag/verdicts_v2_llama70b.jsonl` (8810 rows), sha256 `d5e617e6eee5805030d97b503d479b58…`
- judge model: `/work/models/xj_judges/llama70`
- parser-invalid rate: **0.1703%**, retry rate 0.1135%
- mean |Delta| over all pairs: **0.1176** (maximum possible 0.5)

| objective | calls | pairs | decisive | consistency | contradiction | tie/tie | one-order tie | position bias | mean \|D\| | \|D\|=0 |
|---|---|---|---|---|---|---|---|---|---|---|
| helpfulness | 2200 | 1100 | 527 | 0.433 | 0.567 | 0.256 | 0.265 | +0.3595 | 0.1698 | 52.8% |
| honesty | 2200 | 1100 | 162 | 0.710 | 0.290 | 0.693 | 0.160 | +0.0836 | 0.0923 | 73.5% |
| instruction_following | 2200 | 1100 | 161 | 0.571 | 0.429 | 0.616 | 0.237 | +0.1395 | 0.1011 | 67.9% |
| truthfulness | 2210 | 1096 | 176 | 0.807 | 0.193 | 0.670 | 0.170 | +0.0684 | 0.1072 | 70.1% |

## What the numbers say

**Position bias is the mechanism, and it is judge-specific.** Llama-3.3-70B carries a large positive bias toward the A slot on every objective, reaching **+0.36 on helpfulness** -- which is exactly why its decisive consistency there (0.433) sits *below* chance: when both orders commit, it is mostly committing to whichever response it was shown first. Qwen3-32B is milder and mixed in sign (+0.14 on helpfulness, within +/-0.04 elsewhere).

**Swap averaging cancels the systematic part of that bias but cannot recover the lost information.** A pair the judge decides oppositely in the two orders lands at |Delta| = 0, the same value a genuine tie produces. Between 40% and 74% of all pairs end up there, and mean |Delta| is 0.18 (Qwen) and 0.12 (Llama) against a maximum of 0.5. The measurement is unbiased and badly underpowered.

**Neither judge is the fix.** The 70B is better on truthfulness and honesty only because it abstains far more (it ties both orders on 69% and 67% of those pairs), and it is worse on the other two. No judge tested clears 0.85 on more than one objective.

This is why the protocol moves to calibrated, uncertainty-aware scoring rather than to a different judge or a lower threshold: the instrument has to be able to *say* it is uncertain, instead of encoding uncertainty as a coin flip that the tensor then cannot distinguish from indifference.

## What this bank is for

Protocol development only. It is not the v3 gate set, and none of these labels may be mixed into the v3 judgment bank.
