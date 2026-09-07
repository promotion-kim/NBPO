# What PKU-SafeRLHF's released annotations can and cannot support

Built 2026-09-08 by `scripts/experiments/iclr2027_table1_v2/build_saferlhf_splits.py`
at seed 20260908; manifest with all hashes at `saferlhf_splits/split_manifest.json`.
The split files themselves are 122 MB of raw responses and are reproducible from
the builder, so they are not in the repository.

## The splits

Prompt-level, never row-level: a prompt appears in a mean of 1.9 rows and in up
to 95, so a row split would leak a prompt's responses across sides. Assignment is
`sha256("saferlhf-split-20260908-" + prompt_sha256) -> [0,1)`, which is
reproducible without carrying an RNG state and is stable if the dataset grows.

| split | rows | prompts | helpfulness (frac resp-1) | harmlessness (frac resp-1) | **helpfulness/harmlessness conflict** |
|---|---|---|---|---|---|
| train | 59 624 | 31 052 | 0.640 | 0.710 | 0.2407 |
| validation | 7 328 | 3 833 | 0.636 | 0.711 | 0.2394 |
| test | 6 955 | 3 754 | 0.636 | 0.708 | **0.2433** |

Pairwise prompt overlap is 0 on all three pairings, re-derived from the written
files rather than from the assignment map. Duplicated prompts are kept -- they
are distinct comparisons, not duplicated labels -- and travel together.
The released fields are binary ids with no tie encoding, so **no tie is invented**.

## The finding that matters: there are no observable cycles, and there cannot be

Counted two ways with two independent definitions of response identity (the
released `response_*_sha256`, and a hash of the NFKC-normalised text), over the
full 73 907 rows:

```
rows 73907   prompts 38639   nodes 143668   distinct edges 73906
nodes with degree >= 2 :   1369   of 143668   (0.95%)
triangles              :      0
```

**The comparison graph is a near-perfect matching.** Each response appears in
exactly one annotated comparison, apart from 1 369 responses out of 143 668. A
three-cycle needs all three pairs of a triple annotated; not one such triple
exists anywhere in the dataset, for either objective.

The consequence is structural, not statistical, and no amount of data from this
source changes it:

* **the directly observable three-cycle count is 0, by design of the annotation
  protocol** -- and 0 is what is reported, per split, in the manifest;
* a "human preferences are intransitive" claim **cannot be evidenced from
  PKU-SafeRLHF**, and must not be;
* the pre-registered stop condition "no defensible benefit over BT on any
  observable nontransitive subset" cannot be *evaluated* here, because that
  subset is empty. That is reported as unavailable, not as a pass.

Prompts with at least three distinct responses do exist (2 387 of 3 754 in test,
19 785 of 31 052 in train), and 1 369 of 3 754 test prompts have a connected
comparison graph -- but connected
with two edges over four nodes is impossible, so those are the two-response
prompts. Neither statistic rescues the triangle count.

## What SafeRLHF does support

A genuine, large, *two-dimensional* conflict signal: **24.33 % of test rows have
`better_response_id != safer_response_id`**, i.e. the more helpful response is
the less safe one. These are independent human judgements on the same pair, not
induced from a single score, so the conflict is observed rather than constructed.

That is the property the bargaining argument actually needs -- objectives that
disagree about the same pair -- and it is far stronger here than in the panels
this project has used before. Intransitivity is a different property, and this
dataset is silent on it.

## Contrast with UltraFeedback

UltraFeedback's four aspects match the paper's terms exactly and are complete on
63 966 of 63 967 rows, but they are **per-completion ordinal ratings**, so
induced pairwise labels inherit a total order per objective and are transitive by
construction. UltraFeedback may serve only as a transitive score-induced
aggregation control. It may not be cited as cyclicity evidence.
