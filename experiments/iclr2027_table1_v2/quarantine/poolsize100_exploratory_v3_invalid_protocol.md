# QUARANTINED — exploratory_v3_invalid_protocol

Everything in this directory was produced by **judge protocol v3 / P2_deliberative**,
which was subsequently found inadmissible: its final unresolved invalid rate is
4.0-20.9% per objective against a 0.2% gate, because the 96-token budget often
ends before the verdict marker.

## Status

* `scored_4x4/` — complete (8800 pairs, finished 05:50:35 PDT)
* `scored_8x8/` — **stopped mid-run** at ~19 minutes on 2026-09-07, by design.
  Empty: the runner writes its results file only at the end, so there is no
  partial result to salvage. The pools, prompt list and pair lists are intact.
* pools, `pairs_4x4.jsonl`, `pairs_8x8.jsonl`, manifests — preserved unchanged.

## What this may not be used for

* selecting 4+4 versus 8+8 pool geometry;
* any number in the paper;
* combination with any future v4 measurement.

The pool-size comparison must be **repeated from scratch** with a judge protocol
that has passed a fresh holdout, on a new development set, per the governing
instruction. Reusing a measurement taken through a broken instrument to choose
the geometry the next instrument will run on would launder the defect forward.

## Why it is kept

It is evidence about the failure, not about the pool. The 4+4 arm in particular
records how P2 behaved on 100 unseen prompts and can be cited when explaining
why v3 was retired.
