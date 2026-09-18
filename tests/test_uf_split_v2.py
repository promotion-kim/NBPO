"""The UltraFeedback v2 prompt-level split and its decontamination.

Every earlier UltraFeedback run in this repository split at the *pair* level, so
one prompt could sit in training and in evaluation at once. These tests pin the
matcher that prevents that, and -- when the split has actually been built --
re-derive the disjointness assertions from the files on disk rather than
trusting the manifest that the builder wrote about itself.
"""
import json
from pathlib import Path

import pytest

import importlib.util

_SRC = Path(__file__).resolve().parents[1] / "scripts/experiments/iclr2027_table1_v2/build_ultrafeedback_split.py"
_spec = importlib.util.spec_from_file_location("uf_split_builder", _SRC)
builder = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(builder)

SPLIT_DIR = Path(__file__).resolve().parents[1] / "experiments/iclr2027_table1_v2/splits"
SPLIT_NAMES = ("train", "validation", "test")
BUDGETS = {"train": 7000, "validation": 500, "test": 1000}

built = pytest.mark.skipif(
    not (SPLIT_DIR / "split_manifest.json").exists(),
    reason="split not built yet (needs network); build_ultrafeedback_split.py writes it")


# --- normalization ----------------------------------------------------------

def test_normalization_collapses_whitespace_and_unicode_forms():
    assert builder.normalize_prompt("  a\n\t b  ") == "a b"
    # NFKC folds the compatibility ligature and the fullwidth digit
    assert builder.normalize_prompt("ﬁle １") == builder.normalize_prompt("file 1")
    assert builder.normalize_prompt("") == ""


def test_normalization_is_idempotent():
    for s in ("  x  y ", "ﬁ z", "a\r\nb"):
        once = builder.normalize_prompt(s)
        assert builder.normalize_prompt(once) == once


# --- the near-duplicate matcher --------------------------------------------

def test_exact_jaccard_endpoints():
    a = builder.shingles("the quick brown fox jumps over it")
    assert builder.exact_jaccard(a, a) == 1.0
    assert builder.exact_jaccard(a, builder.shingles("zzzzzzzzzzzzzzzzzzzzzzz")) == 0.0
    assert builder.exact_jaccard(set(), set()) == 1.0
    assert builder.exact_jaccard(a, set()) == 0.0


def test_minhash_estimates_jaccard():
    a = builder.shingles("a moderately long sentence about preference optimization")
    b = builder.shingles("a moderately long sentence about preference optimisation")
    est = sum(1 for x, y in zip(builder.minhash(a), builder.minhash(b)) if x == y) / builder.NUM_PERM
    assert abs(est - builder.exact_jaccard(a, b)) < 0.12


def test_lsh_proposes_the_near_duplicate_pair():
    """The LSH only ever proposes; the exact Jaccard decides. This checks the
    proposal actually fires for a pair above threshold."""
    texts = ["Explain how photosynthesis works in simple terms for a child.",
             "Explain how photosynthesis works in simple terms for a kid.",
             "Write a Python function that reverses a linked list in place."]
    sh = [builder.shingles(t) for t in texts]
    pairs = set(builder.candidate_pairs(builder.lsh_index([builder.minhash(s) for s in sh])))
    assert (0, 1) in pairs or (1, 0) in pairs
    assert builder.exact_jaccard(sh[0], sh[1]) >= 0.8
    assert builder.exact_jaccard(sh[0], sh[2]) < 0.8


def test_union_find_builds_connected_groups():
    u = builder.Union(5)
    u.union(0, 1)
    u.union(1, 2)
    u.union(3, 4)
    groups = {}
    for i in range(5):
        groups.setdefault(u.find(i), []).append(i)
    assert sorted(len(g) for g in groups.values()) == [2, 3]


# --- assertions against what was actually written --------------------------

def _load(name):
    return [json.loads(l) for l in
            (SPLIT_DIR / f"ultrafeedback_{name}.jsonl").read_text().splitlines() if l.strip()]


@built
@pytest.mark.parametrize("name", SPLIT_NAMES)
def test_split_has_the_declared_size(name):
    assert len(_load(name)) == BUDGETS[name]


@built
def test_splits_are_prompt_disjoint_under_exact_normalized_matching():
    sets = {n: {r["prompt"] for r in _load(n)} for n in SPLIT_NAMES}
    for a in SPLIT_NAMES:
        for b in SPLIT_NAMES:
            if a < b:
                assert not (sets[a] & sets[b]), f"{a} and {b} overlap"


@built
def test_no_prompt_appears_twice_inside_one_split():
    for n in SPLIT_NAMES:
        rows = _load(n)
        assert len({r["prompt"] for r in rows}) == len(rows)


@built
def test_prompt_ids_are_content_derived_and_unique():
    seen = set()
    for n in SPLIT_NAMES:
        for r in _load(n):
            assert r["prompt_sha256"] == builder.sha256_text(r["prompt"])
            assert r["prompt_id"] == f"uf2-{r['prompt_sha256'][:16]}"
            assert r["prompt_id"] not in seen
            seen.add(r["prompt_id"])


@built
def test_prompts_are_stored_in_normalized_form():
    for n in SPLIT_NAMES:
        for r in _load(n):
            assert builder.normalize_prompt(r["prompt"]) == r["prompt"]


@built
def test_near_duplicate_groups_do_not_straddle_splits():
    """A paraphrase in training and its twin in test would leak just as surely
    as an exact repeat, so grouping happens before splitting."""
    rows = {n: _load(n) for n in SPLIT_NAMES}
    sh = {n: [builder.shingles(r["prompt"]) for r in rows[n]] for n in SPLIT_NAMES}
    all_sh = [s for n in SPLIT_NAMES for s in sh[n]]
    owner = [n for n in SPLIT_NAMES for _ in sh[n]]
    buckets = builder.lsh_index([builder.minhash(s) for s in all_sh])
    for i, j in builder.candidate_pairs(buckets):
        if owner[i] != owner[j] and builder.exact_jaccard(all_sh[i], all_sh[j]) >= 0.8:
            pytest.fail(f"near-duplicate across {owner[i]} and {owner[j]}")


@built
def test_manifest_records_zero_overlap_and_matches_the_files():
    m = json.loads((SPLIT_DIR / "split_manifest.json").read_text())
    assert all(v == 0 for v in m["overlap_checks_exact"].values()), m["overlap_checks_exact"]
    assert m["seed"] == 20260907
    for n in SPLIT_NAMES:
        assert m["files"][n]["prompts"] == BUDGETS[n] == len(_load(n))
        import hashlib
        path = SPLIT_DIR / f"ultrafeedback_{n}.jsonl"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == m["files"][n]["sha256"]


@built
def test_decontamination_report_covers_every_table1_benchmark():
    rep = json.loads((SPLIT_DIR / "decontamination_report.json").read_text())
    assert set(rep["benchmarks"]) == {"arena_hard_v2", "alpaca_eval_2", "mt_bench",
                                      "ifeval", "truthfulqa"}
    for name, meta in rep["benchmarks"].items():
        assert meta["prompts"] > 0, f"{name} contributed no prompts"
    assert rep["jaccard_threshold"] == 0.8
    assert rep["ngram"] == 13
    assert "exact_matches_removed" in rep and "approximate_matches_removed" in rep
