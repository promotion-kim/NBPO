"""Manuscript patches must not introduce dangling references.

Each file in `nbpo_iclr/patches/` is a drop-in replacement for a named block of
`main_v5.tex`. A patch that compiles only after someone hunts down a missing
label is a patch that will be applied wrong, so every `\\ref` and `\\eqref` it
makes is checked against the manuscript here -- except labels the patch set
defines itself.
"""
import re
from pathlib import Path

import pytest

PATCH_DIR = Path("nbpo_iclr/patches")
MANUSCRIPT = Path("nbpo_iclr/main_v5.tex")
REF = re.compile(r"\\(?:eq)?ref\{([^}]+)\}")
LABEL = re.compile(r"\\label\{([^}]+)\}")


def _patches():
    return sorted(PATCH_DIR.glob("*.tex"))


@pytest.mark.skipif(not MANUSCRIPT.exists(), reason="manuscript not present")
def test_every_patch_reference_resolves():
    manuscript_labels = set(LABEL.findall(MANUSCRIPT.read_text()))
    patch_labels = set()
    for f in _patches():
        patch_labels |= set(LABEL.findall(f.read_text()))
    known = manuscript_labels | patch_labels
    assert _patches(), "no patches found"

    dangling = {}
    for f in _patches():
        missing = sorted({r for r in REF.findall(f.read_text()) if r not in known})
        if missing:
            dangling[f.name] = missing
    assert not dangling, f"patches reference labels that do not exist: {dangling}"


@pytest.mark.skipif(not MANUSCRIPT.exists(), reason="manuscript not present")
def test_patches_do_not_silently_redefine_a_manuscript_label():
    """A patch that reuses an existing label would produce a duplicate at compile."""
    manuscript_labels = set(LABEL.findall(MANUSCRIPT.read_text()))
    # These are the blocks each patch REPLACES, so redefining them is intended.
    replaced = {"alg:nbpo"}
    for f in _patches():
        clashes = sorted({l for l in LABEL.findall(f.read_text())
                          if l in manuscript_labels and l not in replaced})
        assert not clashes, f"{f.name} redefines manuscript labels: {clashes}"


def test_patch_readme_lists_every_patch():
    readme = (PATCH_DIR / "README.md").read_text()
    for f in _patches():
        assert f.name in readme, f"{f.name} is not documented in patches/README.md"


def test_no_patch_claims_the_whole_algorithm_is_exact():
    """The terminology contract, enforced where it is easiest to break.

    Only the *adjectival* use is policed. "exactly mirror ascent" is an adverb
    describing an identity between two maps and is fine; "the exact algorithm"
    is the claim this project must never make.
    """
    overclaim = re.compile(
        r"exact\s+(?:neural\s+)?(?:NBPO\s+)?(?:algorithm|method|pipeline)"
        r"|(?:algorithm|method|pipeline)\s+is\s+exact",
        re.IGNORECASE)
    adjectival = re.compile(r"\bexact\b(?!ly)", re.IGNORECASE)
    scoping = ("direct finite-pool concave inner solve",
               "not describe the algorithm", "never describe the algorithm",
               "exact by Theorem", "exact population")
    for f in _patches():
        text = f.read_text()
        assert not overclaim.search(text), f"{f.name} overclaims exactness"
        if adjectival.search(text):
            assert any(p in text for p in scoping), (
                f"{f.name} uses 'exact' adjectivally without scoping it")
