"""Safe-objective copies of the two solvers, one constant changed each.

The UF-4 solvers hardcode the four UltraFeedback objectives and validate the
score tensors' first axis against them, so a two-objective panel cannot pass
through them. Rather than add a flag to campaign code whose byte-identity is
part of the UF campaign's reproducibility record, this writes copies with
OBJECTIVES replaced and records the source hash inside each copy.

Everything else -- loaders, hashing, dataset writers, certificates -- is the
same code path the UF arms used.
"""
import hashlib
import re
from pathlib import Path

SRC = Path("/work/uf4_20260910/code")
DST = Path("/work/sub_20260914/code")
OBJ_OLD = 'OBJECTIVES = ("instruction_following", "truthfulness", "honesty", "helpfulness")'
OBJ_NEW = 'OBJECTIVES = ("helpfulness", "harmlessness")'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def note(source, source_sha, changes):
    return ('"""Generated copy -- do not edit by hand.\n\n'
            'Copied from %s\n'
            'source sha256 %s\n'
            'changes: %s\n'
            'Reason: the Safe policy panel has two global objectives '
            '(helpfulness, harmlessness).\n"""\n' % (source, source_sha, changes))


def main():
    base_src = SRC / "solve_uf4_targets.py"
    base_sha = sha(base_src)
    text = base_src.read_text()
    assert text.count(OBJ_OLD) == 1, "objective constant not found exactly once"
    text = text.replace(OBJ_OLD, OBJ_NEW, 1)
    text = note(base_src, base_sha, "OBJECTIVES -> helpfulness, harmlessness") + text
    (DST / "solve_safe_targets.py").write_text(text)

    pro_src = SRC / "solve_prosper_targets.py"
    pro_sha = sha(pro_src)
    ptext = pro_src.read_text()
    assert ptext.count("import solve_uf4_targets as base") == 1
    ptext = ptext.replace("import solve_uf4_targets as base",
                          "import solve_safe_targets as base", 1)
    ptext = note(pro_src, pro_sha,
                 "base module -> solve_safe_targets (two objectives)") + ptext
    (DST / "solve_safe_prosper.py").write_text(ptext)

    for name in ("solve_safe_targets.py", "solve_safe_prosper.py"):
        print("  %-28s %s" % (name, sha(DST / name)[:16]))
    print("  source solve_uf4_targets.py     %s" % base_sha[:16])
    print("  source solve_prosper_targets.py %s" % pro_sha[:16])


if __name__ == "__main__":
    main()
