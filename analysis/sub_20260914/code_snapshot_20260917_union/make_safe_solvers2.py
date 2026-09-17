"""Safe-objective solver copies, with provenance as COMMENTS not a docstring.

The first attempt prepended a module docstring, which made the original
docstring a plain expression statement and pushed `from __future__ import
annotations` out of first position. Python rejects that, so all four solves
died with a SyntaxError before reading a single tensor. Comments are legal
before a __future__ import; a second docstring is not.
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
    return ("# Generated copy -- do not edit by hand.\n"
            "# Copied from %s\n"
            "# source sha256 %s\n"
            "# changes: %s\n"
            "# Reason: the Safe policy panel has two global objectives\n"
            "# (helpfulness, harmlessness), and the UF-4 solver validates the score\n"
            "# tensors' first axis against its own four.\n" % (source, source_sha, changes))


def main():
    base_src = SRC / "solve_uf4_targets.py"
    base_sha = sha(base_src)
    text = base_src.read_text()
    assert text.count(OBJ_OLD) == 1
    text = note(base_src, base_sha, "OBJECTIVES -> helpfulness, harmlessness") + \
        text.replace(OBJ_OLD, OBJ_NEW, 1)
    (DST / "solve_safe_targets.py").write_text(text)

    pro_src = SRC / "solve_prosper_targets.py"
    pro_sha = sha(pro_src)
    ptext = pro_src.read_text()
    assert ptext.count("import solve_uf4_targets as base") == 1
    ptext = note(pro_src, pro_sha, "base module -> solve_safe_targets") + \
        ptext.replace("import solve_uf4_targets as base",
                      "import solve_safe_targets as base", 1)
    (DST / "solve_safe_prosper.py").write_text(ptext)

    import ast
    for name in ("solve_safe_targets.py", "solve_safe_prosper.py"):
        path = DST / name
        ast.parse(path.read_text())
        head = path.read_text().splitlines()
        first_stmt = next(i for i, l in enumerate(head)
                          if l.strip() and not l.startswith("#"))
        print("  %-26s parses; first non-comment line %d: %s"
              % (name, first_stmt + 1, head[first_stmt][:48]))
    import subprocess
    for name in ("solve_safe_targets.py", "solve_safe_prosper.py"):
        r = subprocess.run(["python3", "-c",
                            "import sys; sys.path.insert(0, '%s'); "
                            "import %s as m; print('  %s OBJECTIVES =', m.OBJECTIVES)"
                            % (DST, name[:-3], name)],
                           capture_output=True, text=True,
                           env={"PYTHONPATH": "/work/pylibs_eval:%s:%s:/work/nbpo_repair_20260909/code"
                                % (DST, SRC), "PATH": "/usr/bin:/bin",
                                "HF_HUB_OFFLINE": "1"})
        print((r.stdout or r.stderr).strip().splitlines()[-1][:120])


if __name__ == "__main__":
    main()
