"""Point the PROSPER copy at the Safe data and give it the same quantiser.

solve_prosper_targets.py takes its score, pool and split locations from module
constants rather than flags, so the copy has to name the Safe ones. Left as it
was, the job's --splits/--train-scores flags were rejected outright -- which
was the lucky outcome, because a version that accepted and ignored them would
have solved PROSPER targets on UF-4 data and written them under a safe_ name.
"""
from pathlib import Path

P = Path("/work/sub_20260914/code/solve_safe_prosper.py")
src = P.read_text()

pairs = [
    ('(ROOT / "scores/v1", ROOT / "pools/v1", "policy_train")',
     '(ROOT / "scores/safe_v1", ROOT / "pools/safe_v1", "policy_train")'),
    ('(ROOT / "scores/dev_v1", ROOT / "pools/dev_v1", "policy_dev")',
     '(ROOT / "scores/safe_dev_v1", ROOT / "pools/safe_dev_v1", "policy_dev")'),
    ('ROOT / "splits/v1"', 'ROOT / "splits/safe_v1cc"'),
]
for old, new in pairs:
    if src.count(old) != 1:
        raise SystemExit("expected exactly one %r, found %d" % (old, src.count(old)))
    src = src.replace(old, new, 1)

# the same canonical-target quantisation as the base copy
if "quantize_canonical_row" not in src:
    marker = "                for row in rows:\n"
    if src.count(marker) != 1:
        raise SystemExit("cannot find the pair-row loop (found %d)" % src.count(marker))
    src = src.replace(marker, marker + "                    row = base.quantize_canonical_row(row)\n", 1)

P.write_text(src)
import ast
ast.parse(src)
print("patched solve_safe_prosper.py: Safe paths + quantiser")
for old, new in pairs:
    print("   %s -> %s" % (old[-40:], new[-40:]))
