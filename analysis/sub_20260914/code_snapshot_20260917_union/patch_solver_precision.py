"""Make the pair rows survive the dataset loader's 10-decimal JSON parse.

prepare_nbpo_dataset validates that every row's canonical target equals
log(w_a/c_a) - log(w_b/c_b) to 1e-9, recomputed from the row's own masses. The
loader it reads through keeps only ten decimal places, so the check is really
run on rounded masses against an unrounded target. UF-4 survived that because
its Nash masses were around 5e-2, where a 1e-10 absolute rounding is a 1e-9
relative change and the log difference shifts by about 1e-9. The Safe solution
concentrates mass, leaving masses near 6e-5, where the same rounding is a 5e-7
relative change and the target shifts by 1e-6 -- three orders past tolerance.

So the masses are rounded to ten decimals HERE, before writing, and the target
is recomputed from exactly those rounded masses. The row is then internally
consistent under both an exact and a ten-decimal parse. The change is a 1e-6
relative quantisation of the regression target on the smallest masses, which is
far below the judge noise the target is fitted from, and it is recorded in the
provenance rather than left implicit.
"""
from pathlib import Path

P = Path("/work/sub_20260914/code/solve_safe_targets.py")
src = P.read_text()

helper = '''
CANONICAL_DECIMALS = 10


def quantize_canonical_row(row):
    """Round the solver masses to the loader's precision and rebuild the target.

    prepare_nbpo_dataset reads these rows through a JSON parser that keeps ten
    decimal places, then checks target == log(w_a/c_a) - log(w_b/c_b) to 1e-9.
    Rounding the masses first and deriving the target from the rounded values
    makes the row consistent under that parse instead of only before it.
    """
    import math

    for key in ("nbpo_weight_a", "nbpo_weight_b"):
        if key in row:
            row[key] = round(float(row[key]), CANONICAL_DECIMALS)
    if row.get("target_mode") == "canonical_logratio" and "nbpo_weight_a" in row:
        wa, wb = float(row["nbpo_weight_a"]), float(row["nbpo_weight_b"])
        ca = float(row.get("nbpo_center_a", 1.0 / POOL))
        cb = float(row.get("nbpo_center_b", 1.0 / POOL))
        if min(wa, wb) > 0:
            row["nbpo_logratio_target"] = round(
                math.log(wa / ca) - math.log(wb / cb), CANONICAL_DECIMALS)
            row["canonical_target_quantized_decimals"] = CANONICAL_DECIMALS
    return row

'''

anchor = "def write_jsonl(path, rows):"
assert src.count(anchor) == 1
src = src.replace(anchor, helper.lstrip("\n") + "\n" + anchor, 1)

old = """                rows = build_rows("""
assert src.count(old) == 1
# wrap the generator's yielded rows
old_yield_block = src[src.index("                rows = build_rows("):]
# find the loop that yields rows and insert the quantizer
marker = "                for row in rows:\n"
if marker in src:
    src = src.replace(marker, marker + "                    row = quantize_canonical_row(row)\n", 1)
    where = "for-row loop"
else:
    # fall back: quantize the whole list right after build_rows returns
    tail = src.index(")\n", src.index("                rows = build_rows(")) + 2
    src = src[:tail] + "                rows = [quantize_canonical_row(r) for r in rows]\n" + src[tail:]
    where = "post-build_rows list comprehension"

P.write_text(src)
print("patched solve_safe_targets.py via %s" % where)
