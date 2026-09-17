"""Add the contract's missing-data bounds to the bank aggregator.

The complete case conditions on every one of a prompt-objective's 76 verdicts
parsing, which discards most of the panel. The protocol therefore also requires
bounds: put every unresolved preference at 0 and at 1 and report the interval
those give, on all 200 planned prompts rather than on the complete subset.

For the direct win rate the bound is linear in the missing entries. For the
finite-bank surplus the value is monotone in each margin, so the conservative
range pairs the lower policy value with the upper reference value and the
reverse, as the appendix states.
"""
from pathlib import Path

PATH = Path("/work/uf4_20260910/code/diag_aggregate_bank.py")
src = PATH.read_text()

OLD = """    # ---- whole-prompt paired bootstrap over the complete intersection"""
NEW = '''    # ---- bounds: every unresolved preference at 0 and at 1, on all 200 prompts
    bounds = {}
    for criterion in CRITERIA:
        raw = {pid: matrices(cells, pid, criterion) for pid in ids}
        for name, table in dists.items():
            lo_w, hi_w, lo_s, hi_s, used = [], [], [], [], 0
            for pid in ids:
                if pid not in table:
                    continue
                A, B, _ = raw[pid]
                p = np.asarray(table[pid])
                A_lo = np.where(np.isfinite(A), A, -0.5)
                A_hi = np.where(np.isfinite(A), A, 0.5)
                lo_w.append(0.5 + float(p @ A_lo @ mu))
                hi_w.append(0.5 + float(p @ A_hi @ mu))
                # B is filled by antisymmetry; unresolved reference pairs stay 0,
                # so its bounds move with the same +/-0.5 envelope on zero entries
                zero = (B == 0) & ~np.eye(N_REF, dtype=bool)
                B_lo = np.where(zero, -0.5, B)
                B_hi = np.where(zero, 0.5, B)
                lo_s.append(softmin(A_lo.T @ p, mu, BETA_EVAL) - softmin(B_hi.T @ mu, mu, BETA_EVAL))
                hi_s.append(softmin(A_hi.T @ p, mu, BETA_EVAL) - softmin(B_lo.T @ mu, mu, BETA_EVAL))
                used += 1
            if used:
                bounds.setdefault(name, {})[criterion] = {
                    "n_planned": used,
                    "win_rate_lower": float(np.mean(lo_w)),
                    "win_rate_upper": float(np.mean(hi_w)),
                    "surplus_lower": float(np.mean(lo_s)),
                    "surplus_upper": float(np.mean(hi_s)),
                }

    # ---- judge contract on this panel: actual self-pairs and order sensitivity
    # ---- whole-prompt paired bootstrap over the complete intersection'''
assert src.count(OLD) == 1
src = src.replace(OLD, NEW, 1)

OLD2 = """        "bootstrap": summary,
        "paired_differences": diffs,"""
NEW2 = """        "bootstrap": summary,
        "bounds_all_planned_prompts": bounds,
        "paired_differences": diffs,"""
assert src.count(OLD2) == 1
src = src.replace(OLD2, NEW2, 1)
PATH.write_text(src)
import ast
ast.parse(src)
print("bounds added and parsed")
