"""Have the pool-reweighting job keep the per-candidate log-ratios it computes.

tab:target-signal needs the fitted policy's pair log-ratio h_theta, which is a
difference of the same per-candidate deltas this job already forms in order to
reweight the pool. Saving them costs nothing and avoids a second forward pass.
"""
from pathlib import Path

PATH = Path("/work/uf4_20260910/code/diag_neural_pools.py")
src = PATH.read_text()

OLD = """        payload[arm] = table
        diagnostics[arm] = {"""
NEW = """        payload[arm] = table
        deltas[arm] = {pid: [float(arm_logp[c] - base_logp[c]) for c in candidates[pid]]
                       for pid in ids}
        diagnostics[arm] = {"""
assert src.count(OLD) == 1
src = src.replace(OLD, NEW, 1)

OLD2 = """    base_logp = score(args.base)
    payload, diagnostics = {}, {}"""
NEW2 = """    base_logp = score(args.base)
    payload, diagnostics, deltas = {}, {}, {}"""
assert src.count(OLD2) == 1
src = src.replace(OLD2, NEW2, 1)

OLD3 = """    Path(args.out).write_text(json.dumps(payload, indent=2) + "\\n")"""
NEW3 = """    Path(args.out).write_text(json.dumps(payload, indent=2) + "\\n")
    (DIAG / "pool_log_ratios.json").write_text(json.dumps(deltas) + "\\n")"""
assert src.count(OLD3) == 1
src = src.replace(OLD3, NEW3, 1)

PATH.write_text(src)
import ast
ast.parse(src)
print("patched: per-candidate log-ratios are now saved")
