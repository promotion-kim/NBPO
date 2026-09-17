"""Give the aggregator what the stage figure needs: fresh CIs and stage differences.

Two additions, both from the same whole-prompt paired resample the table already
uses:

* a per-arm interval for every fresh row, on that arm's own complete prompts;
* paired NBPO-minus-utilitarian differences at the fitted-pool stage and at the
  fresh stage, so panel (b) can show the same contrast at all three stages
  rather than only at the target.
"""
from pathlib import Path

PATH = Path("/work/uf4_20260910/code/diag_aggregate_bank.py")
src = PATH.read_text()

OLD = """    payload = {
        "panel_sha256": panel["panel_sha256"],"""
NEW = '''    # ---- fresh rows: per-arm interval and the paired NBPO-minus-utilitarian stage
    fresh_ci, fresh_diffs = {}, {}
    fresh_per_prompt = {}
    for path in sorted(glob.glob(str(Path(args.fresh_dir) / "*.jsonl"))):
        arm = Path(path).stem
        fcells, _ = read_verdicts(path)
        per_c = {}
        for criterion in CRITERIA:
            vals = {}
            for pid in ids:
                pairs = fcells.get((pid, criterion), {})
                got = [0.5 * (v[0][0] + v[1][0]) for v in pairs.values()
                       if 0 in v and 1 in v]
                if len(got) == N_REF:
                    vals[pid] = float(np.mean(got))
            per_c[criterion] = vals
        fresh_per_prompt[arm] = per_c
        shared = sorted(set.intersection(*[set(per_c[c]) for c in CRITERIA])) or []
        if not shared:
            continue
        d = rng.integers(0, len(shared), size=(args.replicates, len(shared)))
        entry, mins = {}, []
        for criterion in CRITERIA:
            x = np.array([per_c[criterion][pid] for pid in shared])
            b = x[d].mean(axis=1)
            entry[criterion] = {"win_rate": float(x.mean()),
                                "ci95": [float(np.percentile(b, 2.5)),
                                         float(np.percentile(b, 97.5))]}
            mins.append(b)
        w = np.min(np.stack(mins), axis=0)
        entry["w_min"] = {"point": float(np.min([entry[c]["win_rate"] for c in CRITERIA])),
                          "ci95": [float(np.percentile(w, 2.5)), float(np.percentile(w, 97.5))]}
        entry["n"] = len(shared)
        fresh_ci[arm] = entry

    a_arm, b_arm = "nbpo_mse_s42", "util_mse_s42"
    if a_arm in fresh_per_prompt and b_arm in fresh_per_prompt:
        A, B = fresh_per_prompt[a_arm], fresh_per_prompt[b_arm]
        shared = sorted(set.intersection(*[set(A[c]) for c in CRITERIA],
                                        *[set(B[c]) for c in CRITERIA]))
        if shared:
            d = rng.integers(0, len(shared), size=(args.replicates, len(shared)))
            fresh_diffs = {"n": len(shared)}
            for criterion in CRITERIA:
                x = np.array([A[criterion][pid] for pid in shared])
                y = np.array([B[criterion][pid] for pid in shared])
                b = (x - y)[d].mean(axis=1)
                fresh_diffs[criterion] = {
                    "point": float((x - y).mean()),
                    "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]}

    pool_diffs = {}
    if "neural_nbpo_mse_s42" in boot and "neural_util_mse_s42" in boot:
        for criterion in CRITERIA:
            b = (boot["neural_nbpo_mse_s42"][criterion]["w"]
                 - boot["neural_util_mse_s42"][criterion]["w"])
            pool_diffs[criterion] = {
                "point": (summary["neural_nbpo_mse_s42"][criterion]["win_rate_common"]
                          - summary["neural_util_mse_s42"][criterion]["win_rate_common"]),
                "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]}

    payload = {
        "panel_sha256": panel["panel_sha256"],'''
assert src.count(OLD) == 1
src = src.replace(OLD, NEW, 1)

OLD2 = """        "bounds_all_planned_prompts": bounds,
        "paired_differences": diffs,"""
NEW2 = """        "bounds_all_planned_prompts": bounds,
        "paired_differences": diffs,
        "fresh_intervals": fresh_ci,
        "fresh_nbpo_minus_util": fresh_diffs,
        "pool_nbpo_minus_util": pool_diffs,"""
assert src.count(OLD2) == 1
src = src.replace(OLD2, NEW2, 1)
PATH.write_text(src)
import ast
ast.parse(src)
print("stage intervals and differences added")
