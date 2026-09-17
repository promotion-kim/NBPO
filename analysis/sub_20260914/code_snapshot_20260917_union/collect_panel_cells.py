"""Turn a panel's measured artifacts into the manuscript's own result cells.

Reads only artifacts, never a transcript: the objective-wise matrix for the
W1/W2/Wmin columns of tab:union-objectives, and the solve records and training
logs for that panel's two rows of tab:union-diagnostic-template. Emits the
cells file that inject_union_results.py consumes, which refuses any key the
manuscript has not already declared.

Method ids are the manuscript's, not the pipeline's: the arm directory
`pw_nbpo` is the row `nbpopw`, `dpo_soft` is `dpo`, `inpo_soft` is `inpo`. An
arm with no measurement is simply absent from the output, so its cell stays
\\pending rather than being filled with something else.
"""
from __future__ import annotations

import argparse, json, re
from pathlib import Path

UF = Path("/work/uf4_20260910")
SUB = Path("/work/sub_20260914")

METHOD = {"base": "base", "dpo_soft": "dpo", "inpo_soft": "inpo",
          "prosper": "prosper", "nbpo": "nbpo", "pw_nbpo": "nbpopw",
          "mopo": "mopo", "modpo": "modpo"}
NMSE = re.compile(r"'eval_nbpo/nmse':\s*([0-9.eE+-]+)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", required=True, help="us1 or ut1")
    ap.add_argument("--key-prefix", required=True, help="US or UT")
    ap.add_argument("--out", required=True)
    ap.add_argument("--diagnostic-arms", nargs="*", default=["nbpo", "pw_nbpo"],
                    help="arms that have a row in the diagnostic table")
    args = ap.parse_args()

    p, P = args.panel, args.key_prefix
    cells, skipped = {}, {}

    # --- objective-wise wins -------------------------------------------------
    mpath = SUB / "objectivewise" / ("%s_objwise" % p) / "matrix.json"
    if mpath.exists():
        m = json.loads(mpath.read_text())
        for arm, row in m["table"].items():
            method = METHOD.get(arm)
            if method is None:
                skipped["arm:%s" % arm] = "no manuscript method id"
                continue
            for cell in ("w1", "w2", "wmin"):
                if cell not in row:
                    continue
                cells["%s.%s.%s" % (P, method, cell)] = {
                    "value": row[cell]["value"],
                    "ci": row[cell]["ci"],
                    "contains_half": row[cell]["contains_half"],
                    "source": ("mean over %d common test prompts of the "
                               "order-balanced probability against the fixed base "
                               "draw; W_min is the minimum over objectives"
                               % m["prompts"]["common_to_every_cell"]),
                    "artifact": str(mpath)}
    else:
        skipped["objectivewise"] = "no matrix at %s" % mpath

    # --- solver and cost diagnostics ----------------------------------------
    for arm in args.diagnostic_arms:
        method = METHOD.get(arm)
        done = UF / "targets" / ("%s_%s" % (p, arm)) / "complete.json"
        if method is None or not done.exists():
            skipped["diag:%s" % arm] = "no solve record at %s" % done
            continue
        rec = json.loads(done.read_text())
        cells["%s.%s.stages" % (P, method)] = {
            "value": 1, "source": "one Algorithm-1 outer stage",
            "artifact": str(done)}

        res = SUB / "diag" / ("%s_%s_residuals.json" % (p, arm))
        if res.exists():
            r = json.loads(res.read_text())["reported"]
            cells["%s.%s.inner_residual" % (P, method)] = {
                "value": r["inner_residual_max_over_both_splits"],
                "source": r["definition_inner"], "artifact": str(res)}
            cells["%s.%s.dual_residual" % (P, method)] = {
                "value": r["dual_residual_max_over_both_splits"],
                "source": r["definition_dual"], "artifact": str(res)}
        else:
            skipped["residuals:%s" % arm] = "no residual record at %s" % res

        run = UF / "jobs/runs" / ("%s_train_%s" % (p, arm))
        launch, exit_path = run / "launch.json", run / "exit.json"
        log = run / "stdout.log"
        if log.exists():
            hits = NMSE.findall(log.read_text())
            if hits:
                cells["%s.%s.nmse" % (P, method)] = {
                    "value": float(hits[-1]),
                    "source": "eval_nbpo/nmse at the final dev evaluation",
                    "artifact": str(log)}
        if launch.exists() and exit_path.exists():
            import datetime as dt
            fmt = "%Y-%m-%dT%H:%M:%SZ"
            a = dt.datetime.strptime(json.loads(launch.read_text())["started"], fmt)
            b = dt.datetime.strptime(json.loads(exit_path.read_text())["finished"], fmt)
            gpus = int(json.loads(launch.read_text())["gpus"])
            cells["%s.%s.gpu_h" % (P, method)] = {
                "value": gpus * (b - a).total_seconds() / 3600.0,
                "source": ("%d GPUs x %.0f s of this arm's own training; the pool and "
                           "judging are panel-level and counted once in the budget "
                           "ledger" % (gpus, (b - a).total_seconds())),
                "artifact": str(launch)}

        kl = SUB / "diag" / ("%s_pool_kl.json" % p)
        if kl.exists():
            d = json.loads(kl.read_text())["arms"].get("%s_%s" % (p, arm))
            if d:
                cells["%s.%s.pool_kl" % (P, method)] = {
                    "value": d["pool_kl_star_to_neural_mean"],
                    "source": ("mean over dev prompts of KL(p* || p_neural) on the "
                               "sampled support"),
                    "artifact": str(kl)}

    Path(args.out).write_text(json.dumps({"cells": cells, "skipped": skipped},
                                         indent=1) + "\n")
    print(json.dumps({"cells": len(cells), "keys": sorted(cells),
                      "skipped": skipped}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
