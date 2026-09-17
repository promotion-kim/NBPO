"""Sweep status, reading only the CURRENT run of each job.

The controller appends to one stdout.log per job and writes a
"===== <time> launching <job> =====" banner each time it starts, so a job that
was stopped and requeued has its old progress bars still in the file. Parsing
the whole file reported a killed run's eval-loop counter as training progress.
Everything below is read after the last banner only, and the recorded state is
printed beside it so a stale artifact cannot masquerade as live progress.
"""
import json, re
from pathlib import Path

R = Path("/work/uf4_20260910")
STATE = json.loads((R / "jobs/state.json").read_text())["jobs"]
EVAL = re.compile(r"\{'eval_loss'.*?\}")
STEP = re.compile(r"(\d+)/(\d+) \[[^\]]*\]")
BANNER = re.compile(r"=====[^\n]*launching[^\n]*=====")
ORDER = ["repro", "lr1e6", "lr3e6", "lr1e5", "lr1e6_e4"]

for a in ORDER:
    job = "uw2_nbpopw_%s_train" % a
    st = STATE.get(job, {}).get("state", "no record")
    log = R / "jobs/runs" / job / "stdout.log"
    if not log.exists():
        print("%-10s %-9s (no log)" % (a, st)); continue
    txt = log.read_text(errors="replace").replace("\r", "\n")
    cuts = list(BANNER.finditer(txt))
    current = txt[cuts[-1].end():] if cuts else txt
    hits = EVAL.findall(current)
    steps = STEP.findall(current)
    prog = "%s/%s" % steps[-1] if steps else "-"
    if hits:
        m = eval(hits[-1])
        keep = {k.replace("eval_nbpo/", ""): round(v, 5) for k, v in m.items()
                if any(s in k for s in ("nmse", "sign_accuracy", "logratio_rms",
                                        "second_moment", "eval_loss"))}
        print("%-10s %-9s step=%-9s %s" % (a, st, prog, json.dumps(keep)))
    else:
        print("%-10s %-9s step=%-9s runs_in_log=%d" % (a, st, prog, len(cuts)))
