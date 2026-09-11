import json, re
from datetime import datetime
from pathlib import Path
ROOT = Path("/work/uf4_20260910")
start, out = {}, {}
for line in (ROOT / "logs/queue_report.jsonl").open():
    e = json.loads(line)
    j = e["job_id"]
    if not j.startswith("uf4_train_"):
        continue
    if e["state"] == "RUNNING":
        start[j] = e["time"]
    elif e["state"] in ("DONE", "FAILED") and j in start:
        d = (datetime.strptime(e["time"], "%Y-%m-%dT%H:%M:%SZ")
             - datetime.strptime(start[j], "%Y-%m-%dT%H:%M:%SZ")).total_seconds()
        out.setdefault(j, []).append((d / 3600.0, e["state"]))
for j in sorted(out):
    runs = out[j]
    arm = j[len("uf4_train_"):]
    cfg = ROOT / "configs" / f"{arm}.yaml"
    es = "?"
    if cfg.exists():
        m = re.search(r"^eval_steps:\s*(\d+)", cfg.read_text(), re.M)
        es = m.group(1) if m else "?"
    total = sum(h for h, _ in runs)
    print("%-26s runs=%d  wall %5.2fh  GPU %6.2fh  eval_steps=%s  states=%s"
          % (arm, len(runs), total, total * 4, es, ",".join(s for _, s in runs)))
