"""Merge the 400-prompt extension in as shards 4..7, matching the real layout.

The loaders read <root>/shard<k>/complete_shard<k>.json plus chunk*.npz inside
that directory, so a shard is a DIRECTORY and its completion file carries the
shard index in its name. The first version of this script assumed flat
scores_shard<k>.npz files at the root and would have merged nothing while
reporting success.

Every step is counted and verified. load_scores already raises on a duplicate
prompt id, and this checks the count independently so a partial merge cannot
pass as a whole panel.
"""
import hashlib, json, shutil, sys
from pathlib import Path

import numpy as np

UF = Path("/work/uf4_20260910")
S = Path("/work/sub_20260914")
NS = "sub_20260914-prosper-policy-split:"
DEV_FRACTION = 0.10
PAIRS = (("pool", UF / "pools/pros2_pool_v1", UF / "pools/pros2_pool_ext"),
         ("scores", UF / "scores/pros_scores_v1", UF / "scores/pros_scores_ext"))

moved = []
for kind, main, ext in PAIRS:
    if not ext.exists():
        raise SystemExit("extension directory missing: %s" % ext)
    for k in range(4):
        src = ext / ("shard%d" % k)
        dst = main / ("shard%d" % (4 + k))
        if not src.is_dir():
            raise SystemExit("missing extension shard: %s" % src)
        if dst.exists():
            print("already merged:", dst)
            continue
        shutil.copytree(src, dst)
        old_c = dst / ("complete_shard%d.json" % k)
        new_c = dst / ("complete_shard%d.json" % (4 + k))
        if old_c.exists():
            rec = json.loads(old_c.read_text())
            rec["shard"] = 4 + k
            rec["merged_from"] = str(src)
            new_c.write_text(json.dumps(rec, indent=2) + "\n")
            old_c.unlink()
        moved.append({"kind": kind, "from": str(src), "to": str(dst)})

# independent verification: count prompts across all eight score shards
all_pids = []
for k in range(8):
    d = UF / "scores/pros_scores_v1" / ("shard%d" % k)
    c = d / ("complete_shard%d.json" % k)
    if not c.exists():
        raise SystemExit("score shard %d has no completion file at %s" % (k, c))
    for p in sorted(d.glob("chunk*.npz")):
        z = np.load(p, allow_pickle=True)
        all_pids += [str(x) for x in z["prompt_ids"]]
if len(all_pids) != len(set(all_pids)):
    raise SystemExit("duplicate prompt ids across merged score shards")

# the pool must cover every scored prompt, or the solver refuses later
pool_pids = set()
for k in range(8):
    d = UF / "pools/pros2_pool_v1" / ("shard%d" % k)
    for p in sorted(d.glob("chunk*.jsonl")):
        with p.open() as f:
            for line in f:
                if line.strip():
                    pool_pids.add(json.loads(line)["prompt_id"])
missing_pool = [p for p in all_pids if p not in pool_pids]
if missing_pool:
    raise SystemExit("%d scored prompts have no pool entry, e.g. %s"
                     % (len(missing_pool), missing_pool[:3]))

scored = set(all_pids)
rows, seen = [], set()
for name in ("pros_v1/pros_train.jsonl", "pros_ext/pros_train.jsonl"):
    for line in (UF / "splits" / name).open():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["prompt_id"] in scored and r["prompt_id"] not in seen:
            seen.add(r["prompt_id"])
            rows.append(r)
if len(rows) != len(scored):
    raise SystemExit("%d scored prompts but %d split rows matched"
                     % (len(scored), len(rows)))
ordered = sorted(rows, key=lambda r: hashlib.sha256((NS + r["prompt_id"]).encode()).hexdigest())
n_dev = max(20, int(round(len(ordered) * DEV_FRACTION)))
dev, train = ordered[:n_dev], ordered[n_dev:]
out_dir = UF / "splits/pros_v1m"
out_dir.mkdir(parents=True, exist_ok=True)
for name, part in (("policy_train", train), ("policy_dev", dev)):
    with (out_dir / ("%s.jsonl" % name)).open("w") as f:
        for r in part:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

report = {"shard_dirs_moved": len(moved), "score_shards": 8,
          "prompts_in_merged_scores": len(all_pids),
          "prompts_in_merged_pool": len(pool_pids),
          "policy_train": len(train), "policy_dev": len(dev),
          "learner_pairs_available": len(train) * 28,
          "split_dir": str(out_dir),
          "split_rule": "sort scored prompt ids by SHA256('%s' + id); first %d are dev" % (NS, n_dev),
          "detail": moved}
(S / "prosper/merge_ext.json").write_text(json.dumps(report, indent=1) + "\n")
print(json.dumps({k: report[k] for k in
                  ("shard_dirs_moved", "prompts_in_merged_scores", "prompts_in_merged_pool",
                   "policy_train", "policy_dev", "learner_pairs_available")}, indent=1))
