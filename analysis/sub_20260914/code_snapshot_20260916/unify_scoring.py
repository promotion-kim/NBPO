"""Merge the judgment shards and re-score all 1000 prompts in one pass.

The solver refuses when score shards disagree on the frozen GPM teacher, and
the two separate scoring runs disagreed only on judgment_sources_sha256 (the
per-file hash map) and the BT fit counts. The teacher itself -- judge,
revision, objectives, both orders, draws per order, verdict aggregation -- was
identical.

Loosening that check was the wrong fix: its intent is that every shard came
from one frozen teacher, and the honest way to satisfy it is to score once over
all the verdicts rather than to stop hashing the provenance. So the extension's
judgment shards move in as shards 4..7 of the main tag and the scorer runs once
with 8 judge shards, producing a single teacher record covering all prompts.

The score shards merged by the earlier run are removed first so the re-score
cannot mix old and new.
"""
import json, shutil, sys
from pathlib import Path

UF = Path("/work/uf4_20260910")
S = Path("/work/sub_20260914")

src_root = S / "pool_judgments/pros2_psc_ext"
dst_root = S / "pool_judgments/pros2_psc"
moved = []
for k in range(4):
    src, dst = src_root / ("shard%d" % k), dst_root / ("shard%d" % (4 + k))
    if not src.is_dir():
        raise SystemExit("missing extension judgment shard: %s" % src)
    if dst.exists():
        print("already present:", dst)
        continue
    shutil.copytree(src, dst)
    moved.append({"from": str(src), "to": str(dst)})

# drop the score shards the first merge produced, so nothing stale survives
removed = []
for k in range(4, 8):
    d = UF / "scores/pros_scores_v1" / ("shard%d" % k)
    if d.exists():
        shutil.rmtree(d)
        removed.append(str(d))

verdicts = 0
for k in range(8):
    d = dst_root / ("shard%d" % k)
    for m in sorted(d.glob("chunk*.manifest.json")):
        verdicts += json.loads(m.read_text())["verdicts"]

report = {"judgment_shards_moved": len(moved),
          "stale_score_shards_removed": removed,
          "judgment_shards_total": 8,
          "verdicts_across_all_shards": verdicts,
          "next": ("re-score with --tag pros2_psc --judge-shards 8 --pool "
                   "pros2_pool_v1 --pool-shards 8, giving one GPM teacher record"),
          "detail": moved}
(S / "prosper/unify_scoring.json").write_text(json.dumps(report, indent=1) + "\n")
print(json.dumps({k: report[k] for k in
                  ("judgment_shards_moved", "stale_score_shards_removed",
                   "verdicts_across_all_shards")}, indent=1))
