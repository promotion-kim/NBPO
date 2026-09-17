"""Is the 256-token cross-play parse collapse Qwen3 thinking-mode overrun?

The judge template leaves the assistant turn open unless `enable_thinking=False`
is passed, and none of the judge scripts passed it. Qwen3 then emits a reasoning
block before its answer. If that block is what consumed the 256-token budget,
the failing outputs should be truncated *inside* the reasoning -- their stored
tail never closes `</think>` and the generation stopped on length -- while the
successful ones should have closed it and gone on to a verdict.
"""
import json
import glob
from collections import Counter

for d in sorted(glob.glob("/work/uf4_20260910/evaluation/crossplay/pairs/*")):
    try:
        rows = [json.loads(l) for l in open(d + "/judgments.jsonl")]
    except FileNotFoundError:
        continue
    st = Counter(r["status"] for r in rows)
    fails = [r for r in rows if r["status"] != "ok"]
    oks = [r for r in rows if r["status"] == "ok"]
    n_ok = st.get("ok", 0)
    fail_open = sum(1 for r in fails if "</think>" not in r["raw"])
    fail_think = sum(1 for r in fails if "<think>" in r["raw"] or "</think>" in r["raw"])
    ok_closed = sum(1 for r in oks if "</think>" in r["raw"])
    tag = d.rsplit("/", 1)[1]
    print("%-64s n=%5d ok=%5d (%.1f%%)  fail_tail_no_close=%5d  fail_tail_mentions_think=%5d  ok_tail_closed=%5d"
          % (tag[:64], len(rows), n_ok, 100.0 * n_ok / max(1, len(rows)), fail_open, fail_think, ok_closed))
    if fails:
        print("      failing tail  :", repr(fails[0]["raw"][-200:]))
    if oks:
        print("      succeeding tail:", repr(oks[0]["raw"][-200:]))
