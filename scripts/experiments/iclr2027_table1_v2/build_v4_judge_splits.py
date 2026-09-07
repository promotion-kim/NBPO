#!/usr/bin/env python3
"""Carve three immutable judge-protocol prompt groups, disjoint from everything used so far.

Judge-protocol work has now consumed prompts three times over, and reusing any
of them would make a "fresh" holdout an evaluation on data the protocol has
already been shaped against. These groups are cut from UltraFeedback records
that appear in **none** of:

  train / validation / final test, the five Table-1 benchmarks, the v2 audit
  pool, the v3 calibration controls, the v3 200-prompt holdout, and the
  quarantined v3 pool-size pilot.

Three groups:

  judge_v4_dev              100  protocol selection
  judge_v4_holdout          200  the binding evaluation, opened once
  judge_v4_backup_holdout   200  **sealed** -- opened only if v4 itself needs
                                 revision, so a second attempt does not have to
                                 reuse the first attempt's holdout

Grouping runs before assignment: near-identical prompts are merged by the same
character 13-gram MinHash the dataset split uses, and whole groups are assigned,
so a paraphrase cannot straddle dev and holdout.

Existing train/validation/test membership is not touched.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "build_ultrafeedback_split.py"
_spec = importlib.util.spec_from_file_location("uf_split_builder", _SRC)
sm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sm)

GROUPS = [("judge_v4_dev", 100), ("judge_v4_holdout", 200),
          ("judge_v4_backup_holdout", 200)]


def sha256_text(t):
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def load_used(paths, key="prompt"):
    out = set()
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            v = r.get(key)
            if v:
                out.add(sm.normalize_prompt(v))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--splits-dir", type=Path, required=True)
    ap.add_argument("--used", nargs="*", default=[],
                    help="extra jsonl files whose prompts are already consumed")
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--jaccard", type=float, default=0.8)
    ap.add_argument("--cache-dir", default=None)
    args = ap.parse_args()

    used = load_used([args.splits_dir / f"ultrafeedback_{n}.jsonl"
                      for n in ("train", "validation", "test")] + list(args.used))
    n_repo = len(used)
    bench_meta = {}
    for name, spec in sm.BENCHMARKS.items():
        prompts, meta = sm.load_benchmark_prompts(name, spec, args.cache_dir)
        used.update(prompts)
        bench_meta[name] = meta
    print(f"[1/4] {n_repo} prompts already consumed by repo files, "
          f"{len(used) - n_repo} more by benchmarks", flush=True)

    from datasets import load_dataset
    ds = load_dataset("openbmb/UltraFeedback", split="train", cache_dir=args.cache_dir)
    seen, eligible = set(), []
    for row in ds:
        p = sm.normalize_prompt(row["instruction"])
        if not p or p in used or p in seen:
            continue
        seen.add(p)
        eligible.append({"prompt": p, "source": row.get("source")})
    print(f"[2/4] {len(eligible)} eligible unused prompts", flush=True)

    rng = random.Random(args.seed)
    rng.shuffle(eligible)
    need = sum(n for _, n in GROUPS)
    pool = eligible[:need * 4]                     # headroom for group merging

    print("[3/4] grouping near-duplicates before assignment", flush=True)
    sh = [sm.shingles(r["prompt"]) for r in pool]
    buckets = sm.lsh_index([sm.minhash(s) for s in sh])
    uf = sm.Union(len(pool))
    merged = 0
    for i, j in sm.candidate_pairs(buckets):
        if sm.exact_jaccard(sh[i], sh[j]) >= args.jaccard:
            uf.union(i, j)
            merged += 1
    groups = {}
    for i in range(len(pool)):
        groups.setdefault(uf.find(i), []).append(i)
    glist = sorted(groups.values(), key=len, reverse=True)
    print(f"      {len(glist)} groups over {len(pool)} candidates "
          f"({merged} near-duplicate pairs merged)", flush=True)

    assigned = {name: [] for name, _ in GROUPS}
    budget = dict(GROUPS)
    for g in glist:
        fits = [n for n, _ in GROUPS if len(assigned[n]) + len(g) <= budget[n]]
        if not fits:
            continue
        target = min(fits, key=lambda n: (budget[n] - len(assigned[n]), n))
        assigned[target].extend(g)
    for name, want in GROUPS:
        if len(assigned[name]) != want:
            raise SystemExit(f"FATAL: {name} got {len(assigned[name])}, wanted {want}")

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"seed": args.seed, "jaccard_threshold": args.jaccard,
                "eligible_unused_prompts": len(eligible),
                "near_duplicate_pairs_merged": merged,
                "excluded_sources": [str(x) for x in args.used],
                "benchmarks": bench_meta, "files": {},
                "sealed": {"judge_v4_backup_holdout":
                           "must remain unread unless v4 itself requires revision"}}
    written = {}
    for name, _ in GROUPS:
        path = out / f"{name}.jsonl"
        rows = []
        for i in sorted(assigned[name]):
            p = pool[i]["prompt"]
            rows.append({"prompt_id": f"{name}-{sha256_text(p)[:16]}", "prompt": p,
                         "prompt_sha256": sha256_text(p), "group": name,
                         "source": pool[i]["source"]})
        path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
        written[name] = {r["prompt"] for r in rows}
        manifest["files"][name] = {
            "path": str(path), "prompts": len(rows),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        print(f"      {name}: {len(rows)} -> {path}", flush=True)

    print("[4/4] overlap checks", flush=True)
    checks = {}
    names = [n for n, _ in GROUPS]
    for a in names:
        for b in names:
            if a < b:
                k = len(written[a] & written[b])
                checks[f"{a}_vs_{b}"] = k
                if k:
                    raise SystemExit(f"FATAL: {a} and {b} share {k} prompts")
        k = len(written[a] & used)
        checks[f"{a}_vs_previously_used"] = k
        if k:
            raise SystemExit(f"FATAL: {a} reuses {k} already-consumed prompts")
    manifest["overlap_checks_exact"] = checks
    (out / "judge_v4_split_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("OK:", json.dumps(checks))


if __name__ == "__main__":
    main()
