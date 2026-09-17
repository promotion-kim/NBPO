#!/usr/bin/env python3
"""Build the prompt-level UltraFeedback v2 split and its decontamination report.

Every earlier UltraFeedback experiment in this repository split at the *pair*
level, so the same user prompt could appear in training and in evaluation with
different responses attached.  [[nbpo-uf-eval-contamination]] records what that
cost.  This builder produces one deterministic split whose unit is the prompt,
groups near-identical prompts before splitting so a paraphrase cannot straddle
the boundary, and removes anything that overlaps a Table-1 benchmark.

Only the user prompt is taken.  UltraFeedback's own ``chosen``/``rejected``
responses and its scores are never read: the response pool is generated from
the frozen reference model and judged by the training judge, so importing
UltraFeedback's labels would mix a second, unmeasured preference source into
the objectives.

Pipeline
--------
1. **normalize** -- NFKC, collapse whitespace, strip.  The normalized form is
   what every later comparison and every overlap assertion uses.
2. **exact dedup** -- one survivor per distinct normalized prompt.
3. **near-duplicate grouping** -- character 13-gram MinHash (128 permutations,
   banded LSH) with a Jaccard threshold of 0.8, verified by an exact Jaccard
   recomputation on every candidate pair so the LSH only ever *proposes*.
   Connected components become groups, and the split is taken over groups.
4. **decontamination** -- the same matcher against the union of Arena-Hard v2,
   AlpacaEval 2, MT-Bench, IFEval and TruthfulQA prompts: exact normalized
   matches and approximate matches at Jaccard >= 0.8 are both dropped, and both
   counts are reported.
5. **split** -- groups shuffled by ``--seed`` (20260907) and assigned to
   train / validation / test by prompt budget, largest-group-first so a single
   huge group cannot overshoot a small split.

Nothing here silently falls back: a benchmark that cannot be fetched aborts the
build rather than producing a report that claims a decontamination which never
ran.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import unicodedata
from pathlib import Path

# --- normalization ----------------------------------------------------------

_WS = re.compile(r"\s+")


def normalize_prompt(text: str) -> str:
    """NFKC, whitespace-collapsed, stripped.  The canonical comparison form."""
    return _WS.sub(" ", unicodedata.normalize("NFKC", str(text))).strip()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --- character 13-gram MinHash ----------------------------------------------

NGRAM = 13
NUM_PERM = 128
BANDS = 32          # 32 bands x 4 rows: ~0.8 threshold
_MERSENNE = (1 << 61) - 1


def shingles(text: str, n: int = NGRAM) -> set:
    """Character n-grams of the *normalized* text, hashed to 64-bit ints."""
    if len(text) < n:
        return {hash(text) & 0xFFFFFFFFFFFFFFFF} if text else set()
    return {hash(text[i:i + n]) & 0xFFFFFFFFFFFFFFFF for i in range(len(text) - n + 1)}


def _permutations(num_perm: int, seed: int = 0):
    rng = random.Random(seed)
    return [(rng.randrange(1, _MERSENNE), rng.randrange(0, _MERSENNE))
            for _ in range(num_perm)]


PERMS = _permutations(NUM_PERM)


def minhash(sh: set) -> tuple:
    if not sh:
        return tuple([_MERSENNE] * NUM_PERM)
    return tuple(min(((a * s + b) % _MERSENNE) for s in sh) for a, b in PERMS)


def exact_jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def lsh_index(signatures):
    """Band the signatures; returns band -> bucket -> [ids]."""
    rows = NUM_PERM // BANDS
    buckets = [{} for _ in range(BANDS)]
    for idx, sig in enumerate(signatures):
        for b in range(BANDS):
            key = sig[b * rows:(b + 1) * rows]
            buckets[b].setdefault(key, []).append(idx)
    return buckets


def candidate_pairs(buckets):
    seen = set()
    for band in buckets:
        for ids in band.values():
            if len(ids) < 2 or len(ids) > 500:      # a degenerate bucket is not signal
                continue
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    pair = (ids[i], ids[j])
                    if pair not in seen:
                        seen.add(pair)
                        yield pair


class Union:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


# --- sources ----------------------------------------------------------------

# Every Table-1 benchmark whose prompts must not appear in training.
#
# Arena-Hard v2.0 is NOT on the Hub -- the only authoritative copy is the
# question file in the official lmarena/arena-hard-auto repository -- so it is
# fetched from that URL and pinned by the sha256 of the exact bytes retrieved.
# Guessing a look-alike Hub mirror would decontaminate against the wrong prompts.
BENCHMARKS = {
    "arena_hard_v2": {
        "kind": "url_jsonl",
        "url": "https://raw.githubusercontent.com/lmarena/arena-hard-auto/main/"
               "data/arena-hard-v2.0/question.jsonl",
        "extract": lambda r: r.get("prompt"),
        "note": "official arena-hard-v2.0 question set (751 prompts)",
    },
    # The Hub entry for AlpacaEval ships a loading SCRIPT, so `load_dataset`
    # demands trust_remote_code. The evaluation set is a plain JSON file in the
    # same repository, so it is fetched directly and pinned by content hash --
    # decontamination does not need to execute anybody's code.
    "alpaca_eval_2": {
        "kind": "url_json",
        "url": "https://huggingface.co/datasets/tatsu-lab/alpaca_eval/resolve/main/"
               "alpaca_eval.json",
        "extract": lambda r: r.get("instruction"),
        "note": "AlpacaEval 2 evaluation set (805 instructions), fetched as data "
                "rather than through the trust_remote_code loader",
    },
    "mt_bench": {
        "kind": "hf", "repo": "HuggingFaceH4/mt_bench_prompts", "config": None,
        "split": "train", "extract": lambda r: r.get("prompt"),
    },
    "ifeval": {
        "kind": "hf", "repo": "google/IFEval", "config": None, "split": "train",
        "extract": lambda r: r.get("prompt"),
    },
    "truthfulqa": {
        "kind": "hf", "repo": "truthfulqa/truthful_qa", "config": "generation",
        "split": "validation", "extract": lambda r: r.get("question"),
    },
}


def _flatten(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        out = []
        for v in value:
            out.extend(_flatten(v.get("content") if isinstance(v, dict) else v))
        return out
    if isinstance(value, dict):
        return _flatten(value.get("content") or value.get("prompt"))
    return [str(value)]


def load_benchmark_prompts(name, spec, cache_dir):
    """Fetch one benchmark's prompts and return them with a provenance record.

    Both source kinds pin what was actually read: a Hub dataset by its
    fingerprint, a URL by the sha256 of the exact bytes retrieved.
    """
    extract = spec["extract"]
    if spec["kind"] == "url_json":
        import urllib.request
        raw = urllib.request.urlopen(spec["url"], timeout=120).read()
        rows = json.loads(raw)
        meta = {"kind": "url_json", "url": spec["url"], "rows": len(rows),
                "content_sha256": hashlib.sha256(raw).hexdigest(),
                "note": spec.get("note")}
    elif spec["kind"] == "url_jsonl":
        import urllib.request
        raw = urllib.request.urlopen(spec["url"], timeout=120).read()
        # Two Arena-Hard prompts embed Unicode line separators (U+2028 and
        # friends) inside their JSON strings, which str.splitlines() cuts on and
        # json.loads then rejects. Split on "\n" only, and buffer continuation
        # lines so a record that legitimately spans several of them is
        # reassembled instead of dropped.
        rows, buf = [], ""
        for chunk in raw.decode("utf-8").split("\n"):
            buf = chunk if not buf else buf + "\n" + chunk
            if not buf.strip():
                buf = ""
                continue
            try:
                rows.append(json.loads(buf))
            except json.JSONDecodeError:
                continue
            buf = ""
        if buf.strip():
            raise ValueError(f"{name}: trailing bytes that never parsed as JSON")
        meta = {"kind": "url_jsonl", "url": spec["url"], "rows": len(rows),
                "content_sha256": hashlib.sha256(raw).hexdigest(),
                "note": spec.get("note")}
    else:
        from datasets import load_dataset
        ds = load_dataset(spec["repo"], spec["config"], split=spec["split"],
                          cache_dir=cache_dir)
        rows = list(ds)
        meta = {"kind": "hf", "repo": spec["repo"], "config": spec["config"],
                "split": spec["split"], "rows": len(ds),
                "fingerprint": getattr(ds, "_fingerprint", None)}
    prompts = []
    for row in rows:
        for text in _flatten(extract(row)):
            n = normalize_prompt(text)
            if n:
                prompts.append(n)
    if not prompts:
        raise ValueError(f"benchmark {name} yielded no prompts -- the extractor is wrong "
                         f"for this schema; refusing to report an empty decontamination")
    meta["prompts"] = len(prompts)
    return prompts, meta


def load_ultrafeedback(repo, split, cache_dir):
    from datasets import load_dataset
    ds = load_dataset(repo, split=split, cache_dir=cache_dir)
    rows = []
    for i, r in enumerate(ds):
        text = r.get("prompt")
        if text is None:                       # never fall back to chosen/rejected
            raise KeyError(f"{repo}:{split} row {i} has no 'prompt' field")
        rows.append({"source_index": i,
                     "source_id": r.get("prompt_id") or r.get("id"),
                     "raw_prompt": str(text)})
    return rows, {"repo": repo, "split": split, "rows": len(ds),
                  "fingerprint": getattr(ds, "_fingerprint", None)}


# --- main -------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--dataset", default="HuggingFaceH4/ultrafeedback_binarized")
    ap.add_argument("--dataset-split", default="train_prefs")
    ap.add_argument("--seed", type=int, default=20260907)
    ap.add_argument("--n-train", type=int, default=7000)
    ap.add_argument("--n-validation", type=int, default=500)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--jaccard", type=float, default=0.8)
    ap.add_argument("--cache-dir", default=None)
    args = ap.parse_args()

    out = args.out_dir
    (out / "splits").mkdir(parents=True, exist_ok=True)
    report = {"seed": args.seed, "jaccard_threshold": args.jaccard,
              "ngram": NGRAM, "num_perm": NUM_PERM, "bands": BANDS}

    print("[1/6] loading UltraFeedback prompts", flush=True)
    rows, ds_meta = load_ultrafeedback(args.dataset, args.dataset_split, args.cache_dir)
    report["dataset"] = ds_meta
    print(f"      {len(rows)} rows", flush=True)

    print("[2/6] normalizing and exact-deduplicating", flush=True)
    seen, unique, dup_exact = {}, [], 0
    for r in rows:
        n = normalize_prompt(r["raw_prompt"])
        if not n:
            continue
        if n in seen:
            dup_exact += 1
            continue
        seen[n] = True
        r["prompt"] = n
        r["prompt_sha256"] = sha256_text(n)
        unique.append(r)
    report["removed"] = {"empty_after_normalization": len(rows) - len(unique) - dup_exact,
                         "exact_duplicate_prompt": dup_exact}
    print(f"      {len(unique)} unique ({dup_exact} exact duplicates dropped)", flush=True)

    print("[3/6] loading benchmark prompts for decontamination", flush=True)
    bench_meta, bench_norm = {}, set()
    bench_shingles = []
    for name, spec in BENCHMARKS.items():
        try:
            prompts, meta = load_benchmark_prompts(name, spec, args.cache_dir)
        except Exception as exc:                       # no silent fallback
            print(f"FATAL: benchmark {name} could not be loaded: "
                  f"{type(exc).__name__}: {exc}", file=sys.stderr)
            print("Refusing to write a split whose decontamination report would "
                  "claim a check that never ran.", file=sys.stderr)
            raise SystemExit(2)
        bench_meta[name] = meta
        for p in prompts:
            bench_norm.add(p)
            bench_shingles.append(shingles(p))
        print(f"      {name}: {meta['prompts']} prompts", flush=True)
    report["benchmarks"] = bench_meta
    report["benchmark_prompt_count"] = len(bench_norm)

    print("[4/6] decontaminating", flush=True)
    uf_shingles = [shingles(r["prompt"]) for r in unique]
    exact_hits, approx_hits, kept = [], [], []
    all_sigs = [minhash(s) for s in uf_shingles] + [minhash(s) for s in bench_shingles]
    n_uf = len(unique)
    buckets = lsh_index(all_sigs)
    near_bench = {}
    for i, j in candidate_pairs(buckets):
        a, b = (i, j) if i < j else (j, i)
        if a < n_uf <= b:
            jac = exact_jaccard(uf_shingles[a], bench_shingles[b - n_uf])
            if jac >= args.jaccard:
                near_bench[a] = max(near_bench.get(a, 0.0), jac)
    for idx, r in enumerate(unique):
        if r["prompt"] in bench_norm:
            exact_hits.append({"prompt_sha256": r["prompt_sha256"], "reason": "exact_benchmark"})
        elif idx in near_bench:
            approx_hits.append({"prompt_sha256": r["prompt_sha256"],
                                "jaccard": round(near_bench[idx], 4),
                                "reason": "approx_benchmark"})
        else:
            kept.append((idx, r))
    report["decontamination"] = {
        "exact_matches_removed": len(exact_hits),
        "approximate_matches_removed": len(approx_hits),
        "approximate_examples": approx_hits[:50],
        "remaining": len(kept),
    }
    print(f"      removed {len(exact_hits)} exact + {len(approx_hits)} approximate; "
          f"{len(kept)} remain", flush=True)

    print("[5/6] grouping near-identical prompts", flush=True)
    kept_idx = [i for i, _ in kept]
    pos = {orig: p for p, orig in enumerate(kept_idx)}
    sigs = [all_sigs[i] for i in kept_idx]
    uf_only_buckets = lsh_index(sigs)
    uf = Union(len(kept))
    grouped_pairs = 0
    for i, j in candidate_pairs(uf_only_buckets):
        if exact_jaccard(uf_shingles[kept_idx[i]], uf_shingles[kept_idx[j]]) >= args.jaccard:
            uf.union(i, j)
            grouped_pairs += 1
    groups = {}
    for p in range(len(kept)):
        groups.setdefault(uf.find(p), []).append(p)
    group_list = list(groups.values())
    report["grouping"] = {
        "groups": len(group_list),
        "prompts": len(kept),
        "near_duplicate_pairs_merged": grouped_pairs,
        "largest_group": max(len(g) for g in group_list) if group_list else 0,
        "multi_prompt_groups": sum(1 for g in group_list if len(g) > 1),
    }
    print(f"      {len(group_list)} groups over {len(kept)} prompts "
          f"({grouped_pairs} near-duplicate pairs merged)", flush=True)

    print("[6/6] splitting at group level", flush=True)
    rng = random.Random(args.seed)
    rng.shuffle(group_list)
    # Largest groups first so one big group cannot overshoot a small split.
    group_list.sort(key=len, reverse=True)
    budgets = {"train": args.n_train, "validation": args.n_validation, "test": args.n_test}
    assigned = {k: [] for k in budgets}
    for g in group_list:
        room = {k: budgets[k] - len(assigned[k]) for k in budgets}
        fits = [k for k in ("test", "validation", "train") if room[k] >= len(g)]
        if not fits:
            continue
        # Fill the scarcest split first; ties broken deterministically by name.
        target = min(fits, key=lambda k: (budgets[k], k))
        assigned[target].extend(g)
    for name, want in budgets.items():
        got = len(assigned[name])
        if got != want:
            raise SystemExit(
                f"FATAL: split '{name}' got {got} prompts, wanted {want}. "
                "Refusing to write a short split silently (invariant: no smaller "
                "sample count without saying so).")

    manifest = {"seed": args.seed, "budgets": budgets, "files": {}, **report}
    for name in budgets:
        path = out / "splits" / f"ultrafeedback_{name}.jsonl"
        with path.open("w") as fh:
            for p in sorted(assigned[name]):
                r = kept[p][1]
                fh.write(json.dumps({
                    "prompt_id": f"uf2-{r['prompt_sha256'][:16]}",
                    "prompt": r["prompt"],
                    "prompt_sha256": r["prompt_sha256"],
                    "source_index": r["source_index"],
                    "source_id": r["source_id"],
                    "split": name,
                }, ensure_ascii=False) + "\n")
        manifest["files"][name] = {"path": str(path), "prompts": len(assigned[name]),
                                   "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        print(f"      {name}: {len(assigned[name])} -> {path}", flush=True)

    # --- hard assertions, run on what was actually written -----------------
    written = {}
    for name in budgets:
        written[name] = {json.loads(l)["prompt"]
                         for l in (out / "splits" / f"ultrafeedback_{name}.jsonl").read_text()
                         .splitlines()}
    checks = {}
    for a in budgets:
        for b in budgets:
            if a < b:
                inter = written[a] & written[b]
                checks[f"{a}_vs_{b}"] = len(inter)
                if inter:
                    raise SystemExit(f"FATAL: {a} and {b} share {len(inter)} prompts")
        inter_b = written[a] & bench_norm
        checks[f"{a}_vs_benchmarks"] = len(inter_b)
        if inter_b:
            raise SystemExit(f"FATAL: {a} shares {len(inter_b)} prompts with benchmarks")
    manifest["overlap_checks_exact"] = checks
    manifest["overlap_assertion"] = "all zero under exact normalized matching"

    (out / "splits" / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "splits" / "decontamination_report.json").write_text(
        json.dumps({"dataset": ds_meta, "benchmarks": bench_meta,
                    **report["decontamination"],
                    "grouping": report["grouping"],
                    "removed": report["removed"],
                    "jaccard_threshold": args.jaccard,
                    "ngram": NGRAM, "num_perm": NUM_PERM, "bands": BANDS}, indent=2))
    print("\nOK. overlap checks:", json.dumps(checks))


if __name__ == "__main__":
    main()
