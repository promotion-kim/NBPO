"""Freeze the Arena-Hard v2.0 panel, beside the v0.1 one rather than over it.

Table 1's Arena-Hard column is v0.1: the 500-question `question.jsonl` judged
against the released gpt-4-0314 answers. v2.0 is a different benchmark, not a
refresh of the same one, and this records the differences instead of blurring
them:

  * 750 questions, not 500;
  * two categories -- 500 hard_prompt (coding and math) and 250
    creative_writing -- where v0.1 had one;
  * 26 languages, only 504 of the 750 in English, where v0.1 is English only;
  * the official baseline is gpt-4.1, not gpt-4-0314.

Everything else is kept identical to the v0.1 run so the two columns differ in
the benchmark and nothing else: the same prompt_id hashing, the same static
official files with no generation or judging here, and no paid API.

The baseline answers arrive with the assistant content nested as
`messages[1].content.answer`; the batch judge already handles that shape, so the
file is copied verbatim rather than rewritten.
"""
import hashlib, json, re, shutil, unicodedata
from pathlib import Path

SRC = Path("/work/sub_20260914/prosper/evalsets/ah_v2_src/data/arena-hard-v2.0")
E = Path("/work/sub_20260914/prosper/evalsets")
P = Path("/work/sub_20260914/panel")
NS = "sub_20260914-eval-panel:"          # the v0.1 panel's namespace, unchanged


def norm(t):
    t = unicodedata.normalize("NFKC", t).replace("​", "")
    return re.sub(r"\s+", " ", t).strip().lower()


def pid(t):
    return hashlib.sha256((NS + norm(t)).encode("utf-8")).hexdigest()[:16]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as s:
        for c in iter(lambda: s.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


# the baseline file, copied verbatim to the campaign's evalsets directory
baseline_src = SRC / "model_answer/gpt-4.1.jsonl"
baseline_dst = E / "ahv2_baseline_gpt41.jsonl"
if not baseline_dst.exists():
    shutil.copyfile(baseline_src, baseline_dst)

rows, seen, dup = [], set(), 0
import collections
cats, langs = collections.Counter(), collections.Counter()
for line in (SRC / "question.jsonl").open():
    if not line.strip():
        continue
    d = json.loads(line)
    t = d["prompt"]
    k = pid(t)
    if k in seen:
        dup += 1
        continue
    seen.add(k)
    cats[d.get("category")] += 1
    langs[d.get("language")] += 1
    rows.append({"prompt_id": k, "instruction": t, "source": "arena-hard-v2.0",
                 "uid": d["uid"], "category": d.get("category"),
                 "subcategory": d.get("subcategory"),
                 "language": d.get("language")})

# every question must have a baseline answer, or the join would silently shrink
base_uids = set()
for line in baseline_dst.open():
    if line.strip():
        base_uids.add(json.loads(line)["uid"])
missing = [r["uid"] for r in rows if r["uid"] not in base_uids]
if missing:
    raise SystemExit("%d questions have no gpt-4.1 baseline answer" % len(missing))

dest = P / "eval_arenahard_v2.jsonl"
dest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))

report = {
    "panel": str(dest), "panel_sha256": sha(dest), "prompts": len(rows),
    "duplicate_prompts_dropped": dup,
    "question_source": "arena-hard-v2.0 question.jsonl",
    "question_file": str(SRC / "question.jsonl"),
    "question_sha256": sha(SRC / "question.jsonl"),
    "baseline": "gpt-4.1",
    "baseline_file": str(baseline_dst),
    "baseline_sha256": sha(baseline_dst),
    "categories": dict(cats), "languages": dict(langs.most_common()),
    "english_fraction": round(langs["English"] / len(rows), 4),
    "differences_from_v0_1": {
        "prompts": "750 versus 500",
        "categories": "hard_prompt 500 (coding, math) plus creative_writing 250; v0.1 has one category",
        "languages": "26 languages, %d of %d in English; v0.1 is English only"
                     % (langs["English"], len(rows)),
        "baseline": "gpt-4.1 versus gpt-4-0314",
        "conclusion": ("v2.0 is a different benchmark, not a refresh of v0.1; the two "
                       "win rates are not two measurements of one quantity"),
    },
    "unchanged_from_the_v0_1_run": [
        "prompt_id hashing and its namespace",
        "official static question and baseline files only",
        "no generation or judging in this step",
        "no paid API",
    ],
    "not_the_official_metric": ("the official v2.0 pipeline judges with gpt-4.1 and "
                                "applies style control; this campaign judges locally "
                                "with Qwen2.5-72B-Instruct and reports the raw "
                                "order-averaged win rate, so these numbers are "
                                "comparable across our arms and are not leaderboard "
                                "numbers"),
    "source_sha256": sha(Path(__file__)),
}
out = Path("/work/sub_20260914/prosper/eval_panel_ahv2.json")
out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
print(json.dumps({k: report[k] for k in
                  ("prompts", "panel_sha256", "categories", "english_fraction",
                   "baseline", "differences_from_v0_1")}, indent=1, ensure_ascii=False))
