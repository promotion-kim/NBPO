"""Check the MT-Bench grader's parsing and template selection before it runs.

The judge job is about two hours out and costs a GPU slot; the parts that do not
need a GPU are the parts that have gone wrong before (a regex that silently
matches the wrong thing, a template that quietly drops the reference answer).
"""
import sys
sys.path.insert(0, "/work/uf4_20260910/code")
import json
from score_mtbench import (parse_score, build_prompt, NEED_REF_CATS,
                           SINGLE_V1, SINGLE_MATH_V1, SINGLE_V1_MULTI, SINGLE_MATH_V1_MULTI)

fails = []


def check(name, got, want):
    if got != want:
        fails.append("%s: got %r want %r" % (name, got, want))


# --- the official grammar, and the official backup pattern -------------------
check("plain", parse_score("Good answer.\n\nRating: [[7]]"), (7.0, "ok"))
check("decimal", parse_score("Rating: [[8.5]]"), (8.5, "ok"))
check("ten", parse_score("Rating: [[10]]"), (10.0, "ok"))
check("one", parse_score("Rating: [[1]]"), (1.0, "ok"))
check("backup", parse_score("The response is fine. Rating: [6]"),
      (6.0, "ok_backup_pattern"))
check("none", parse_score("I cannot rate this."), (None, "no_rating"))
# A judge that restates the example and then rates is the realistic multi-match
# case; taking the first or last silently would score the example, so both are
# refused and the judgment stays missing.
check("two ratings", parse_score('format "[[rating]]" e.g. [[5]]. Rating: [[9]]'),
      (None, "multiple_ratings"))
check("truncated mid-reasoning", parse_score("The assistant begins by explaining that"),
      (None, "no_rating"))
# The double-bracket pattern must win over the single-bracket backup when both
# appear, or a citation marker like [3] would override the real rating.
check("prefers double", parse_score("See [3]. Rating: [[4]]"), (4.0, "ok"))

# --- template selection ------------------------------------------------------
rows = [json.loads(l) for l in
        open("/work/nbpo_repair_20260909/data/mt_bench_question.jsonl")]
by_id = {r["question_id"]: r for r in rows}
answers = {1: "first answer", 2: "second answer"}

math_q = next(r for r in rows if r["category"] == "math")
names = [n for n, _ in build_prompt(math_q, answers)]
check("math templates", names, ["single-math-v1", "single-math-v1-multi-turn"])
texts = [t for _, t in build_prompt(math_q, answers)]
check("math t1 carries ref 1", math_q["reference"][0] in texts[0], True)
check("math t2 carries ref 2", math_q["reference"][1] in texts[1], True)
check("math t2 carries both answers",
      all(a in texts[1] for a in answers.values()), True)

write_q = next(r for r in rows if r["category"] == "writing")
check("writing templates", [n for n, _ in build_prompt(write_q, answers)],
      ["single-v1", "single-v1-multi-turn"])

# A reference outside the declared categories must be ignored, exactly as
# MT-Bench's own judge ignores it.
extra = next(r for r in rows
             if "reference" in r and r["category"] not in NEED_REF_CATS)
names = [n for n, _ in build_prompt(extra, answers)]
check("non-declared category ignores its reference (%s q%d)"
      % (extra["category"], extra["question_id"]), names,
      ["single-v1", "single-v1-multi-turn"])
texts = [t for _, t in build_prompt(extra, answers)]
check("and does not leak the reference text",
      any(extra["reference"][0][:40] in t for t in texts), False)

# The one declared-category question with no reference falls back cleanly.
q123 = by_id[123]
check("q123 has no reference", "reference" in q123, False)
check("q123 category is declared", q123["category"] in NEED_REF_CATS, True)
check("q123 falls back to plain", [n for n, _ in build_prompt(q123, answers)],
      ["single-v1", "single-v1-multi-turn"])

# --- the four templates are the official wordings, not paraphrases -----------
check("v1 asks for a short explanation first",
      "Begin your evaluation by providing a short explanation" in SINGLE_V1, True)
check("math-v1 compares against the reference",
      "comparing the assistant's answer with the reference answer" in SINGLE_MATH_V1, True)
check("multi-turn focuses on the second answer",
      "focus on the assistant's answer to the second user question" in SINGLE_V1_MULTI, True)
check("math multi-turn focuses on the second question",
      "focus on the assistant's answer to the second question" in SINGLE_MATH_V1_MULTI, True)
for name, tpl in (("v1", SINGLE_V1), ("math-v1", SINGLE_MATH_V1),
                  ("v1-multi", SINGLE_V1_MULTI), ("math-v1-multi", SINGLE_MATH_V1_MULTI)):
    check("%s states the 1-10 scale and the grammar" % name,
          "scale of 1 to 10" in tpl and '"[[rating]]"' in tpl, True)

# --- coverage ----------------------------------------------------------------
declared = [r for r in rows if r["category"] in NEED_REF_CATS]
with_ref = [r for r in declared if "reference" in r]
check("29 of the 30 declared-category questions carry a reference",
      (len(declared), len(with_ref)), (30, 29))

if fails:
    print("FAIL")
    for f in fails:
        print("  " + f)
    raise SystemExit(1)
print("all %d checks passed" % 24)
