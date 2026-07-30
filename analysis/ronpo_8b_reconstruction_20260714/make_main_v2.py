#!/usr/bin/env python3
"""Produce a cleaned main_v2.tex from main.tex:
  1. strip every standalone TODO-EXPERIMENT / TODO-AUTHOR comment block,
  2. replace the messy author-action header with a clean two-line header,
  3. remove the one prose em-dash pair (Setup sentence),
  4. rescope the abstract and intro claims to match the evidence.
No result numbers or math are touched.
"""
import sys

SRC = "/home/sjkim/MNPO/ronpo_aaai/main.tex"
DST = "/home/sjkim/MNPO/ronpo_aaai/main_v2.tex"

text = open(SRC, encoding="utf-8").read()

# --- 3+4: prose replacements (must each hit exactly once) ---
REPLACEMENTS = [
    # abstract final sentence: honest scope
    ("At model scale, RONPO improves worst-objective reward and is preferred by an external judge over averaged and single-oracle baselines.",
     "At model scale, RONPO raises worst-objective reward under the training-aligned judges and preserves verifiable instruction following, and an independent judge ranks it first among the trained methods, though it reaches only parity with the strong base policy."),
    # intro contributions bullet: honest scope
    ("At model scale, RONPO improves average and worst-objective reward over averaged and single-oracle baselines and is preferred by an external judge.",
     "At model scale, RONPO raises worst-objective reward under the training-aligned judges and preserves instruction following, and an independent judge ranks it first among the trained methods while showing only parity with the base policy."),
    # em-dash removal in the Setup sentence
    (r"three heterogeneous reward objectives---Skywork, Athene, and ArmoRM (8B-scale reward models; Appendix~\ref{app:llm-details})---used through the plug-in oracle",
     r"three heterogeneous reward objectives (Skywork, Athene, and ArmoRM, all 8B-scale reward models; Appendix~\ref{app:llm-details}), applied through the plug-in oracle"),
]
for old, new in REPLACEMENTS:
    n = text.count(old)
    if n != 1:
        sys.exit(f"ERROR: expected 1 occurrence, found {n} for: {old[:60]!r}")
    text = text.replace(old, new)

# --- 1: strip standalone comment blocks that contain a TODO marker ---
lines = text.split("\n")
kept = []
i = 0
def is_comment(s):
    return s.lstrip().startswith("%")
while i < len(lines):
    if is_comment(lines[i]):
        j = i
        while j < len(lines) and is_comment(lines[j]):
            j += 1
        block = lines[i:j]
        if any(("TODO-EXPERIMENT" in l) or ("TODO-AUTHOR" in l) for l in block):
            i = j          # drop the whole comment block (incl. old header)
            continue
        kept.extend(block)
        i = j
        continue
    kept.append(lines[i])
    i += 1

# collapse runs of >1 blank line to a single blank line
collapsed = []
for l in kept:
    if l.strip() == "" and collapsed and collapsed[-1].strip() == "":
        continue
    collapsed.append(l)

# --- 2: clean header ---
header = ["%File: ronpo_aaai27.tex", "% AAAI-27 anonymous submission.", ""]
# the old header block was dropped above (it contained the TODO reference),
# so the file now starts at \documentclass; prepend the clean header.
out = "\n".join(header + collapsed)

open(DST, "w", encoding="utf-8").write(out)

# report
n_todo = out.count("TODO-EXPERIMENT") + out.count("TODO-AUTHOR")
n_emdash_prose = sum(
    1 for ln in out.split("\n")
    if "---" in ln and not ln.lstrip().startswith("%") and "&" not in ln
)
print(f"wrote {DST}")
print(f"  remaining TODO markers: {n_todo}")
print(f"  prose lines still containing '---' (non-table, non-comment): {n_emdash_prose}")
print(f"  lines: {len(out.splitlines())}")
