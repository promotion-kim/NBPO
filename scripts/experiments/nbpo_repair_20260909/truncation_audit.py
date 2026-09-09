"""Audit the teacher's 32% truncation: what is the denominator, and what is lost.

Reported earlier as "about a third of what the teacher scores is truncated".
That needs a unit (prompts? responses? pairs?), the tokenizer and budget that
produced it, which side is cut, and how much of a response disappears.
"""
import json, glob, statistics
from pathlib import Path

ROOT = Path("/work/nbpo_repair_20260909")

pool = {"occurrences": 0, "response_truncated": 0.0, "prompt_truncated": 0.0}
for f in sorted(glob.glob(str(ROOT / "scores/shard*/manifest.json"))):
    m = json.loads(Path(f).read_text())
    t = m["truncation"]
    pool["occurrences"] += t["n_occurrences"]
    pool["response_truncated"] += t["response_fraction"] * t["n_occurrences"]
    pool["prompt_truncated"] += t["prompt_fraction"] * t["n_occurrences"]

rows = [json.loads(l) for l in (ROOT / "evaluations/saferlhf_base_v1/teacher_truncation.jsonl").read_text().splitlines()]
kept = [r["teacher_response_tokens_kept"] for r in rows if "teacher_response_tokens_kept" in r]
full = [r["teacher_response_tokens"] for r in rows if "teacher_response_tokens" in r]
cut = [(f, k) for f, k in zip(full, kept) if f > k]
lost = [f - k for f, k in cut]
frac_kept = [k / f for f, k in cut]

enc = json.loads((ROOT / "scores/shard0/manifest.json").read_text())
out = {
  "budget": {"max_encoder_tokens": 384,
             "declared_in": "protocols/resolved_protocol_frozen_v1.yaml teacher.max_encoder_tokens",
             "encoder_config_sha256": enc.get("encoder_config_sha256"),
             "note": "this is the teacher's own context budget; it was not raised, and raising a "
                     "max_length past the positions a model was trained on would be a model change, "
                     "not a configuration change"},
  "training_pool_side": {
     "unit": "candidate occurrences scored by the teacher (8 learner + 8 comparator per prompt)",
     "occurrences": pool["occurrences"],
     "response_truncated": round(pool["response_truncated"]),
     "response_truncated_fraction": round(pool["response_truncated"] / pool["occurrences"], 4),
     "prompt_truncated": round(pool["prompt_truncated"]),
     "prompt_truncated_fraction": round(pool["prompt_truncated"] / pool["occurrences"], 6)},
  "evaluation_side": {
     "unit": "scored responses on the SafeRLHF panel",
     "rows": len(rows),
     "response_truncated": sum(1 for r in rows if r.get("response_truncated")),
     "response_truncated_fraction": round(sum(1 for r in rows if r.get("response_truncated")) / len(rows), 4),
     "prompt_truncated": sum(1 for r in rows if r.get("prompt_truncated"))},
  "how_much_is_lost_when_cut": {
     "n_truncated_rows": len(cut),
     "tokens_lost_median": statistics.median(lost) if lost else None,
     "tokens_lost_mean": round(statistics.fmean(lost), 1) if lost else None,
     "tokens_lost_max": max(lost) if lost else None,
     "fraction_of_response_kept_median": round(statistics.median(frac_kept), 4) if frac_kept else None,
     "fraction_of_response_kept_min": round(min(frac_kept), 4) if frac_kept else None,
     "side": "the tail is dropped; the kept prefix starts at the beginning of the response"},
  "reading": (
     "The 32% figure is the training-pool number and its unit is candidate occurrences, not "
     "prompts and not pairs. Prompt truncation is essentially nil, so what is lost is always the "
     "END of a response. For the third that is cut, the median kept fraction says how much of the "
     "answer the teacher actually saw."),
  "what_this_does_not_establish": (
     "It does not say the teacher's judgement is wrong on those responses, only that it was made "
     "on a prefix. Whether the missing tail would change the preference is what the stratified "
     "full-context comparison is for, and that has not been run."),
}
print(json.dumps(out, indent=2))
(ROOT / "analysis_claude/teacher_truncation_audit.json").write_text(json.dumps(out, indent=2) + "\n")
