"""Turn the official Arena-Hard and AlpacaEval sets into campaign panels.

Both files are the projects own static releases: Arena-Hard v0.1 question.jsonl
and AlpacaEvals 805 instructions taken from the released gpt4_1106_preview
baseline file. Nothing is generated or judged here, and no paid API is touched.
The prompt_id is a hash of the normalized instruction so an arms responses can
be joined back to the right question without trusting file order.
"""
import hashlib, json, re, unicodedata
from pathlib import Path

E = Path("/work/sub_20260914/prosper/evalsets")
P = Path("/work/sub_20260914/panel")
NS = "sub_20260914-eval-panel:"


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


rows, report = [], {}
seen = set()
for line in (E / "ah_question.jsonl").open():
    if not line.strip():
        continue
    d = json.loads(line)
    t = d["prompt"]
    k = pid(t)
    if k in seen:
        continue
    seen.add(k)
    rows.append({"prompt_id": k, "instruction": t, "source": "arena-hard-v0.1",
                 "uid": d["uid"], "cluster": d.get("cluster")})
(P / "eval_arenahard.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                                                for r in rows))
report["arena_hard"] = {"prompts": len(rows), "source_sha256": sha(E / "ah_question.jsonl")}

rows, seen = [], set()
for d in json.load((E / "ae_baseline_gpt4_1106.json").open()):
    t = d["instruction"]
    k = pid(t)
    if k in seen:
        continue
    seen.add(k)
    rows.append({"prompt_id": k, "instruction": t, "source": "alpaca-eval-2.0",
                 "ae_dataset": d.get("dataset")})
(P / "eval_alpacaeval.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                                                 for r in rows))
report["alpaca_eval"] = {"prompts": len(rows),
                         "source_sha256": sha(E / "ae_baseline_gpt4_1106.json")}
report["note"] = ("baselines stay the official static files: gpt-4-0314 answers for "
                  "Arena-Hard and gpt4_1106_preview outputs for AlpacaEval 2.0. The "
                  "judge is the local Qwen3-14B, so scores are internally comparable "
                  "across our arms but are NOT the official leaderboard metric.")
(Path("/work/sub_20260914/prosper") / "eval_panels.json").write_text(
    json.dumps(report, indent=2) + "\n")
print(json.dumps(report))
