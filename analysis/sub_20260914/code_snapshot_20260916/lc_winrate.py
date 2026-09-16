"""Length-adjusted win rate: the win rate an arm would have at equal length.

Table 8 showed the trained-minus-base gap is monotone in response length, which
leaves the win rates themselves uninterpretable: an arm that answers at twice
the length of the released baseline is partly being rewarded for that. The
published length-controlled AlpacaEval metric handles this by regressing the
verdict on the length difference and reading the fit at zero difference. This is
the same idea in its simplest form, stated rather than hidden:

    y_p = a + b * tanh((len_arm_p - len_baseline_p) / sigma) + e_p

fitted by least squares over prompts, with sigma the standard deviation of the
raw length difference. y_p is the arm's score on prompt p, already averaged over
both presentation orders, so ties enter as 0.5. The intercept a is the win rate
at equal length and b is the length slope in win-rate units; both carry a
whole-prompt bootstrap. Baseline lengths are tokenized here with the policy
tokenizer, since the judged text is what carries the bias.

This is a linear adjustment on averaged scores, not the official GLM, and it is
still conditioning on a variable the training changed. It answers one question:
how much of the reported win rate survives setting the length difference to zero.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

RESP = Path("/work/sub_20260914/responses")
EVAL = Path("/work/sub_20260914/eval_pairwise")
PROS = Path("/work/sub_20260914/prosper/evalsets")
QWEN = "/work/models/bases/Qwen2.5-7B-Instruct"


def arm_scores(tag, orders=2):
    table, bad = {}, set()
    for line in (EVAL / tag / "verdicts.jsonl").open():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["status"] != "ok":
            bad.add(r["prompt_id"]); continue
        table.setdefault(r["prompt_id"], []).append(float(r["value_for_arm"]))
    return {p: float(np.mean(v)) for p, v in table.items()
            if len(v) == orders and p not in bad}


def arm_tokens(tag):
    out = {}
    for line in (RESP / tag / "responses.jsonl").open():
        if line.strip():
            e = json.loads(line)
            out[e["prompt_id"]] = int(e["n_tokens"])
    return out


def baseline_tokens(kind, panel_path, tok):
    """prompt_id -> token count of the released baseline answer."""
    by_text, by_uid = {}, {}
    for line in Path(panel_path).open():
        if line.strip():
            row = json.loads(line)
            by_text[row["instruction"].strip()] = row["prompt_id"]
            if row.get("uid"):
                by_uid[row["uid"]] = row["prompt_id"]
    panel = by_text
    out = {}
    if kind == "arenahard":
        # the baseline file keys answers by uid and stores the answer as the
        # assistant turn of a two-message list
        for line in (PROS / "ah_baseline_gpt4_0314.jsonl").open():
            if not line.strip():
                continue
            d = json.loads(line)
            pid = by_uid.get(d.get("uid"))
            if pid is None:
                continue
            answer = ""
            for m in d.get("messages", []):
                if not (isinstance(m, dict) and m.get("role") == "assistant"):
                    continue
                content = m.get("content")
                if isinstance(content, dict):
                    content = content.get("answer") or content.get("text") or ""
                if not isinstance(content, str):
                    raise ValueError("unexpected assistant content shape: %s"
                                     % type(content).__name__)
                answer = content
            if answer:
                out[pid] = len(tok(answer)["input_ids"])
    else:
        rows = json.loads((PROS / "ae_baseline_gpt4_1106.json").read_text())
        for d in rows:
            key = (d.get("instruction") or "").strip()
            text = d.get("output") or ""
            if key in panel and text:
                out[panel[key]] = len(tok(text)["input_ids"])
    return out


def fit(y, x, rng, reps):
    """Least squares y = a + b * x, plus a whole-prompt bootstrap."""
    def solve(idx):
        X = np.column_stack([np.ones(len(idx)), x[idx]])
        coef, *_ = np.linalg.lstsq(X, y[idx], rcond=None)
        return coef
    a, b = solve(np.arange(len(y)))
    draws = np.empty((reps, 2))
    for k in range(reps):
        draws[k] = solve(rng.integers(0, len(y), len(y)))
    lo_a, hi_a = np.percentile(draws[:, 0], [2.5, 97.5])
    lo_b, hi_b = np.percentile(draws[:, 1], [2.5, 97.5])
    return ({"estimate": float(a), "ci95": [float(lo_a), float(hi_a)]},
            {"estimate": float(b), "ci95": [float(lo_b), float(hi_b)]})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True, help="eval_pairwise tag")
    ap.add_argument("--responses", required=True, help="responses tag of the same run")
    ap.add_argument("--kind", required=True, choices=("arenahard", "alpacaeval"))
    ap.add_argument("--panel", required=True, help="panel file under panel/")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260916)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(QWEN, local_files_only=True)
    y_map = arm_scores(args.tag)
    a_tok = arm_tokens(args.responses)
    b_tok = baseline_tokens(args.kind, "/work/sub_20260914/panel/" + args.panel, tok)
    common = sorted(set(y_map) & set(a_tok) & set(b_tok))
    if len(common) < 100:
        raise SystemExit("only %d prompts matched arm, baseline and verdicts" % len(common))
    y = np.array([y_map[p] for p in common])
    delta = np.array([a_tok[p] - b_tok[p] for p in common], dtype=float)
    sigma = float(delta.std(ddof=1))
    x = np.tanh(delta / sigma)
    rng = np.random.default_rng(args.seed)
    intercept, slope = fit(y, x, rng, args.bootstrap)
    report = {
        "tag": args.tag, "kind": args.kind, "n_prompts": len(common),
        "raw_win_rate": float(y.mean()),
        "length_adjusted_win_rate": intercept,
        "length_slope_in_win_rate_units": slope,
        "length_difference_tokens": {"mean": float(delta.mean()),
                                     "median": float(np.median(delta)),
                                     "sd": sigma},
        "median_tokens": {"arm": float(np.median([a_tok[p] for p in common])),
                          "baseline": float(np.median([b_tok[p] for p in common]))},
        "model": "y = a + b * tanh((arm tokens - baseline tokens) / sd)",
        "bootstrap": {"replicates": args.bootstrap, "unit": "whole prompt"},
        "caveat": ("a linear adjustment on order-averaged scores, not the official "
                   "length-controlled GLM, and length is a post-treatment variable"),
    }
    out = Path("/work/sub_20260914/analysis") / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in
                      ("tag", "n_prompts", "raw_win_rate", "length_adjusted_win_rate",
                       "length_slope_in_win_rate_units", "median_tokens")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
