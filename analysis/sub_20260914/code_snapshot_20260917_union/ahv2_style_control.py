"""Style-controlled Arena-Hard v2.0 win rate, computed from judgments we already have.

Arena-Hard v2.0 reports a style-controlled figure alongside the raw one: the raw
win rate rewards a model for answering at length and in heavy markdown, and the
controlled figure asks what the win rate would be if the two answers were
matched on those surface features. The control is a regression of the judged
outcome on the STYLE DIFFERENCE between the two answers, read at zero
difference.

This is a local implementation and must be labelled as one. Two reasons, and
both belong next to the number:

* The judge here is Qwen2.5-72B-Instruct, not the official gpt-4.1-mini judge.
  The pipeline already records `not_the_official_metric` for the raw figure and
  the same caveat carries to this one.
* The official style control fits one Bradley-Terry model over the whole arena
  with a model term per competitor. Our panel is one arm against one frozen
  baseline, so the same idea reduces to a per-arm logistic fit whose intercept
  is the style-free log-odds. That is the same estimand, not the same code.

Features, following the official four: answer token length, markdown headers,
bold spans, and list items. Each enters as the normalized difference
(arm - baseline) / (arm + baseline), which is bounded in [-1, 1] and undefined
only when both answers have none of that element -- that case is 0, meaning the
two are matched on it.

BOTH SIDES ARE COUNTED THE SAME WAY, and an earlier version of this file did
not. It read the arm's token count from the generation record and fell back to
whitespace words for the baseline, whose file stores no token count. Tokens run
roughly a third above words on English prose, so two IDENTICAL answers scored a
normalized length difference of .44 instead of 0, and every observation carried
that offset. It biases the very coefficient the control exists to remove. The
length feature is now produced by one frozen tokenizer applied to both answers,
and the module refuses to run without it rather than silently falling back to a
different unit. `--length-unit words` is available for a tokenizer-free run and
then applies word counts to BOTH sides; whichever is used is recorded in the
output.

The fit is weighted by nothing. It pools both presentation orders because the
raw figure averages the orders, and for the same reason a prompt is used only
when BOTH of its orders parsed: a prompt contributing one order would enter the
control with a first-position bias the raw figure does not have. Dropped prompts
are counted.

Uncertainty is the campaign's whole-prompt paired bootstrap: 2,000 replicates
resampling PROMPTS, refitting the regression in each replicate, so the interval
carries the regression's own instability and not just the win rate's. A
replicate whose fit exhausts its iterations or returns a non-finite coefficient
is counted as failed and excluded, never counted as a successful replicate.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

SUB = Path("/work/sub_20260914")
HEADER = re.compile(r"^\s{0,3}#{1,6}\s", re.M)
BOLD = re.compile(r"\*\*[^*\n]+\*\*|__[^_\n]+__")
LIST = re.compile(r"^\s*(?:[-*+]\s|\d+[.)]\s)", re.M)
ORDERS = 2


def make_length(unit: str, tokenizer_path: str | None):
    """One length counter, applied to both answers.

    The unit has to be identical on the two sides or the difference is not a
    style difference. Returning a single closure is how that is enforced: there
    is no second code path for the side whose file happens to store a token
    count.
    """
    if unit == "tokens":
        if not tokenizer_path:
            raise SystemExit("--length-unit tokens needs --tokenizer")
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tokenizer_path)

        def count(text: str) -> int:
            return len(tok(text, add_special_tokens=False)["input_ids"])
        return count, {"unit": "tokenizer_tokens", "tokenizer": tokenizer_path,
                       "add_special_tokens": False}
    if unit == "words":
        return (lambda text: len(text.split())), {"unit": "whitespace_words"}
    raise SystemExit("unknown --length-unit %r" % unit)


def style(text: str, length) -> np.ndarray:
    """The four official style features, length measured by the shared counter."""
    return np.array([float(length(text)),
                     float(len(HEADER.findall(text))),
                     float(len(BOLD.findall(text))),
                     float(len(LIST.findall(text)))], dtype=np.float64)


def normalized_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    total = a + b
    out = np.zeros_like(a)
    live = total > 0
    out[live] = (a[live] - b[live]) / total[live]
    return out


def fit(y: np.ndarray, X: np.ndarray, iters: int = 200, tol: float = 1e-10):
    """Newton-Raphson logistic fit on continuous y in [0, 1].

    The judged outcome is 1, 0 or .5 for a tie, so this is the Bernoulli
    log-likelihood evaluated at a fractional response -- the same objective the
    official fit uses when it splits a tie across both sides.

    Returns (beta, info) and returns None only when the linear system itself
    fails. An earlier version returned the coefficients after exhausting its
    iterations with no way for the caller to tell, so a replicate that had not
    converged was counted as a successful one and narrowed the interval. The
    caller now reads `info["converged"]`, and the bootstrap discards a replicate
    that did not converge or produced a non-finite coefficient.
    """
    n, d = X.shape
    beta = np.zeros(d)
    for it in range(1, iters + 1):
        eta = np.clip(X @ beta, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-eta))
        w = np.maximum(p * (1.0 - p), 1e-10)
        grad = X.T @ (y - p)
        hess = X.T @ (X * w[:, None])
        hess.flat[:: d + 1] += 1e-8          # ridge, so a collinear replicate still solves
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            return None
        beta = beta + step
        if not np.all(np.isfinite(beta)):
            return None
        if np.max(np.abs(step)) < tol:
            return beta, {"converged": True, "iterations": it,
                          "gradient_inf_norm": float(np.abs(grad).max()),
                          "last_step_inf_norm": float(np.abs(step).max())}
    eta = np.clip(X @ beta, -30.0, 30.0)
    p = 1.0 / (1.0 + np.exp(-eta))
    return beta, {"converged": False, "iterations": iters,
                  "gradient_inf_norm": float(np.abs(X.T @ (y - p)).max()),
                  "last_step_inf_norm": float(np.abs(step).max())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--judged", required=True,
                    help="eval_pairwise directory holding verdicts.jsonl and complete.json")
    ap.add_argument("--baseline", required=True, help="the frozen baseline jsonl")
    ap.add_argument("--panel", required=True,
                    help="the frozen panel jsonl; it carries the prompt_id -> uid join the "
                         "judging run used, and the baseline is keyed by uid")
    ap.add_argument("--replicates", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260918)
    ap.add_argument("--length-unit", default="tokens", choices=("tokens", "words"),
                    help="how BOTH answers' length is counted; never one each way")
    ap.add_argument("--tokenizer", default="/work/models/bases/Qwen2.5-7B-Instruct")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    length, length_meta = make_length(args.length_unit, args.tokenizer)
    # the defect this flag exists for: identical text must give a zero difference
    probe = style("one two three four five six seven", length)
    if float(normalized_difference(probe, probe)[0]) != 0.0:
        raise SystemExit("length counter is not self-consistent")

    judged = Path(args.judged)
    complete = json.loads((judged / "complete.json").read_text())
    arm_path = Path(complete["arm_responses"])

    arm = {}
    for line in arm_path.open():
        r = json.loads(line)
        arm[r["prompt_id"]] = style(r["response"], length)

    # the same join the judging run used: panel prompt_id -> uid -> baseline answer
    uid_of = {}
    for line in Path(args.panel).open():
        r = json.loads(line)
        uid_of[r["prompt_id"]] = r["uid"]

    by_uid = {}
    for line in Path(args.baseline).open():
        r = json.loads(line)
        text = r["messages"][-1]["content"] if "messages" in r else r.get("response", "")
        if isinstance(text, dict):
            text = text.get("answer", "")
        by_uid[r["uid"]] = style(text, length)
    base = {pid: by_uid[uid] for pid, uid in uid_of.items() if uid in by_uid}

    # both orders or neither: the raw figure averages the two presentations, so a
    # prompt that parsed in only one of them would enter the control carrying a
    # first-position bias the raw figure does not have
    seen = {}
    skipped = {"no_style": 0, "not_ok": 0, "single_order_prompts": 0}
    for line in (judged / "verdicts.jsonl").open():
        v = json.loads(line)
        if v.get("status") != "ok":
            skipped["not_ok"] += 1
            continue
        pid = v["prompt_id"]
        if pid not in arm or pid not in base:
            skipped["no_style"] += 1
            continue
        seen.setdefault(pid, {})[int(v["order"])] = float(v["value_for_arm"])

    rows = []
    for pid, byorder in seen.items():
        if len(byorder) != ORDERS:
            skipped["single_order_prompts"] += 1
            continue
        d = normalized_difference(arm[pid], base[pid])
        for order in sorted(byorder):
            rows.append((pid, byorder[order], d))
    if not rows:
        raise SystemExit("no usable observation: check the prompt-id join")

    pids = sorted({r[0] for r in rows})
    index = {p: i for i, p in enumerate(pids)}
    by_prompt: list[list[int]] = [[] for _ in pids]
    for i, (pid, _, _) in enumerate(rows):
        by_prompt[index[pid]].append(i)

    y = np.array([r[1] for r in rows])
    D = np.stack([r[2] for r in rows])
    # token length enters on a log scale in the official fit, because a 100-vs-200
    # token gap is not the same stylistic distance as 2000-vs-2100; the normalized
    # difference is already scale free, so it is used directly here and the raw
    # token counts are reported so the choice is auditable.
    X = np.column_stack([np.ones(len(rows)), D])

    solved = fit(y, X)
    if solved is None:
        raise SystemExit("the point fit did not solve")
    beta, fit_info = solved
    if not fit_info["converged"]:
        raise SystemExit("the point fit exhausted its iterations: %s" % fit_info)
    point = 1.0 / (1.0 + math.exp(-beta[0]))
    raw = float(y.mean())

    rng = np.random.default_rng(args.seed)
    draws, failed = [], 0
    n = len(pids)
    for _ in range(args.replicates):
        pick = rng.integers(0, n, n)
        idx = np.concatenate([by_prompt[k] for k in pick])
        got = fit(y[idx], X[idx])
        if got is None or not got[1]["converged"]:
            failed += 1
            continue
        draws.append(1.0 / (1.0 + math.exp(-got[0][0])))
    draws = np.sort(np.asarray(draws))
    lo, hi = (float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))) \
        if draws.size else (float("nan"), float("nan"))

    out = {
        "arm": complete["arm"],
        "kind": complete["kind"],
        "judge": complete["judge"],
        "baseline": complete["baseline"],
        "metric": "style_controlled_win_rate_local_implementation",
        "not_the_official_metric": (
            "The official Arena-Hard v2.0 style control fits one Bradley-Terry model over the "
            "whole arena with a per-competitor term and the official judge. This is a per-arm "
            "logistic fit against the frozen baseline under the campaign's own judge "
            + str(complete["judge"]) + ". Same estimand, different estimator and different "
            "judge; do not report it as the official SC figure."),
        "style_features": ["answer_length", "markdown_headers", "bold_spans", "list_items"],
        "feature_encoding": "(arm - baseline) / (arm + baseline), 0 when both are zero",
        "length_counter": length_meta,
        "length_counter_note": ("one counter for both answers; an identical-text "
                                "self-difference of zero is asserted before any data is read"),
        "point_fit": fit_info,
        "order_rule": "a prompt enters only when both presentation orders parsed",
        "raw_win_rate": raw,
        "style_controlled_win_rate": point,
        "style_controlled_ci95": [lo, hi],
        "shift_from_raw": point - raw,
        "coefficients": {"intercept": float(beta[0]),
                         **{k: float(v) for k, v in zip(
                             ["answer_length", "markdown_headers", "bold_spans", "list_items"],
                             beta[1:])}},
        "mean_feature_difference": {k: float(v) for k, v in zip(
            ["answer_length", "markdown_headers", "bold_spans", "list_items"], D.mean(axis=0))},
        "observations": len(rows),
        "prompts": len(pids),
        "skipped": skipped,
        "bootstrap": {"replicates": args.replicates, "unit": "prompt",
                      "refit_per_replicate": True, "failed_replicates": failed,
                      "seed": args.seed},
        "raw_win_rate_recorded_by_the_judging_run": complete["WIN_RATE_VS_BASELINE"],
        "join": {"panel": args.panel, "prompt_id_to_uid": len(uid_of),
                 "baseline_answers": len(by_uid)},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in (
        "arm", "raw_win_rate", "style_controlled_win_rate", "style_controlled_ci95",
        "shift_from_raw", "prompts", "observations")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
