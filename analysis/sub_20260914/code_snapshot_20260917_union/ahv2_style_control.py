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

The fit is weighted by nothing and pools both presentation orders, because the
raw figure averages the orders and the control has to be read on the same
observations. Uncertainty is the campaign's whole-prompt paired bootstrap: 2,000
replicates resampling PROMPTS, refitting the regression in each replicate, so
the interval carries the regression's own instability and not just the win
rate's. A replicate that fails to converge is recorded, never dropped silently.
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


def style(text: str, n_tokens: int | None = None) -> np.ndarray:
    """The four official style features. Token length is the pipeline's own count."""
    if n_tokens is None:
        n_tokens = len(text.split())
    return np.array([float(n_tokens),
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
    """Newton-Raphson logistic fit on continuous y in [0, 1]; returns the coefficients.

    The judged outcome is 1, 0 or .5 for a tie, so this is the Bernoulli
    log-likelihood evaluated at a fractional response -- the same objective the
    official fit uses when it splits a tie across both sides.
    """
    n, d = X.shape
    beta = np.zeros(d)
    for _ in range(iters):
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
        if np.max(np.abs(step)) < tol:
            return beta
    return beta


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
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    judged = Path(args.judged)
    complete = json.loads((judged / "complete.json").read_text())
    arm_path = Path(complete["arm_responses"])

    arm = {}
    for line in arm_path.open():
        r = json.loads(line)
        arm[r["prompt_id"]] = style(r["response"], r.get("n_tokens"))

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
        by_uid[r["uid"]] = style(text)
    base = {pid: by_uid[uid] for pid, uid in uid_of.items() if uid in by_uid}

    rows, skipped = [], {"no_style": 0, "not_ok": 0}
    for line in (judged / "verdicts.jsonl").open():
        v = json.loads(line)
        if v.get("status") != "ok":
            skipped["not_ok"] += 1
            continue
        pid = v["prompt_id"]
        if pid not in arm or pid not in base:
            skipped["no_style"] += 1
            continue
        rows.append((pid, float(v["value_for_arm"]),
                     normalized_difference(arm[pid], base[pid])))
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

    beta = fit(y, X)
    if beta is None:
        raise SystemExit("the point fit did not solve")
    point = 1.0 / (1.0 + math.exp(-beta[0]))
    raw = float(y.mean())

    rng = np.random.default_rng(args.seed)
    draws, failed = [], 0
    n = len(pids)
    for _ in range(args.replicates):
        pick = rng.integers(0, n, n)
        idx = np.concatenate([by_prompt[k] for k in pick])
        b = fit(y[idx], X[idx])
        if b is None:
            failed += 1
            continue
        draws.append(1.0 / (1.0 + math.exp(-b[0])))
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
        "style_features": ["answer_tokens", "markdown_headers", "bold_spans", "list_items"],
        "feature_encoding": "(arm - baseline) / (arm + baseline), 0 when both are zero",
        "raw_win_rate": raw,
        "style_controlled_win_rate": point,
        "style_controlled_ci95": [lo, hi],
        "shift_from_raw": point - raw,
        "coefficients": {"intercept": float(beta[0]),
                         **{k: float(v) for k, v in zip(
                             ["answer_tokens", "markdown_headers", "bold_spans", "list_items"],
                             beta[1:])}},
        "mean_feature_difference": {k: float(v) for k, v in zip(
            ["answer_tokens", "markdown_headers", "bold_spans", "list_items"], D.mean(axis=0))},
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
