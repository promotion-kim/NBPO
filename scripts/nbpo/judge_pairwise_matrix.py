#!/usr/bin/env python3
"""Judge the full pairwise preference matrix for the paper-exact NBPO pipeline.

For every prompt and objective this queries the judge for:

1. current-policy learner responses vs reference comparator responses
   (the tensor behind ``V_{k,beta}(pi)``), and
2. reference learner responses vs reference comparator responses
   (the tensor behind the disagreement point ``d_k = V_{k,beta}(mu)``,
   Eq. (10) -- ``d`` is measured, never assumed zero).

Every semantic comparison is queried in BOTH presentation orders; downstream
swap-averaging happens in ``build_preference_tensor.py``. The reference pool
sits on both sides of its own game, so only UNORDERED pairs ``i < j`` are
judged there and the tensor builder writes ``a`` and ``-a``: exact skew
symmetry by construction. Self-comparisons are the definitional tie of
Eq. (1) and are neither judged nor imputed.

Invalid verdicts are never counted as completed work: they are retried up to
``--max-judge-retries`` times, and if the matrix is still incomplete the script
fails loudly listing the missing cells. There is no 0.5 fallback anywhere in
this path.

Rubrics come from the versioned YAML files under
``training_configs/nbpo/objectives/``; requesting an objective whose exact
experimental prompt is not in this repository is a hard error.

Backends: ``vllm`` (real judging) and ``mock`` (deterministic, LLM-free; used
by tests and ``run_nbpo_stage.py --dry-run``). The mock is intentionally
order-dependent, so a broken swap-average cannot pass the pipeline tests, and
its invalid rate is controllable to exercise the retry logic.
"""
from __future__ import annotations

import argparse
import itertools
import re
from pathlib import Path

from scripts.nbpo.nbpo_common import (
    PRESENTATION_ORDERS,
    append_jsonl,
    load_objective_rubrics,
    load_response_files,
    read_jsonl,
    stable_hash_int,
)


def build_task_grid(policy, reference, objectives, include_reference_learner, max_prompts=0):
    """Enumerate every judging cell: one per (prompt, objective, learner, comparator, order)."""
    prompt_ids = sorted(set.intersection(*[set(m) for m in policy.values()],
                                         *[set(m) for m in reference.values()]))
    if max_prompts:
        prompt_ids = prompt_ids[:max_prompts]
    if not prompt_ids:
        raise ValueError("no prompt ids shared by all response files")
    tasks = []
    for pid in prompt_ids:
        prompt_text = str(next(iter(policy.values()))[pid]["prompt"])
        pools = [("policy", policy, reference)]
        if include_reference_learner:
            pools.append(("reference", reference, reference))
        for pool_name, learners, comparators in pools:
            if pool_name == "policy":
                seed_pairs = list(itertools.product(sorted(learners), sorted(comparators)))
            else:
                # One response set on both sides: judge each UNORDERED pair i < j
                # once (both presentation orders) and let the tensor builder set
                # A[i, j] = a, A[j, i] = -a, A[i, i] = 0 exactly (Eqs. (1)-(2)).
                seed_pairs = list(itertools.combinations(sorted(comparators), 2))
            for lseed, cseed in seed_pairs:
                lid = f"{'policy' if pool_name == 'policy' else 'ref'}:{lseed}"
                cid = f"ref:{cseed}"
                for obj in objectives:
                    for order in PRESENTATION_ORDERS:
                        tasks.append({
                            "prompt_id": str(pid),
                            "prompt": prompt_text,
                            "objective": obj,
                            "learner_pool": pool_name,
                            "learner_response_id": lid,
                            "comparator_response_id": cid,
                            "learner_seed": lseed,
                            "comparator_seed": cseed,
                            "presentation_order": order,
                            "learner_text": str(learners[lseed][pid]["generated_text"]),
                            "comparator_text": str(comparators[cseed][pid]["generated_text"]),
                        })
    return tasks


def task_key(t) -> str:
    return "|".join([t["prompt_id"], t["objective"], t["learner_pool"],
                     t["learner_response_id"], t["comparator_response_id"],
                     t["presentation_order"]])


def parse_verdict(raw: str):
    """Last [[A]]/[[B]]/[[TIE]] marker -> win probability of response A, else None."""
    m = re.findall(r"\[\[(A|B|TIE)\]\]", raw.upper())
    if not m:
        return None
    return {"A": 1.0, "B": 0.0, "TIE": 0.5}[m[-1]]


def to_policy_win(win_of_a, order):
    """Convert P(A better) to P(learner > comparator) given the presentation order."""
    if win_of_a is None:
        return None
    return win_of_a if order == "learner_first" else 1.0 - win_of_a


class MockJudge:
    """Deterministic LLM-free judge for tests and dry runs.

    Each response id gets a latent per-(prompt, objective) strength; the
    verdict thresholds the strength gap. A fixed +/- bias is added by
    presentation order, so results are deliberately order-dependent (a correct
    swap-average cancels it; a broken one is caught by tests). A cell is
    invalid on attempt ``a`` iff a hash of (cell, a) falls below
    ``invalid_rate`` -- deterministic, and retries can succeed.
    """

    ORDER_BIAS = 0.08
    TIE_BAND = 0.10

    def __init__(self, invalid_rate: float = 0.0):
        self.invalid_rate = float(invalid_rate)

    def _strength(self, t, rid):
        return (stable_hash_int("strength", t["prompt_id"], t["objective"], rid) % 10_000) / 10_000.0

    def judge(self, tasks, attempt: int):
        outs = []
        for t in tasks:
            if self.invalid_rate > 0.0:
                u = (stable_hash_int("invalid", task_key(t), str(attempt)) % 1_000_000) / 1_000_000.0
                if u < self.invalid_rate:
                    outs.append(None)
                    continue
            gap = self._strength(t, t["learner_response_id"]) - self._strength(
                t, t["comparator_response_id"]
            )
            p_learner = 0.5 + 0.5 * gap
            bias = self.ORDER_BIAS if t["presentation_order"] == "learner_first" else -self.ORDER_BIAS
            p_first = p_learner + bias  # judged score of the first-presented response
            if p_first > 0.5 + self.TIE_BAND:
                win_learner = 1.0
            elif p_first < 0.5 - self.TIE_BAND:
                win_learner = 0.0
            else:
                win_learner = 0.5
            # report as P(A better): A is the first-presented response
            win_of_a = win_learner if t["presentation_order"] == "learner_first" else 1.0 - win_learner
            outs.append(win_of_a)
        return outs


class VllmJudge:
    """Real judging backend (lazy vLLM import; greedy on attempt 0, seeded T=0.3 retries)."""

    def __init__(self, model_path, rubrics, max_model_len=8192, tensor_parallel_size=1,
                 batch_size=512):
        from vllm import LLM, SamplingParams  # lazy: never imported in tests/dry runs

        self._SamplingParams = SamplingParams
        self.llm = LLM(model=model_path, max_model_len=max_model_len,
                       tensor_parallel_size=tensor_parallel_size,
                       gpu_memory_utilization=0.92, enable_prefix_caching=True, seed=42)
        self.tok = self.llm.get_tokenizer()
        self.rubrics = rubrics
        self.batch_size = batch_size

    def _render(self, t):
        rub = self.rubrics[t["objective"]]
        a, b = ((t["learner_text"], t["comparator_text"])
                if t["presentation_order"] == "learner_first"
                else (t["comparator_text"], t["learner_text"]))
        user = rub["user_template"].format(prompt=t["prompt"][:4000], a=a[:6000], b=b[:6000])
        msgs = [{"role": "system", "content": rub["system"]},
                {"role": "user", "content": user}]
        try:  # Qwen3: suppress the reasoning trace for deterministic verdicts
            return self.tok.apply_chat_template(msgs, tokenize=False,
                                                add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def judge(self, tasks, attempt: int):
        if attempt == 0:
            params = self._SamplingParams(n=1, temperature=0.0, max_tokens=512, seed=42)
        else:  # greedy retries would repeat the invalid output verbatim
            params = self._SamplingParams(n=1, temperature=0.3, max_tokens=512, seed=42 + attempt)
        outs = []
        for i in range(0, len(tasks), self.batch_size):
            batch = tasks[i:i + self.batch_size]
            gen = self.llm.generate([self._render(t) for t in batch], params, use_tqdm=False)
            outs.extend(parse_verdict(o.outputs[0].text) for o in gen)
        return outs


def run_judging(tasks, backend, output: Path, judge_model: str, rubric_version,
                max_retries: int) -> int:
    """Judge all cells with retries; hard-fail if any cell is still invalid."""
    done = set()
    if output.exists():
        for r in read_jsonl(output):
            if r.get("valid"):
                done.add("|".join([r["prompt_id"], r["objective"], r.get("learner_pool", "policy"),
                                   r["learner_response_id"], r["comparator_response_id"],
                                   r["presentation_order"]]))
    pending = [t for t in tasks if task_key(t) not in done]
    for attempt in range(1 + max_retries):
        if not pending:
            break
        wins = backend.judge(pending, attempt)
        if len(wins) != len(pending):
            raise RuntimeError(
                f"judge backend returned {len(wins)} verdicts for {len(pending)} cells "
                f"on attempt {attempt}; refusing to pair them up (a short output would "
                "silently drop cells)"
            )
        rows, still = [], []
        for t, win_of_a in zip(pending, wins):
            pw = to_policy_win(win_of_a, t["presentation_order"])
            rows.append({
                "prompt_id": t["prompt_id"],
                "objective": t["objective"],
                "learner_pool": t["learner_pool"],
                "learner_response_id": t["learner_response_id"],
                "comparator_response_id": t["comparator_response_id"],
                "learner_seed": t["learner_seed"],
                "comparator_seed": t["comparator_seed"],
                "presentation_order": t["presentation_order"],
                "policy_win": pw,
                "valid": pw is not None,
                "judge_model": judge_model,
                "rubric_version": rubric_version,
                "attempt": attempt,
            })
            if pw is None:
                still.append(t)
        append_jsonl(output, rows)
        pending = still
    if pending:
        sample = [task_key(t) for t in pending[:10]]
        raise RuntimeError(
            f"judging matrix incomplete after {max_retries} retries: "
            f"{len(pending)} cells still invalid; first missing cells: {sample}"
        )
    return len(tasks)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy-files", nargs="+", required=True, help="seed=path.json (learner pool)")
    ap.add_argument("--reference-files", nargs="+", required=True,
                    help="seed=path.json (comparator pool; also the d-tensor learner pool)")
    ap.add_argument("--objectives-config", type=Path, required=True,
                    help="versioned rubric YAML under training_configs/nbpo/objectives/")
    ap.add_argument("--objectives", required=True, help="comma-separated objective names")
    ap.add_argument("--judge-model-path", required=True)
    ap.add_argument("--backend", choices=["vllm", "mock"], default="vllm")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-judge-retries", type=int, default=2)
    ap.add_argument("--max-prompts", type=int, default=0)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--mock-invalid-rate", type=float, default=0.0)
    ap.add_argument("--skip-reference-learner", action="store_true",
                    help="omit the reference-vs-reference rows (d cannot be computed then)")
    args = ap.parse_args()

    objectives = [o.strip() for o in args.objectives.split(",") if o.strip()]
    loaded = load_objective_rubrics(args.objectives_config, objectives)  # fails loudly
    policy = load_response_files(args.policy_files)
    reference = load_response_files(args.reference_files)
    tasks = build_task_grid(policy, reference, objectives,
                            include_reference_learner=not args.skip_reference_learner,
                            max_prompts=args.max_prompts)
    if args.backend == "mock":
        backend = MockJudge(invalid_rate=args.mock_invalid_rate)
    else:
        backend = VllmJudge(args.judge_model_path, loaded["rubrics"],
                            max_model_len=args.max_model_len,
                            tensor_parallel_size=args.tensor_parallel_size,
                            batch_size=args.batch_size)
    total = run_judging(tasks, backend, args.output, judge_model=args.judge_model_path,
                        rubric_version=loaded["version"], max_retries=args.max_judge_retries)
    print(f"[judge_pairwise_matrix] complete: {total} cells valid -> {args.output}")


if __name__ == "__main__":
    main()
