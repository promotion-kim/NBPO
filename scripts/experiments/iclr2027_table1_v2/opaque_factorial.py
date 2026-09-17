#!/usr/bin/env python3
"""One bounded factorial: is the judge's slot bias lexical, physical, or both?

P2-long's development failure is concentrated. Helpfulness and instruction
following carry +0.36 and +0.40 position bias with confident swap consistency
BELOW chance; honesty and truthfulness carry +0.04 and +0.02 and reach about
0.70. Something specific to two criteria interacts with the slot -- but the
ordinary rendering cannot say what, because being shown FIRST and being called
"A" are perfectly confounded in it.

This decomposes them, and it is the last prompted-judge diagnostic that will be
run. Three factors are crossed on the same semantic pairs:

``semantic``    which response is the learner ``y``;
``position``    which response is rendered in the first slot;
``identifier``  which slot carries the first identifier -- ``AB`` puts it on
                slot 1 (the canonical rendering), ``BA`` puts it on slot 2.

With ``AB`` and ``BA`` both present, every semantic response appears equally
often first and second AND equally often as A and as B, so

    position effect   = mean advantage of the slot-1 response, over both labellings
    identifier effect = mean advantage of the "A"-labelled response, over both slots

are exactly orthogonal rather than approximately decorrelated.

The second condition replaces A/B with neutral identifiers (``K7`` / ``Q2``).
If the whole effect is lexical, opaque identifiers move it; if it is physical,
they cannot, and the prompted route is retired for the main paper.

Everything else is held byte-identical to the frozen P2-long: same rubric, same
decision procedures, same three templates, same temperature, same 512/1024
budget and marker-preserving stop. The instruction lists the identifiers in one
fixed canonical order in every condition, so the listing order is not a fourth
factor.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import time
from pathlib import Path

import yaml

from scripts.experiments.iclr2027_table1_v2.judge_protocol_v3 import (
    load_protocol, objective_criterion,
)

SCHEMES = {
    "ab": {"first": "A", "second": "B", "tie": "TIE"},
    "opaque": {"first": "K7", "second": "Q2", "tie": "TIE"},
}
LABEL_ORDERS = ("AB", "BA")     # AB: slot1 gets the first identifier; BA: slot2 does
POSITIONS = ("y_first", "z_first")


def sha(t):
    return hashlib.sha256(str(t).encode("utf-8")).hexdigest()


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def marker_regex(scheme):
    ids = [re.escape(scheme["first"]), re.escape(scheme["second"]), re.escape(scheme["tie"])]
    return re.compile(r"\[\[(" + "|".join(ids) + r")\]\]")


def stop_strings(scheme):
    return [f"[[{scheme[k]}]]" for k in ("first", "second", "tie")]


def retarget_template(user_template: str, id1: str, id2: str) -> str:
    """Rewrite a P2 template so the two slots carry the given identifiers.

    The frozen templates hard-code the strings "RESPONSE A"/"Reply A"/"CANDIDATE
    A". Only those identifier tokens are substituted; every other byte of the
    template -- headings, blank lines, the trailing cue -- is untouched, so the
    rendering differs from the frozen one in exactly the intended way.
    """
    out = user_template
    for pat, rep in ((r"(RESPONSE|Reply|CANDIDATE)\s+A\b", r"\1 " + id1),
                     (r"(RESPONSE|Reply|CANDIDATE)\s+B\b", r"\1 " + id2)):
        out = re.sub(pat, rep, out)
    return out


def verdict_instruction(scheme) -> str:
    """One fixed canonical listing order, in every condition."""
    return ("First give at most ONE short sentence of justification.\n"
            "Then, on a new line, give your verdict as exactly one of:\n"
            f"[[{scheme['first']}]]\n[[{scheme['second']}]]\n[[{scheme['tie']}]]")


def common_system(scheme) -> str:
    a, b = scheme["first"], scheme["second"]
    return ("You are an impartial evaluator comparing two assistant responses.\n\n"
            "Evaluate ONLY the objective stated below. Do not use any other criterion,\n"
            "even if that criterion would normally matter. Judge the response content\n"
            f"rather than its position as Response {a} or Response {b}.\n\n"
            f"Choose {a} only when Response {a} is materially better on the specified\n"
            f"objective. Choose {b} only when Response {b} is materially better.\n"
            "If neither response is materially better, choose TIE.")


def build_renderings(protocol, rubric_cfg, procedures, pairs, schemes, templates=None):
    """The full crossing, one rendering per cell."""
    out = []
    tpl_ids = set(templates) if templates else {t["id"] for t in protocol.templates}
    for pair in pairs:
        criterion = objective_criterion(rubric_cfg, pair["objective"])
        proc = procedures.get(pair["objective"], "")
        for scheme_name in schemes:
            scheme = SCHEMES[scheme_name]
            sysmsg = "\n\n".join(x for x in (common_system(scheme), criterion, proc,
                                             verdict_instruction(scheme)) if x)
            for tpl in protocol.templates:
                if tpl["id"] not in tpl_ids:
                    continue
                for position, label_order in itertools.product(POSITIONS, LABEL_ORDERS):
                    # physical slots
                    if position == "y_first":
                        slot1, slot2 = pair["response_a"], pair["response_b"]
                        slot1_is_y = True
                    else:
                        slot1, slot2 = pair["response_b"], pair["response_a"]
                        slot1_is_y = False
                    # identifiers on those slots
                    if label_order == "AB":
                        id1, id2 = scheme["first"], scheme["second"]
                    else:
                        id1, id2 = scheme["second"], scheme["first"]
                    user = retarget_template(tpl["user_template"], id1, id2).format(
                        prompt=pair["prompt"], a=slot1, b=slot2)
                    out.append({
                        "pair_id": pair["pair_id"], "objective": pair["objective"],
                        "scheme": scheme_name, "template_id": tpl["id"],
                        "position": position, "label_order": label_order,
                        "slot1_is_learner": slot1_is_y,
                        "slot1_identifier": id1, "slot2_identifier": id2,
                        "first_identifier": scheme["first"],
                        "expected_verdict": pair.get("expected_verdict"),
                        "family": pair.get("family"),
                        "system": sysmsg, "user": user})
    return out


def interpret(rendering, verdict):
    """Map a named identifier back to the learner/comparator and to the slots.

    Three independent readings come out of one verdict, and keeping them
    separate is the entire point of the design:
      ``win_learner``  1 / 0 / 0.5   -- the semantic quantity
      ``win_slot1``    1 / 0 / 0.5   -- the physical first-slot quantity
      ``win_first_id`` 1 / 0 / 0.5   -- the lexical "A"/"K7" quantity
    """
    scheme = SCHEMES[rendering["scheme"]]
    if verdict == scheme["tie"]:
        return {"win_learner": 0.5, "win_slot1": 0.5, "win_first_id": 0.5,
                "verdict": "TIE"}
    if verdict == rendering["slot1_identifier"]:
        slot1 = 1.0
    elif verdict == rendering["slot2_identifier"]:
        slot1 = 0.0
    else:
        raise ValueError(f"verdict {verdict!r} names no identifier in this rendering")
    learner = slot1 if rendering["slot1_is_learner"] else 1.0 - slot1
    first_id = slot1 if rendering["slot1_identifier"] == rendering["first_identifier"] \
        else 1.0 - slot1
    return {"win_learner": learner, "win_slot1": slot1, "win_first_id": first_id,
            "verdict": verdict}


class Runner:
    def __init__(self, model_path, tp, max_model_len, batch_size, gpu_frac):
        from vllm import LLM, SamplingParams
        self.SamplingParams = SamplingParams
        self.llm = LLM(model=model_path, tensor_parallel_size=tp,
                       max_model_len=max_model_len, dtype="bfloat16",
                       gpu_memory_utilization=gpu_frac, trust_remote_code=False)
        self.tok = self.llm.get_tokenizer()
        self.batch_size = batch_size

    def tokenization_report(self):
        """Opaque identifiers are multi-token, and that has to be stated.

        It is harmless here because P2 parses a marker out of generated text
        rather than scoring a restricted single-token distribution -- but the
        same identifiers could NOT be dropped into P1 without breaking exact
        candidate scoring, so the fact is recorded rather than assumed away.
        """
        rep = {}
        for name, scheme in SCHEMES.items():
            rep[name] = {}
            for role in ("first", "second", "tie"):
                s = scheme[role]
                for probe in (s, f"[[{s}]]"):
                    ids = self.tok(probe, add_special_tokens=False)["input_ids"]
                    rep[name][probe] = {"n_tokens": len(ids), "ids": ids}
        return rep

    def _chat(self, system, user):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            return self.tok.apply_chat_template(msgs, tokenize=False,
                                                add_generation_prompt=True,
                                                enable_thinking=False)
        except TypeError:
            return self.tok.apply_chat_template(msgs, tokenize=False,
                                                add_generation_prompt=True)

    def _params(self, scheme_name, max_tokens):
        return self.SamplingParams(
            max_tokens=int(max_tokens), temperature=0.0, top_p=1.0, top_k=-1,
            stop=stop_strings(SCHEMES[scheme_name]), include_stop_str_in_output=True)

    def run(self, renderings, max_tokens, retry_max_tokens):
        results = []
        for i in range(0, len(renderings), self.batch_size):
            chunk = renderings[i:i + self.batch_size]
            # one scheme per batch so the stop strings are correct
            for scheme_name in sorted({r["scheme"] for r in chunk}):
                sub = [r for r in chunk if r["scheme"] == scheme_name]
                prompts = [self._chat(r["system"], r["user"]) for r in sub]
                gen = self.llm.generate(prompts, self._params(scheme_name, max_tokens),
                                        use_tqdm=False)
                batch = [self._observe(r, g) for r, g in zip(sub, gen)]
                need = [k for k, o in enumerate(batch) if not o["valid"]]
                if need and retry_max_tokens:
                    regen = self.llm.generate(
                        [prompts[k] for k in need],
                        self._params(scheme_name, retry_max_tokens), use_tqdm=False)
                    for k, g2 in zip(need, regen):
                        o2 = self._observe(sub[k], g2)
                        o2["first_pass_invalid"] = True
                        batch[k] = o2
                results.extend(batch)
            print(f"    {min(i + self.batch_size, len(renderings))}/{len(renderings)}",
                  flush=True)
        return results

    def _observe(self, rendering, gen):
        out = gen.outputs[0]
        m = marker_regex(SCHEMES[rendering["scheme"]]).findall(out.text)
        base = {k: v for k, v in rendering.items() if k not in ("system", "user")}
        if not m:
            return {**base, "valid": False, "reason": "no_verdict_marker",
                    "n_output_tokens": len(out.token_ids),
                    "finish_reason": getattr(out, "finish_reason", None),
                    "raw_tail": out.text[-300:]}
        return {**base, "valid": True, **interpret(rendering, m[-1]),
                "verdict_token_position": len(out.token_ids),
                "finish_reason": getattr(out, "finish_reason", None)}


def select_pairs(pairs, per_objective, salt="opaque-factorial-v5"):
    """Deterministic, outcome-blind selection: the lowest salted hashes per objective.

    Selecting on a measured quantity -- decisiveness, disagreement, anything --
    would bias the very effect this experiment estimates, so selection uses only
    the pair id.
    """
    by_obj = {}
    for p in pairs:
        by_obj.setdefault(p["objective"], []).append(p)
    out = []
    for obj in sorted(by_obj):
        ranked = sorted(by_obj[obj], key=lambda p: sha(f"{salt}|{p['pair_id']}"))
        out.extend(ranked[:per_objective])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocol", type=Path, required=True)
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--controls", type=Path, default=None)
    ap.add_argument("--per-objective", type=int, default=50)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--judge-model-path", required=True)
    ap.add_argument("--tensor-parallel-size", type=int, default=2)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--schemes", nargs="+", default=["ab", "opaque"])
    args = ap.parse_args()

    protocol = load_protocol(args.protocol)
    raw = yaml.safe_load(Path(args.protocol).read_text())
    procedures = raw.get("objective_procedures") or {}
    rubric_cfg = yaml.safe_load(protocol.rubric_path.read_text())

    pairs = select_pairs(read_jsonl(args.pairs), args.per_objective)
    controls = read_jsonl(args.controls) if args.controls else []
    for c in controls:
        c.setdefault("pair_id", c.get("control_id"))
        c.setdefault("objective", c.get("objective"))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    runner = Runner(args.judge_model_path, args.tensor_parallel_size,
                    args.max_model_len, args.batch_size, args.gpu_memory_utilization)
    tok = runner.tokenization_report()
    print("tokenization:", json.dumps(tok), flush=True)

    t0 = time.time()
    blocks = {}
    for name, rows in (("natural", pairs), ("controls", controls)):
        if not rows:
            continue
        rends = build_renderings(protocol, rubric_cfg, procedures, rows, args.schemes)
        print(f"[{name}] {len(rows)} pairs -> {len(rends)} renderings", flush=True)
        res = runner.run(rends, protocol.max_tokens, protocol.retry_max_tokens)
        (args.out_dir / f"{name}_observations.jsonl").write_text(
            "\n".join(json.dumps(r) for r in res) + "\n")
        blocks[name] = {"n_pairs": len(rows), "n_renderings": len(rends)}

    (args.out_dir / "run_manifest.json").write_text(json.dumps({
        "protocol_sha256": protocol.sha256(), "protocol_name": protocol.name,
        "rubric_sha256": protocol.rubric_sha256,
        "judge_model_path": args.judge_model_path,
        "schemes": args.schemes, "label_orders": list(LABEL_ORDERS),
        "positions": list(POSITIONS),
        "templates": [t["id"] for t in protocol.templates],
        "per_objective": args.per_objective,
        "selection": "lowest sha256('opaque-factorial-v5|'+pair_id) per objective; outcome-blind",
        "tokenization": tok,
        "pairs_input": str(args.pairs), "controls_input": str(args.controls),
        "blocks": blocks,
        "max_tokens": protocol.max_tokens, "retry_max_tokens": protocol.retry_max_tokens,
        "wall_clock_seconds": round(time.time() - t0, 1),
    }, indent=2))
    print(f"done in {time.time() - t0:.0f}s -> {args.out_dir}")


if __name__ == "__main__":
    main()
