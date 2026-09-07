#!/usr/bin/env python3
"""Run a v3 judge protocol over a set of pairs, with adaptive adjudication.

Works on two kinds of input, because the protocol has to be selected on
calibration controls and then applied unchanged to the real pool:

``--controls``  the calibration set (each row carries an expected verdict)
``--pairs``     any jsonl of {pair_id, objective, prompt, response_a, response_b}

For ``P1`` the verdict distribution is read exactly. The chat template with
``enable_thinking: false`` ends precisely where the verdict token goes, so one
constrained decoding step over the three verified single-token labels yields the
restricted softmax directly. The three log-probabilities are re-normalized over
the label set explicitly and the pre-normalization mass is recorded, so the
result does not depend on whether the backend already normalized after masking.

``P0`` and ``P2`` generate text and parse a marker; their one-hot distribution
goes through the identical downstream algebra, which is what makes the
protocols comparable on one set of metrics.

``P3`` is the composition the calibration argued for. P2's explicit
objective-specific decision procedure is what repairs directional accuracy and
position bias -- P1's soft logits do not, on their own, help the judge apply the
criterion -- while P1's constrained scoring is what supplies the uncertainty
that P2, being a hard verdict, cannot express. P3 runs both in two exact stages:
generate the one-sentence justification, then append it to the assistant turn
and score the three label tokens at the position that follows. Stage two is the
same single constrained step as P1, so the distribution is exact and the
rationale is in context when it is taken.

Adaptive adjudication (Section D): a pair whose forward/reverse semantic scores
differ by more than ``order_gap_threshold``, or whose mean normalized entropy
exceeds ``entropy_threshold``, is re-scored under the remaining templates in
both orders. A pair still unstable at the budget is **not** forced to a winner:
it keeps its soft margin, is flagged ``unresolved_uncertainty``, and stays in
the tensor.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

import yaml

from scripts.experiments.iclr2027_table1_v2.judge_protocol_v3 import (
    CHOICES,
    FORWARD,
    REVERSE,
    build_observation,
    combine,
    hard_probs_from_verdict,
    load_protocol,
    needs_adjudication,
    objective_criterion,
    render,
    verify_label_tokenization,
)

MARKER = re.compile(r"\[\[(A|B|TIE)\]\]")


def parse_marker(text: str):
    m = MARKER.findall(str(text).upper())
    return m[-1] if m else None


def read_jsonl(path: Path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def build_renderings(protocol, rubric_cfg, pairs, template_ids):
    """One rendering per (pair, template, presentation order)."""
    out = []
    for pair in pairs:
        criterion = objective_criterion(rubric_cfg, pair["objective"])
        for tpl in protocol.templates:
            if tpl["id"] not in template_ids:
                continue
            t = dict(tpl)
            extra = t.get("extra_system") or ""
            if protocol.kind == "P2" and "__PROCEDURE__" in extra:
                procs = yaml.safe_load(protocol_path_cache["raw"]).get(
                    "objective_procedures") or {}
                # substitute in place so an appended analysis-order clause is kept
                t["extra_system"] = extra.replace(
                    "__PROCEDURE__", procs.get(pair["objective"], ""))
            for order in (FORWARD, REVERSE):
                a, b = ((pair["response_a"], pair["response_b"]) if order == FORWARD
                        else (pair["response_b"], pair["response_a"]))
                system, user = render(protocol, t, criterion, pair["prompt"], a, b)
                out.append({"pair_id": pair["pair_id"], "objective": pair["objective"],
                            "template_id": t["id"], "presentation_order": order,
                            "system": system, "user": user})
    return out


protocol_path_cache = {}


class VllmRunner:
    def __init__(self, protocol, model_path, tp, max_model_len, batch_size, gpu_frac):
        from vllm import LLM, SamplingParams
        self.protocol = protocol
        self.SamplingParams = SamplingParams
        self.llm = LLM(model=model_path, tensor_parallel_size=tp,
                       max_model_len=max_model_len, dtype="bfloat16",
                       gpu_memory_utilization=gpu_frac, trust_remote_code=False)
        self.tok = self.llm.get_tokenizer()
        self.batch_size = batch_size
        self.label_ids = protocol.label_ids
        self.id_to_choice = {protocol.labels[c]["id"]: c for c in CHOICES}

    def _chat(self, system, user):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            return self.tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return self.tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)

    def verify_tokenization(self):
        prefix = self._chat("s", "u")
        return verify_label_tokenization(self.protocol, self.tok, prefix)

    def _label_params(self):
        d = self.protocol.decoding
        return self.SamplingParams(
            max_tokens=1, temperature=float(d.get("temperature", 0.0)),
            top_p=float(d.get("top_p", 1.0)), top_k=int(d.get("top_k", -1)),
            logprobs=len(self.label_ids), allowed_token_ids=list(self.label_ids))

    def run_two_stage(self, renderings, rationale_tokens: int, cue: str):
        """P3: generate the rationale, then score the labels after it, exactly.

        The rationale is put back into the assistant turn before scoring, so the
        distribution is conditioned on the reasoning the judge just produced
        rather than on the prompt alone.
        """
        d = self.protocol.decoding
        gen_params = self.SamplingParams(
            max_tokens=rationale_tokens, temperature=float(d.get("temperature", 0.0)),
            top_p=float(d.get("top_p", 1.0)), top_k=int(d.get("top_k", -1)),
            stop=["\n"])
        results = []
        for i in range(0, len(renderings), self.batch_size):
            chunk = renderings[i:i + self.batch_size]
            stage1 = [self._chat(r["system"], r["user"]) for r in chunk]
            rationales = [g.outputs[0].text.strip().replace("\n", " ")
                          for g in self.llm.generate(stage1, gen_params, use_tqdm=False)]
            stage2 = [p + rat + cue for p, rat in zip(stage1, rationales)]
            gen = self.llm.generate(stage2, self._label_params(), use_tqdm=False)
            for r, g, rat in zip(chunk, gen, rationales):
                obs = self._to_observation(r, g, force_p1=True)
                if obs.get("valid"):
                    obs["rationale"] = rat[:400]
                results.append(obs)
            print(f"    scored {min(i + self.batch_size, len(renderings))}"
                  f"/{len(renderings)}", flush=True)
        return results

    def run(self, renderings):
        d = self.protocol.decoding
        if self.protocol.kind == "P3":
            return self.run_two_stage(
                renderings, int(getattr(self.protocol, "rationale_tokens", 64)),
                getattr(self.protocol, "verdict_cue", "\nVerdict: "))
        if self.protocol.kind == "P1":
            params = self.SamplingParams(
                max_tokens=1, temperature=float(d.get("temperature", 0.0)),
                top_p=float(d.get("top_p", 1.0)), top_k=int(d.get("top_k", -1)),
                logprobs=len(self.label_ids), allowed_token_ids=list(self.label_ids))
        else:
            params = self._marker_params(self.protocol.max_tokens)
        results = []
        for i in range(0, len(renderings), self.batch_size):
            chunk = renderings[i:i + self.batch_size]
            prompts = [self._chat(r["system"], r["user"]) for r in chunk]
            gen = self.llm.generate(prompts, params, use_tqdm=False)
            batch = [self._to_observation(r, g) for r, g in zip(chunk, gen)]

            # Deterministic single retry at a larger budget, for renderings whose
            # rationale ran past the cap before the marker. Same rendering, same
            # temperature -- the only change is headroom. A verdict is NEVER
            # inferred from a truncated rationale and an invalid output is NEVER
            # converted to TIE; it stays invalid and is counted.
            if self.protocol.retry_max_tokens and self.protocol.kind != "P1":
                need = [k for k, o in enumerate(batch) if not o.get("valid")]
                if need:
                    rp = self._marker_params(self.protocol.retry_max_tokens)
                    regen = self.llm.generate([prompts[k] for k in need], rp,
                                              use_tqdm=False)
                    for k, g2 in zip(need, regen):
                        o2 = self._to_observation(chunk[k], g2)
                        o2["first_pass_invalid"] = True
                        o2["retried_at_max_tokens"] = self.protocol.retry_max_tokens
                        batch[k] = o2
                    print(f"    retried {len(need)} unparsed renderings at "
                          f"{self.protocol.retry_max_tokens} tokens", flush=True)
            results.extend(batch)
            print(f"    scored {min(i + self.batch_size, len(renderings))}"
                  f"/{len(renderings)}", flush=True)
        return results

    def _marker_params(self, max_tokens):
        """Sampling params that stop at the first complete verdict marker.

        `include_stop_str_in_output` keeps the marker in the text -- without it
        the stop string is trimmed and the very thing being parsed disappears.
        Stopping on the three FULL markers rather than on `]]` avoids halting on
        a bracket that happens to occur inside the rationale.
        """
        d = self.protocol.decoding
        kw = dict(max_tokens=int(max_tokens),
                  temperature=float(d.get("temperature", 0.0)),
                  top_p=float(d.get("top_p", 1.0)), top_k=int(d.get("top_k", -1)))
        if self.protocol.stop_after_marker:
            kw["stop"] = ["[[A]]", "[[B]]", "[[TIE]]"]
            kw["include_stop_str_in_output"] = True
        return self.SamplingParams(**kw)

    def _to_observation(self, rendering, gen, force_p1: bool = False):
        out = gen.outputs[0]
        if force_p1 or self.protocol.kind == "P1":
            lp = out.logprobs[0] if out.logprobs else {}
            raw = {}
            for tid, choice in self.id_to_choice.items():
                entry = lp.get(tid)
                raw[choice] = math.exp(entry.logprob) if entry is not None else 0.0
            mass = sum(raw.values())
            if mass <= 0:
                return {**rendering, "valid": False, "reason": "no label mass returned",
                        "raw_judge_output": out.text}
            probs = {c: raw[c] / mass for c in CHOICES}
            obs = build_observation(probs, rendering["presentation_order"],
                                    rendering["template_id"], raw=out.text)
            obs["pre_normalization_label_mass"] = mass
        else:
            verdict = parse_marker(out.text)
            if verdict is None:
                # No marker: the rationale was cut off, or the model never
                # committed. Either way this is invalid, not a tie.
                return {**rendering, "valid": False, "reason": "no_verdict_marker",
                        "raw_judge_output": out.text,
                        "n_output_tokens": len(out.token_ids),
                        "finish_reason": getattr(out, "finish_reason", None)}
            obs = build_observation(hard_probs_from_verdict(verdict),
                                    rendering["presentation_order"],
                                    rendering["template_id"], raw=out.text)
            obs["verdict_token_position"] = len(out.token_ids)
            obs["finish_reason"] = getattr(out, "finish_reason", None)
        obs.update({"pair_id": rendering["pair_id"], "objective": rendering["objective"],
                    "valid": True})
        return obs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocol", type=Path, required=True)
    ap.add_argument("--controls", type=Path, default=None)
    ap.add_argument("--pairs", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--judge-model-path", required=True)
    ap.add_argument("--tensor-parallel-size", type=int, default=2)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-adjudication", action="store_true")
    ap.add_argument("--all-templates", action="store_true",
                    help="score EVERY template on every pair in one pass. Calibration "
                         "needs this: split-half reliability requires at least two "
                         "independent observations per (pair, order), which the adaptive "
                         "path only produces for the pairs it happens to flag.")
    args = ap.parse_args()

    protocol = load_protocol(args.protocol)
    protocol_path_cache["raw"] = Path(args.protocol).read_text()
    rubric_cfg = yaml.safe_load(protocol.rubric_path.read_text())

    if bool(args.controls) == bool(args.pairs):
        raise SystemExit("give exactly one of --controls or --pairs")
    rows = read_jsonl(args.controls or args.pairs)
    if args.limit:
        rows = rows[:args.limit]
    for r in rows:
        r.setdefault("pair_id", r.get("control_id"))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    runner = VllmRunner(protocol, args.judge_model_path, args.tensor_parallel_size,
                        args.max_model_len, args.batch_size, args.gpu_memory_utilization)
    tok_report = runner.verify_tokenization()
    print("tokenization verified:", json.dumps(tok_report), flush=True)

    t0 = time.time()
    force_all = args.all_templates or protocol.always_all_templates
    base_templates = ({t["id"] for t in protocol.templates} if force_all
                      else {protocol.templates[0]["id"]})
    print(f"[pass 1] {len(rows)} pairs x 2 orders under template "
          f"{sorted(base_templates)}", flush=True)
    observations = runner.run(build_renderings(protocol, rubric_cfg, rows, base_templates))

    by_pair = {}
    for o in observations:
        by_pair.setdefault(o["pair_id"], []).append(o)

    adjudicated = set()
    if (not args.no_adjudication and not force_all
            and protocol.kind != "P0" and len(protocol.templates) > 1):
        unstable = []
        for r in rows:
            got = [o for o in by_pair.get(r["pair_id"], []) if o.get("valid")]
            if len({o["presentation_order"] for o in got}) < 2:
                continue
            if needs_adjudication(combine(got), protocol.order_gap_threshold,
                                  protocol.entropy_threshold):
                unstable.append(r)
        extra = {t["id"] for t in protocol.templates[1:protocol.max_templates_when_unstable]}
        print(f"[pass 2] {len(unstable)}/{len(rows)} pairs unstable -> "
              f"{len(extra)} extra templates x 2 orders", flush=True)
        if unstable and extra:
            for o in runner.run(build_renderings(protocol, rubric_cfg, unstable, extra)):
                by_pair.setdefault(o["pair_id"], []).append(o)
            adjudicated = {r["pair_id"] for r in unstable}

    results = []
    for r in rows:
        got = [o for o in by_pair.get(r["pair_id"], []) if o.get("valid")]
        invalid = [o for o in by_pair.get(r["pair_id"], []) if not o.get("valid")]
        if len({o["presentation_order"] for o in got}) < 2:
            results.append({**{k: r[k] for k in ("pair_id", "objective") if k in r},
                            "valid": False, "n_invalid_renderings": len(invalid)})
            continue
        agg = combine(got)
        still = needs_adjudication(agg, protocol.order_gap_threshold,
                                   protocol.entropy_threshold)
        results.append({
            "pair_id": r["pair_id"], "objective": r["objective"], "valid": True,
            **agg,
            "adjudicated": r["pair_id"] in adjudicated,
            "unresolved_uncertainty": bool(still and r["pair_id"] in adjudicated),
            "n_invalid_renderings": len(invalid),
            "expected_verdict": r.get("expected_verdict"),
            "family": r.get("family"),
            "observations": [{k: v for k, v in o.items()
                              if k in ("presentation_order", "template_id", "probs",
                                       "semantic_score", "normalized_entropy",
                                       "hard_argmax", "pre_normalization_label_mass")}
                             for o in got],
        })

    tag = protocol.name
    (args.out_dir / f"{tag}_results.jsonl").write_text(
        "\n".join(json.dumps(x) for x in results) + "\n")
    (args.out_dir / f"{tag}_run_manifest.json").write_text(json.dumps({
        **protocol.manifest(),
        "judge_model_path": args.judge_model_path,
        "tokenization_verification": tok_report,
        "n_pairs": len(rows),
        "n_renderings": sum(len(v) for v in by_pair.values()),
        "n_adjudicated": len(adjudicated),
        "wall_clock_seconds": round(time.time() - t0, 1),
        "input": str(args.controls or args.pairs),
        "adjudication_enabled": not args.no_adjudication and not force_all,
        "all_templates_mode": bool(force_all),
        "first_pass_invalid_renderings": sum(
            1 for v in by_pair.values() for o in v if o.get("first_pass_invalid")),
        "final_unresolved_invalid_renderings": sum(
            1 for v in by_pair.values() for o in v if not o.get("valid")),
        "templates_scored": sorted(base_templates),
    }, indent=2))
    print(f"\nwrote {args.out_dir / f'{tag}_results.jsonl'} "
          f"({len(results)} pairs, {sum(len(v) for v in by_pair.values())} renderings, "
          f"{time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
