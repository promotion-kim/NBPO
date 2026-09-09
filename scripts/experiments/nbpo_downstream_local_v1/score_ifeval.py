#!/usr/bin/env python3
"""IFEval strict and loose accuracy with Google's official implementation.

The verifiers are the published ones, not a reimplementation: this only adapts
the response files into the input record the official `evaluation_lib` expects
and reports both the prompt-level and instruction-level accuracies it returns.

Strict is the headline; loose is reported beside it because the official release
reports both and quoting only the flattering one would misrepresent the metric.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--responses", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from instruction_following_eval import evaluation_lib as ev

    ds = {json.loads(l)["prompt"]: json.loads(l)
          for l in Path(args.dataset).open() if l.strip()}
    rows = [json.loads(l) for l in Path(args.responses).open() if l.strip()]

    inputs, prompt_to_response = [], {}
    for r in rows:
        rec = ds.get(r["prompt"])
        if rec is None:
            continue
        # the HF copy pads every kwargs dict with all possible keys set to null;
        # the official verifiers take only the arguments their own instruction
        # declares, so the nulls are stripped rather than passed through
        kw = [{k: v for k, v in d.items() if v is not None} for d in rec["kwargs"]]
        inputs.append(ev.InputExample(
            key=rec["key"], instruction_id_list=rec["instruction_id_list"],
            prompt=rec["prompt"], kwargs=kw))
        prompt_to_response[rec["prompt"]] = r["output"]

    out = {"label": args.label, "n_matched": len(inputs), "n_dataset": len(ds),
           "n_responses": len(rows)}
    for name, fn in (("strict", ev.test_instruction_following_strict),
                     ("loose", ev.test_instruction_following_loose)):
        res = [fn(inp, prompt_to_response) for inp in inputs]
        prompt_acc = sum(r.follow_all_instructions for r in res) / len(res)
        flat = [x for r in res for x in r.follow_instruction_list]
        out[name] = {"prompt_level_accuracy": prompt_acc,
                     "instruction_level_accuracy": sum(flat) / len(flat),
                     "n_instructions": len(flat)}
        print(f"{args.label:>12} {name:>6}  prompt={prompt_acc:.4f}  "
              f"instruction={sum(flat)/len(flat):.4f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
