from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any

from datasets import load_from_disk
from transformers import AutoTokenizer


def _apply_chat_template_non_thinking(tokenizer: Any, messages: list[dict[str, str]], **kwargs: Any) -> str:
    signature = inspect.signature(tokenizer.apply_chat_template)
    supports_extra_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if "enable_thinking" in signature.parameters or supports_extra_kwargs:
        kwargs.setdefault("enable_thinking", False)
    return tokenizer.apply_chat_template(messages, **kwargs)


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _completion_text(messages: list[dict[str, str]]) -> str:
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("Expected chosen/rejected messages ending with an assistant turn")
    return str(messages[-1].get("content", ""))


def _messages_pair(row: dict[str, Any]) -> tuple[str, str, str]:
    chosen = row["chosen"]
    rejected = row["rejected"]
    if not isinstance(chosen, list) or not isinstance(rejected, list):
        raise TypeError("chosen/rejected are not chat message lists")
    prompt_messages = chosen[:-1]
    prompt = _apply_chat_template_non_thinking(
        row["_tokenizer"],
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt, _completion_text(chosen), _completion_text(rejected)


def _string_pair(row: dict[str, Any]) -> tuple[str, str, str]:
    prompt = row.get("prompt")
    chosen = row.get("chosen")
    rejected = row.get("rejected")
    if not isinstance(prompt, str) or not isinstance(chosen, str) or not isinstance(rejected, str):
        raise TypeError("row does not contain string prompt/chosen/rejected fields")
    return prompt, chosen, rejected


def _prompt_and_completions(tokenizer: Any, row: dict[str, Any]) -> tuple[str, str, str]:
    chosen = row.get("chosen")
    rejected = row.get("rejected")
    if isinstance(chosen, list) and isinstance(rejected, list):
        row = dict(row)
        row["_tokenizer"] = tokenizer
        return _messages_pair(row)
    return _string_pair(row)


def convert_rows(tokenizer: Any, rows: Any, output_path: Path) -> tuple[int, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    prompt_control_markers = 0
    with output_path.open("w", encoding="utf-8") as out:
        for row in rows:
            prompt, chosen_text, rejected_text = _prompt_and_completions(tokenizer, row)
            for label, completion in ((True, chosen_text), (False, rejected_text)):
                record = {
                    "prompt_id": row.get("prompt_id"),
                    "prompt": prompt,
                    "completion": completion,
                    "label": label,
                    "source_pair": row.get("pair_source", "avg_oracle"),
                }
                if "<think>" in record["prompt"]:
                    prompt_control_markers += 1
                if "<think>" in record["completion"]:
                    raise ValueError("Qwen3 thinking token leaked into a KTO completion")
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
    return count, prompt_control_markers


def convert_split(tokenizer: Any, input_path: Path, output_path: Path) -> tuple[int, int]:
    return convert_rows(tokenizer, _iter_jsonl(input_path), output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--pref-dir")
    parser.add_argument("--train-pairs")
    parser.add_argument("--test-pairs")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    output_dir = Path(args.output_dir)
    if args.pref_dir:
        dataset = load_from_disk(args.pref_dir)
        train_n, train_prompt_markers = convert_rows(tokenizer, dataset["train"], output_dir / "train_kto.jsonl")
        test_n, test_prompt_markers = convert_rows(tokenizer, dataset["test"], output_dir / "test_kto.jsonl")
        source = args.pref_dir
    else:
        if not args.train_pairs or not args.test_pairs:
            raise ValueError("Provide either --pref-dir or both --train-pairs and --test-pairs")
        train_n, train_prompt_markers = convert_split(tokenizer, Path(args.train_pairs), output_dir / "train_kto.jsonl")
        test_n, test_prompt_markers = convert_split(tokenizer, Path(args.test_pairs), output_dir / "test_kto.jsonl")
        source = {"train_pairs": args.train_pairs, "test_pairs": args.test_pairs}
    summary = {
        "train_examples": train_n,
        "test_examples": test_n,
        "model": args.model,
        "source": source,
        "completion_think_leak_count": 0,
        "prompt_control_marker_count": train_prompt_markers + test_prompt_markers,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
