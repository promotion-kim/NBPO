from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from datasets import load_from_disk

from mnpo_scripts.build_multi_objective_dataset import build_pairs_for_record


USER_RE = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.DOTALL)


MODE_TO_STRATEGY = {
    "full_expect": "sigma",
    "k_only": "sigma_k_only",
    "uniform": "uniform",
    "a_only": "sigma_a_only",
    "maxmin_pointwise": "maxmin_pointwise",
}


def _extract_user_prompt(prompt: Any) -> str:
    if isinstance(prompt, list) and prompt:
        first = prompt[0]
        if isinstance(first, dict) and first.get("role") == "user":
            return str(first.get("content", "")).strip()
    if not isinstance(prompt, str):
        return str(prompt)
    match = USER_RE.search(prompt)
    if match:
        return match.group(1).strip()
    assistant_marker = "<|im_start|>assistant"
    if assistant_marker in prompt:
        return prompt.split(assistant_marker, 1)[0].strip()
    return prompt.strip()


def _clean_record(row: dict[str, Any]) -> dict[str, Any]:
    out = {
        key: value
        for key, value in row.items()
        if not key.endswith("_logps")
        and "_logps_" not in key
        and key not in {"chat_template_applied", "chosen", "rejected"}
    }
    out["prompt"] = _extract_user_prompt(row.get("prompt"))
    return out


def build_split(
    pref_dir: Path,
    split: str,
    mode: str,
    output: Path,
    max_prompts: int,
    pairs_per_prompt: int,
    common_pair_seed: str,
    adversary_steps: int,
    adversary_alpha: float,
    adversary_kappa: float,
    preference_scale: float,
    policy_mode: str,
    policy_temperature: float,
    adversary_selection: str,
    ronpo_policy_pair_mode: str,
    ronpo_policy_samples_per_atom: int,
) -> int:
    dataset = load_from_disk(str(pref_dir))[split]
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    prompt_count = 0
    with output.open("w", encoding="utf-8") as handle:
        for row in dataset:
            record = _clean_record(dict(row))
            responses = record.get("all_generated_responses")
            objective_scores = record.get("objective_scores")
            objective_names = record.get("objective_names")
            if not isinstance(responses, list) or len(responses) < 2:
                continue
            if not isinstance(objective_scores, dict) or not isinstance(objective_names, list):
                continue
            _, ronpo_pairs = build_pairs_for_record(
                record=record,
                normalization="minmax",
                ronpo_pair_strategy=MODE_TO_STRATEGY[mode],
                tie_threshold=0.0,
                adversary_steps=adversary_steps,
                adversary_alpha=adversary_alpha,
                adversary_kappa=adversary_kappa,
                preference_scale=preference_scale,
                policy_mode=policy_mode,
                policy_temperature=policy_temperature,
                pairs_per_prompt=pairs_per_prompt,
                adversary_selection=adversary_selection,
                ronpo_policy_pair_mode=ronpo_policy_pair_mode,
                ronpo_policy_samples_per_atom=ronpo_policy_samples_per_atom,
                k_only_fixed_atom="avg_worst",
                k_only_response_mode="uniform",
                common_pair_seed=common_pair_seed,
            )
            for pair in ronpo_pairs:
                handle.write(json.dumps(pair, ensure_ascii=False) + "\n")
                count += 1
            prompt_count += 1
            if max_prompts > 0 and prompt_count >= max_prompts:
                break
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pref-dir", required=True)
    parser.add_argument("--mode", required=True, choices=sorted(MODE_TO_STRATEGY))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-prompts", type=int, default=0)
    parser.add_argument("--pairs-per-prompt", type=int, default=3)
    parser.add_argument("--common-pair-seed", default="42")
    parser.add_argument("--adversary-steps", type=int, default=25)
    parser.add_argument("--adversary-alpha", type=float, default=1.0)
    parser.add_argument("--adversary-kappa", type=float, default=0.05)
    parser.add_argument("--preference-scale", type=float, default=8.0)
    parser.add_argument("--policy-mode", choices=["uniform", "softmax"], default="uniform")
    parser.add_argument("--policy-temperature", type=float, default=0.2)
    parser.add_argument("--adversary-selection", choices=["all", "top", "sample"], default="all")
    parser.add_argument(
        "--ronpo-policy-pair-mode",
        choices=[
            "best_vs_adversary",
            "all_policy_vs_adversary",
            "relative_policy_vs_policy",
            "expected_relative_policy_vs_policy",
        ],
        default="expected_relative_policy_vs_policy",
    )
    parser.add_argument("--ronpo-policy-samples-per-atom", type=int, default=1)
    args = parser.parse_args()

    pref_dir = Path(args.pref_dir)
    output_dir = Path(args.output_dir)
    train_n = build_split(
        pref_dir=pref_dir,
        split="train",
        mode=args.mode,
        output=output_dir / "train_ronpo.jsonl",
        max_prompts=args.max_prompts,
        pairs_per_prompt=args.pairs_per_prompt,
        common_pair_seed=args.common_pair_seed,
        adversary_steps=args.adversary_steps,
        adversary_alpha=args.adversary_alpha,
        adversary_kappa=args.adversary_kappa,
        preference_scale=args.preference_scale,
        policy_mode=args.policy_mode,
        policy_temperature=args.policy_temperature,
        adversary_selection=args.adversary_selection,
        ronpo_policy_pair_mode=args.ronpo_policy_pair_mode,
        ronpo_policy_samples_per_atom=args.ronpo_policy_samples_per_atom,
    )
    test_n = build_split(
        pref_dir=pref_dir,
        split="test",
        mode=args.mode,
        output=output_dir / "test_ronpo.jsonl",
        max_prompts=args.max_prompts,
        pairs_per_prompt=args.pairs_per_prompt,
        common_pair_seed=args.common_pair_seed,
        adversary_steps=args.adversary_steps,
        adversary_alpha=args.adversary_alpha,
        adversary_kappa=args.adversary_kappa,
        preference_scale=args.preference_scale,
        policy_mode=args.policy_mode,
        policy_temperature=args.policy_temperature,
        adversary_selection=args.adversary_selection,
        ronpo_policy_pair_mode=args.ronpo_policy_pair_mode,
        ronpo_policy_samples_per_atom=args.ronpo_policy_samples_per_atom,
    )
    summary = {
        "source": str(pref_dir),
        "mode": args.mode,
        "strategy": MODE_TO_STRATEGY[args.mode],
        "adversary_steps": args.adversary_steps,
        "adversary_alpha": args.adversary_alpha,
        "adversary_kappa": args.adversary_kappa,
        "preference_scale": args.preference_scale,
        "policy_mode": args.policy_mode,
        "policy_temperature": args.policy_temperature,
        "adversary_selection": args.adversary_selection,
        "ronpo_policy_pair_mode": args.ronpo_policy_pair_mode,
        "ronpo_policy_samples_per_atom": args.ronpo_policy_samples_per_atom,
        "k_only_response_mode": "uniform",
        "pairs_per_prompt": args.pairs_per_prompt,
        "common_pair_seed": args.common_pair_seed,
        "train_examples": train_n,
        "test_examples": test_n,
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
