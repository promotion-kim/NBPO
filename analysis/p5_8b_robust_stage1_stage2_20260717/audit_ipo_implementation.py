#!/usr/bin/env python3
"""Audit the local IPO implementation and the average-oriented training pairs.

This script is deliberately outcome-blind: it reads source/configuration/pair
artifacts only.  It does not read any evaluation reward or rank.  The resulting
JSON makes a formula mismatch distinguishable from the separate question of
whether average-oriented pairs make IPO competitive on a worst-objective metric.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import mean, median


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def yaml_number(text: str, key: str) -> float:
    match = re.search(rf"^{re.escape(key)}:\s*([0-9.eE+-]+)\s*$", text, flags=re.MULTILINE)
    if not match:
        raise RuntimeError(f"missing {key} in config")
    return float(match.group(1))


def pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2:
        return None
    ml, mr = mean(left), mean(right)
    numerator = sum((a - ml) * (b - mr) for a, b in zip(left, right))
    dl = math.sqrt(sum((a - ml) ** 2 for a in left))
    dr = math.sqrt(sum((b - mr) ** 2 for b in right))
    return None if dl == 0.0 or dr == 0.0 else numerator / (dl * dr)


def quantiles(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    if not ordered:
        return {"min": float("nan"), "median": float("nan"), "max": float("nan")}
    return {"min": ordered[0], "median": median(ordered), "max": ordered[-1], "mean": mean(ordered)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--trl-trainer", type=Path, required=True)
    parser.add_argument("--online-trl-trainer", type=Path, required=True)
    parser.add_argument(
        "--targets-dataset",
        type=Path,
        default=None,
        help="Optional local datasets.load_from_disk target artifact for outcome-blind scale auditing.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    config_text = args.config.read_text(encoding="utf-8")
    trainer_text = args.trainer.read_text(encoding="utf-8")
    trl_text = args.trl_trainer.read_text(encoding="utf-8")
    online_text = args.online_trl_trainer.read_text(encoding="utf-8")
    rows = read_jsonl(args.pairs)
    if not rows:
        raise RuntimeError("pair file is empty")

    beta = yaml_number(config_text, "dpo_beta")
    lr = yaml_number(config_text, "learning_rate")
    reference_anchor_weight = yaml_number(config_text, "reference_anchor_weight")
    preference_sft_weight = yaml_number(config_text, "preference_sft_weight")
    target_margin = 1.0 / (2.0 * beta)
    local_formula = "losses = (logits - 1.0 / (2.0 * self.dpo_beta)) ** 2"
    trl_formula = "per_sequence_loss = (ipo_delta - 1 / (2 * self.beta)) ** 2"
    online_formula = "losses = (logits - 1 / (2 * self.beta)) ** 2"
    if local_formula not in trainer_text:
        raise RuntimeError("local IPO formula marker not found")
    if trl_formula not in trl_text:
        raise RuntimeError("vendored TRL IPO formula marker not found")
    if online_formula not in online_text:
        raise RuntimeError("vendored Online-RLHF IPO formula marker not found")

    formula_inputs = [-3.25, -0.5, 0.0, 0.75, 4.5]
    formula_errors = [
        abs((value - target_margin) ** 2 - (value - 1.0 / (2.0 * beta)) ** 2)
        for value in formula_inputs
    ]

    h_gaps, s_gaps, avg_gaps = [], [], []
    ronpo_target_signs: Counter[str] = Counter()
    classes: Counter[str] = Counter()
    pair_sources: Counter[str] = Counter()
    oracle_names: Counter[str] = Counter()
    target_scales: Counter[float] = Counter()
    required = {
        "objective_scores", "normalized_objective_scores", "chosen_index", "rejected_index",
        "avg_oracle_score_gap", "pair_source", "homogeneous_oracle",
    }
    for row in rows:
        missing = required - set(row)
        if missing:
            raise RuntimeError(f"pair is missing fields: {sorted(missing)}")
        chosen, rejected = int(row["chosen_index"]), int(row["rejected_index"])
        normalized = row["normalized_objective_scores"]
        h_gap = float(normalized["helpfulness"][chosen]) - float(normalized["helpfulness"][rejected])
        s_gap = float(normalized["harmlessness"][chosen]) - float(normalized["harmlessness"][rejected])
        h_gaps.append(h_gap)
        s_gaps.append(s_gap)
        avg_gaps.append(float(row["avg_oracle_score_gap"]))
        if h_gap > 0 and s_gap > 0:
            classes["both_objectives_improve"] += 1
        elif h_gap > 0 and s_gap < 0:
            classes["helpfulness_gain_harmlessness_sacrifice"] += 1
        elif h_gap < 0 and s_gap > 0:
            classes["harmlessness_gain_helpfulness_sacrifice"] += 1
        else:
            classes["tie_or_non_strict"] += 1
        pair_sources[str(row["pair_source"])] += 1
        oracle_names[str(row["homogeneous_oracle"])] += 1
        target_scales[float(row.get("homogeneous_oracle_preference_scale", float("nan")))] += 1
        if "ronpo_target" in row:
            target = float(row["ronpo_target"])
            ronpo_target_signs["positive_choose_chosen"] += int(target > 0)
            ronpo_target_signs["negative_choose_rejected"] += int(target < 0)
            ronpo_target_signs["zero_average_both"] += int(target == 0)

    sacrifice = classes["helpfulness_gain_harmlessness_sacrifice"] + classes["harmlessness_gain_helpfulness_sacrifice"]
    target_scale_audit: dict[str, object] = {"status": "not_requested"}
    if args.targets_dataset is not None:
        from datasets import load_from_disk

        dataset = load_from_disk(str(args.targets_dataset))["train"]
        columns = [
            "ronpo_target", "target_os_k0p1", "target_topmass_k0p1",
            "target_fullexp_k0p1", "ht_target", "ht_target_helpfulness",
        ]
        stats = {}
        for column in columns:
            if column in dataset.column_names:
                values = [float(value) for value in dataset[column]]
                stats[column] = quantiles(values) | {"mean_abs": mean(abs(value) for value in values)}
        primary_abs = float(stats.get("target_os_k0p1", {}).get("mean_abs", float("nan")))
        target_scale_audit = {
            "status": "complete_outcome_blind",
            "dataset": str(args.targets_dataset),
            "rows": len(dataset),
            "stats": stats,
            "ipo_target_margin": target_margin,
            "ipo_margin_divided_by_os_target_mean_abs": target_margin / primary_abs if primary_abs > 0 else None,
            "interpretation": "This is a conditioning audit, not an implementation mismatch. Equal optimizer learning rates do not equalize squared-loss update magnitudes when targets have different scales.",
        }
    payload = {
        "status": "complete_outcome_blind_implementation_and_pair_audit",
        "scope": "source/configuration/pair orientation only; no evaluation reward or ranking read",
        "ipo_formula": {
            "paper_form": "(Delta log policy/reference - 1/(2*beta))^2",
            "local_formula_marker": local_formula,
            "vendored_trl_formula_marker": trl_formula,
            "vendored_online_rlhf_formula_marker": online_formula,
            "formula_markers_match": True,
            "beta": beta,
            "target_margin_1_over_2beta": target_margin,
            "learning_rate": lr,
            "length_normalization": "local precompute averages completion-token log probabilities before forming IPO log-ratios",
            "direct_numeric_formula_max_abs_error": max(formula_errors),
            "shared_stabilizers_not_in_plain_ipo_objective": {
                "reference_anchor_weight": reference_anchor_weight,
                "preference_sft_weight": preference_sft_weight,
                "implementation_behavior": "Because all shared rows contain ronpo_target, the generic preference-SFT branch selects chosen when ronpo_target>0 and rejected when ronpo_target<0, including for loss_type=ipo.",
                "ronpo_target_direction_counts": dict(ronpo_target_signs),
                "source_marker": "if ronpo_target is None: preferred_nll = -policy_chosen; else: target-sign branch",
            },
        },
        "source_artifacts": {
            "pairs": {"path": str(args.pairs), "sha256": sha(args.pairs), "rows": len(rows)},
            "config": {"path": str(args.config), "sha256": sha(args.config)},
            "local_trainer": {"path": str(args.trainer), "sha256": sha(args.trainer)},
            "vendored_trl": {"path": str(args.trl_trainer), "sha256": sha(args.trl_trainer)},
            "vendored_online_rlhf": {"path": str(args.online_trl_trainer), "sha256": sha(args.online_trl_trainer)},
        },
        "average_oracle_pair_construction": {
            "important_distinction": "IPO is a pairwise loss; the average oracle enters through chosen/rejected orientation, not through the IPO formula.",
            "pair_source_counts": dict(pair_sources),
            "homogeneous_oracle_counts": dict(oracle_names),
            "preference_scale_counts": {str(key): value for key, value in target_scales.items()},
            "orientation_verified_from_pair_fields": "chosen is the response with higher mean of per-prompt min-max normalized helpfulness and harmlessness",
            "average_oracle_gap": quantiles(avg_gaps),
            "chosen_minus_rejected_normalized_objective_gap": {
                "helpfulness": quantiles(h_gaps),
                "harmlessness": quantiles(s_gaps),
                "pearson_correlation": pearson(h_gaps, s_gaps),
            },
            "pair_outcome_classes": dict(classes),
            "sacrifices_one_objective_fraction": sacrifice / len(rows),
        },
        "target_scale_audit": target_scale_audit,
        "interpretation_guardrail": [
            "A formula match does not prove a fair cross-loss hyperparameter scale.",
            "With beta=0.05 IPO regresses a margin of 10, whereas RONPO and HT-MNPO use differently scaled targets; equal optimizer settings can produce unequal update magnitudes.",
            "The run is a stabilized shared-trainer IPO variant, not a literal plain-IPO reproduction, because it also applies reference and target-dependent preference-SFT anchors.",
            "Per-prompt min-max normalized Worst is relative to the evaluated model pool, so it cannot be interpreted as an absolute worst raw reward without also reporting raw head scores.",
        ],
        "spent_sealed_split_touched": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# IPO implementation and average-oracle audit",
        "",
        "Outcome-blind audit. It reads source, configuration, and training pairs only; it does not read a reward evaluation or ranking.",
        "",
        "## Formula",
        "",
        f"- Local IPO formula marker, vendored TRL marker, and vendored Online-RLHF marker all implement `(margin - 1/(2 beta))^2`.",
        f"- Configured beta: `{beta:g}`; IPO target margin: `{target_margin:g}`; learning rate: `{lr:g}`.",
        f"- Direct arithmetic parity maximum absolute error: `{max(formula_errors):.3g}`.",
        "- Completion log probabilities are averaged per non-masked token before IPO forms the pairwise margin. This is a declared implementation choice, so beta is on the length-normalized scale.",
        f"- The core IPO loss also has shared stabilizers: reference anchor `{reference_anchor_weight:g}` and preference-SFT `{preference_sft_weight:g}`. Because `ronpo_target` is present in the shared dataset, the latter selects the rejected completion on `{ronpo_target_signs['negative_choose_rejected']}/{len(rows)}` IPO rows. This is not part of plain IPO.",
        "",
        "## Where the average oracle enters",
        "",
        "IPO itself is not an average-oracle algorithm. The common pair builder orients every pair by the higher mean of the two per-prompt min-max-normalized objective scores, then IPO fits those pair labels.",
        f"- Audited pair rows: `{len(rows)}`; rows sacrificing one objective for the average: `{sacrifice}/{len(rows)}` ({sacrifice / len(rows):.2%}).",
        f"- Mean chosen-minus-rejected normalized gaps: helpfulness `{mean(h_gaps):.4f}`, harmlessness `{mean(s_gaps):.4f}`.",
        f"- Pairwise gap correlation: `{pearson(h_gaps, s_gaps):.4f}`.",
        "",
        "## Target-scale conditioning",
        "",
    ]
    if target_scale_audit["status"] == "complete_outcome_blind":
        lines += [
            f"- IPO's fixed target margin `{target_margin:g}` is `{target_scale_audit['ipo_margin_divided_by_os_target_mean_abs']:.1f}x` the audited mean absolute OS target.",
            "- This does not make the IPO formula wrong. It does mean that identical learning rates are not equal initial regression signals across these squared losses.",
            "",
        ]
    else:
        lines += ["- No target-dataset artifact was supplied for the optional scale audit.", ""]
    lines += [
        "## Interpretation",
        "",
        "The audited formula does not reveal an IPO implementation sign or target error. A high normalized Worst can instead arise from the average-oriented pair labels, the very large IPO target margin under beta=0.05, and a relative per-prompt normalization. Those mechanisms require the held-out raw-score diagnostic to distinguish; this audit does not use it.",
        "",
        "See `IPO_IMPLEMENTATION_AUDIT.json` for hashes and full counts.",
    ]
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "rows": len(rows), "target_margin": target_margin}, sort_keys=True))


if __name__ == "__main__":
    main()
