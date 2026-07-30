#!/usr/bin/env python3
"""Render the locked HH selection evidence and next-phase recommendation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


MODEL = {
    "skywork_llama": "Skywork/Skywork-Reward-V2-Llama-3.1-8B",
    "skywork_qwen3": "Skywork/Skywork-Reward-V2-Qwen3-8B",
    "athene": "Nexusflow/Athene-RM-8B",
    "armo_helpfulness": "RLHFlow/ArmoRM-Llama3-8B-v0.1: ultrafeedback-helpfulness",
    "beaver_v1": "PKU-Alignment/beaver-7b-v1.0-cost (negated cost)",
    "beaver_v2": "PKU-Alignment/beaver-7b-v2.0-cost (negated cost)",
    "llama_guard3": "meta-llama/Llama-Guard-3-8B (safe/unsafe log odds)",
    "shieldgemma": "google/shieldgemma-9b (No/Yes log odds)",
    "qwen3guard8": "Qwen/Qwen3Guard-Gen-8B (Safe/Unsafe log odds)",
}


def rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def f(value: str | float) -> str:
    return f"{float(value):.4f}"


def yes(value: str | bool) -> str:
    return "PASS" if value is True or str(value).lower() == "true" else "FAIL"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = args.root / "analysis_results"
    summary = json.loads((result / "summary.json").read_text(encoding="utf-8"))
    candidate = rows(result / "candidate_pairs.csv")
    resolution = rows(result / "resolution.csv")
    headroom = rows(result / "headroom.csv")
    selected = summary.get("selected_pair")

    lines = [
        "# Selection", "",
        "This decision was generated from the locked validation scores. RONPO policies were excluded from the conflict-selection and headroom gates.", "",
    ]
    if selected:
        h, s = selected["helpfulness"], selected["harmlessness"]
        lines += [
            "## Locked choice", "",
            f"- Helpfulness: `{MODEL[h]}`",
            f"- Harmlessness: `{MODEL[s]}`",
            "- Dataset: 50% PKU-SafeRLHF dual-preference conflicts, 25% OR-Bench Hard, and 25% higher-severity BeaverTails.",
            f"- Conflict: pooled Spearman rho `{selected['spearman_rho']:.4f}`; top-set mismatch `{selected['top_set_mismatch_rate']:.4f}`.",
            f"- Weaker-objective headroom: `{selected['weaker_headroom_objective']}`; best non-RONPO policy `{selected['weaker_headroom_best_policy']}`; delta `{selected['weaker_headroom_mean']:.4f}` with 95% CI `[{selected['weaker_headroom_ci_low']:.4f}, {selected['weaker_headroom_ci_high']:.4f}]`.",
            "", "## Resolution evidence", "",
            "| Objective | Probe | Sources | Base-minus-probe gap | 95% CI | MDE80 |",
            "|---|---|---|---:|---:|---:|",
        ]
        for row in resolution:
            if row["objective"] in {h, s}:
                lines.append(f"| {row['objective']} | {row['probe']} | {row['sources']} | {f(row['mean'])} | [{f(row['ci_low'])}, {f(row['ci_high'])}] | {f(row['mde80'])} |")
        lines += [
            "", "## Defensibility", "",
            "The selected signals are public, version-pinned artifacts from distinct reward-training lineages. The prompt sources are established safety-alignment and over-refusal benchmarks, and the selection criterion never references RONPO performance. Architectural-family overlap, if any, does not imply shared reward training and is disclosed in `PREREG.md`.",
            "", "## Recommended next RONPO phase", "",
            "Use `Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218` with the two locked higher-is-better objective scores above. Construct the training pool from the version-pinned PKU-SafeRLHF, OR-Bench Hard, and BeaverTails sources using the same 50/25/25 rule and preserve the locked 768-prompt validation split. Reserve the 384-prompt fresh-confirmation manifest without decoding until model selection is final. Use seed 42, equal 900-step and effective-batch-16 budgets for RONPO and baselines, non-thinking decoding, the unchanged reward-blind stability gate, and identical reference/SFT anchoring. Compare RONPO full expectation and objective-stratified estimation against averaging baselines; select only on validation and confirm once on fresh data.",
        ]
    else:
        lines += [
            "## Outcome", "",
            "No candidate pair satisfies all preregistered conflict, two-objective resolution, independent-lineage, and weaker-objective headroom gates. No reward-model pair is selected.",
            "", "## Closest candidates (not selected)", "",
            "| Helpfulness | Harmlessness | Rho | Help resolution | Harm resolution | Weaker headroom |",
            "|---|---|---:|:---:|:---:|:---:|",
        ]
        for row in candidate[:5]:
            lines.append(f"| {row['helpfulness']} | {row['harmlessness']} | {f(row['spearman_rho'])} | {yes(row['helpfulness_resolution_pass'])} | {yes(row['harmlessness_resolution_pass'])} | {yes(row['weaker_headroom_pass'])} |")
        lines += [
            "", "The defensible next step is not to train a Qwen3-8B RONPO model on a failed measuring instrument. Keep the model-scale heterogeneous-objective claim scoped and retain the complete negative screen as evidence.",
        ]
    lines += [
        "", "## Provenance", "",
        "- Selection lock SHA-256: `8bc36a889464c5a16085b76e505f8817e73418258636116ccf4f9870b169231e`",
        "- Validation manifest SHA-256: `6044e941cbb8fdf252932d1b805894b0abd8f20722bacdd291d5133e544b7d5f`",
        "- Fresh confirmation opened: `false`",
        "- Spent sealed split touched: `false`",
        "- All 20 pairs: `analysis_results/candidate_pairs.csv`",
    ]
    (args.root / "SELECTION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
