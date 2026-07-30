#!/usr/bin/env python3
"""Freeze the existing SafeRLHF Stage-4 three-seed response pool."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


METHODS = {
    "ronpo_os": "ronpo_os_stage4",
    "ronpo_topmass": "ronpo_topmass_stage4",
    "inpo_avg": "inpo_avg_stage4",
    "sppo_avg": "sppo_avg_stage4",
    "simpo": "simpo_stage4",
    "ipo": "ipo_stage4",
    "dpo": "dpo_stage4",
    "ht_mnpo_harmless": "ht_mnpo_harmless_stage4",
    "ht_mnpo_helpfulness": "ht_mnpo_helpfulness_stage4",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise TypeError(f"{path}: expected JSON list")
    return value


def source(root: Path, method: str, seed: int) -> tuple[Path, Path]:
    stage_name = METHODS[method]
    if seed == 42:
        base = root / "results/p8_stage4_fresh_default_test_20260718/stage4_eval"
        return (
            base / f"generations/{stage_name}/output_42.json",
            base / f"gates/{stage_name}.json",
        )
    if seed == 43:
        base = root / "results/p10_saferlhf_stage4_seed43_20260718/stage4_stability_p8_locked_panel"
        return base / f"generations/{method}/output_43.json", base / f"gates/{method}.json"
    if seed == 44:
        base = root / "results/p13_saferlhf_seed44_20260718/stage4/stage4_stability_p8_locked_panel"
        return base / f"generations/{method}/output_44.json", base / f"gates/{method}.json"
    raise ValueError(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    manifest_path = args.root / "results/p8_stage4_fresh_default_test_20260718/dataset_manifest/fresh_default_test_1000.jsonl"
    manifest = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(manifest) != 1000 or len({row["prompt_id"] for row in manifest}) != 1000:
        raise RuntimeError("the frozen prompt manifest is not 1,000 unique prompts")
    prompt_ids = [row["prompt_id"] for row in manifest]

    base_dir = args.root / "results/p8_stage4_fresh_default_test_20260718/stage4_eval"
    base_path = base_dir / "generations/base/output_42.json"
    base_gate = base_dir / "gates/base.json"
    items: list[tuple[str, Path, Path]] = [("base", base_path, base_gate)]
    for method in METHODS:
        for seed in (42, 43, 44):
            generation, gate = source(args.root, method, seed)
            items.append((f"{method}_s{seed}", generation, gate))

    discovery = {
        "status": "complete",
        "records": 1000,
        "prompt_manifest": str(manifest_path),
        "prompt_manifest_sha256": sha256(manifest_path),
        "policy_order": [name for name, _, _ in items],
        "policies": {},
        "spent_sealed_split_touched": False,
    }
    loaded: dict[str, list[dict]] = {}
    for name, generation, gate_path in items:
        if not generation.is_file() or not gate_path.is_file():
            raise FileNotFoundError(f"{name}: missing {generation} or {gate_path}")
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        passed = bool(gate.get("passed", gate.get("pass", gate.get("gate_passed", False))))
        if not passed:
            raise RuntimeError(f"{name}: recorded stability gate failed")
        rows = read_rows(generation)
        by_id = {row["prompt_id"]: row for row in rows}
        if len(rows) != 1000 or len(by_id) != 1000 or set(by_id) != set(prompt_ids):
            raise RuntimeError(f"{name}: prompt identity/count mismatch")
        loaded[name] = [by_id[prompt_id] for prompt_id in prompt_ids]
        discovery["policies"][name] = {
            "generation": str(generation),
            "generation_sha256": sha256(generation),
            "gate": str(gate_path),
            "gate_sha256": sha256(gate_path),
            "gate_passed": True,
            "records": len(rows),
        }

    pool_path = args.output / "response_pool_28.jsonl"
    with pool_path.open("w", encoding="utf-8") as handle:
        for index, prompt_row in enumerate(manifest):
            handle.write(json.dumps({
                "prompt_id": prompt_row["prompt_id"],
                "prompt": prompt_row["prompt"],
                "source": prompt_row.get("source"),
                "slice": prompt_row.get("slice"),
                "behavior_label": prompt_row.get("behavior_label"),
                "response_model_names": discovery["policy_order"],
                "all_generated_responses": [
                    loaded[name][index]["generated_text"] for name in discovery["policy_order"]
                ],
            }, ensure_ascii=False) + "\n")
    discovery["response_pool"] = str(pool_path)
    discovery["response_pool_sha256"] = sha256(pool_path)
    (args.output / "DISCOVERY.json").write_text(json.dumps(discovery, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# SafeRLHF Stage-4 three-seed discovery",
        "",
        f"- Frozen prompt panel: 1,000 rows, SHA-256 `{discovery['prompt_manifest_sha256']}`.",
        f"- Frozen response pool: 28 policies, SHA-256 `{discovery['response_pool_sha256']}`.",
        "- Every listed generation has exactly 1,000 prompt-aligned rows and a passing recorded stability gate.",
        "",
        "| Policy | Generation SHA-256 | Gate SHA-256 |",
        "|---|---|---|",
    ]
    for name in discovery["policy_order"]:
        item = discovery["policies"][name]
        lines.append(f"| {name} | `{item['generation_sha256']}` | `{item['gate_sha256']}` |")
    lines.extend(["", "`spent_sealed_split_touched=false`", ""])
    (args.output / "DISCOVERY.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
