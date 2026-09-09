"""Independently reaggregate the fixed three-identity primary evaluation.

CPU/read-only inputs. Never select another result path, launch inference, edit
paper sources, or convert missing scores to zero. TeX requires complete data.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from scripts.experiments.nbpo_repair_20260909.common import digest, file_hash, object_hash, write_json
from scripts.experiments.nbpo_repair_20260909.evaluate_responses import (
    assert_matched_protocol, KEYWORD_REFUSAL,
)

LABELS = ("base", "wbc_primary_v1", "mse_primary_v1")
COUNTS = {"ifeval": 541, "gsm8k": 1319, "harmbench": 320, "alpaca_eval": 805,
          "arena_hard": 750, "saferlhf": 1500, "xstest": 450}
SEED, REPS = 20260909, 2000
DEFINITIONS = {
    "ifeval": {f"{kind}_{unit}_accuracy": (f"{kind}_prompt" if unit == "prompt" else f"{kind}_instructions_correct",
                  None if unit == "prompt" else "n_instructions") for kind in ("strict", "loose") for unit in ("prompt", "instruction")},
    "gsm8k": {"exact_match": ("correct", None), "parse_failure_rate": ("parse_failure", None)},
    "harmbench": {"overall": ("harmful", None)},
}


class Invalid(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise Invalid(message)


def close(a, b, name):
    if a is None or b is None:
        require(a is b, f"{name}: missing and numeric values disagree")
    else:
        aa, bb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        require(aa.shape == bb.shape and np.isfinite(aa).all() and np.isfinite(bb).all()
                and np.allclose(aa, bb, rtol=1e-10, atol=1e-12), f"{name}: independent recomputation disagrees")


def keyed(rows, key="uid"):
    index = {row[key]: row for row in rows}
    require(len(index) == len(rows), f"Duplicate per-item {key}")
    return index


def bootstrap(values, denominators=None):
    """Independent fixed-seed prompt bootstrap; vectors and ratio denominators."""
    a = np.asarray(values, dtype=np.float64)
    if not len(a):
        return {"estimate": None, "ci95": None, "n_prompts": 0}
    require(a.ndim in (1, 2) and np.isfinite(a).all(), "Invalid bootstrap observations")
    den = np.ones(len(a)) if denominators is None else np.asarray(denominators, dtype=np.float64)
    require(den.shape == (len(a),) and np.isfinite(den).all() and (den > 0).all(), "Invalid metric denominators")
    rng, draws = np.random.default_rng(SEED), []
    for start in range(0, REPS, 100):
        ids = rng.integers(len(a), size=(min(100, REPS-start), len(a)))
        divisor = den[ids].sum(1)
        if a.ndim == 2:
            divisor = divisor[:, None]
        draws.append(a[ids].sum(1) / divisor)
    interval = np.quantile(np.concatenate(draws), [.025, .975], axis=0)
    if a.ndim == 2:
        interval = interval.T
    return {"estimate": (a.sum(0)/den.sum()).tolist(), "ci95": interval.tolist(),
        "n_prompts": len(a), "denominator": float(den.sum()), "bootstrap_seed": SEED,
        "bootstrap_repetitions": REPS}


def verify_metric(summary, rows, key, denominator=None, *, paired_base=None):
    selected = [r for r in rows if r.get("status") == "ok" and r.get(key) is not None]
    if paired_base is not None:
        a, b = keyed(rows), keyed(paired_base)
        require(set(a) == set(b), "Paired metric planned UID sets differ")
        selected = [a[u] for u in sorted(a) if a[u].get("status") == b[u].get("status") == "ok"
                    and a[u].get(key) is not None and b[u].get(key) is not None]
        if denominator:
            require(all(r[denominator] == b[r["uid"]][denominator] for r in selected), "Paired denominators differ")
        values = [r[key]-b[r["uid"]][key] for r in selected]
    else:
        values = [r[key] for r in selected]
    result = bootstrap(values, [r[denominator] for r in selected] if denominator else None)
    for name in ("estimate", "ci95"):
        close(summary[name], result[name], f"{key}/{name}")
    require(summary["n_planned"] == len(rows), "Summary planned count differs from item coverage")
    nkey, failkey = ("n_paired", "n_unpaired") if paired_base is not None else ("n_scored", "n_failed")
    require(summary[nkey] == len(selected) and summary[failkey] == len(rows)-len(selected), "Summary success/failure count disagreement")
    require(summary["n_prompts"] == len(selected), "Summary bootstrap observation count disagreement")
    if paired_base is None:
        require(summary["complete"] == (len(selected) == len(rows)), "Summary completeness flag disagreement")
    if len(selected):
        require(summary.get("bootstrap_seed") == SEED and summary.get("bootstrap_repetitions") == REPS,
                "Unexpected bootstrap protocol")
        close(summary.get("denominator"), result["denominator"], "ratio denominator")
    return {**summary, "independently_reaggregated": True}


def diagnostics(rows):
    rows = list(rows)
    lengths = [r["n_output_tokens"] for r in rows]
    return {"n_responses": len(rows), "median_response_tokens": float(np.median(lengths)) if lengths else None,
        "mean_response_tokens": float(np.mean(lengths)) if lengths else None,
        "max_length_hits": sum(r.get("finish_reason") == "length" for r in rows),
        "empty_text": sum(not r["output"].strip() for r in rows),
        "keyword_refusal_count_diagnostic_only": sum(bool(KEYWORD_REFUSAL.search(r["output"])) for r in rows)}


def check_diagnostics(saved, observed):
    for key, value in observed.items():
        require(key in saved, f"Missing response diagnostic: {key}")
        close(saved[key], value, f"diagnostic/{key}")


class Collector:
    def __init__(self, root):
        self.root = Path(root)
        self.ledger, self.missing, self.errors = {}, [], []
        self.raw, self.expected, self.provenance, self.items = {}, {}, {}, {}
        self.report = {"schema": "nbpo_primary_results_v1", "labels": list(LABELS), "benchmarks": {},
            "diagnostics_only": {}, "provenance": self.ledger, "missing": self.missing, "errors": self.errors,
            "checkpoint_selection": "Both final1750 checkpoints fixed prospectively; base is common untrained reference",
            "uncertainty": "Fixed-seed paired prompt bootstrap, not training-seed/teacher-estimation uncertainty",
            "training_comparison_reference": "Separate compare_training_arms.py report; not silently substituted here"}

    def path(self, relative):
        path = Path(relative)
        return path if path.is_absolute() else self.root / path

    def consume(self, relative, expected=None):
        path = self.path(relative)
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = file_hash(path)
        require(str(path) not in self.ledger or self.ledger[str(path)]["sha256"] == actual,
                f"Source changed during collection: {path}")
        if expected is not None:
            require(actual == expected, f"Source hash mismatch: {path}")
        self.ledger[str(path)] = {"sha256": actual, "bytes": path.stat().st_size}
        return path

    def json(self, relative, expected=None):
        return json.loads(self.consume(relative, expected).read_text())

    def jsonl(self, relative, expected=None):
        # Physical JSONL newlines only: str.splitlines() also splits valid
        # Unicode U+2028/U+2029 inside a saved response string.
        with self.consume(relative, expected).open() as stream:
            return [json.loads(line) for line in stream if line.strip()]

    def attempt(self, name, function):
        try:
            return function()
        except FileNotFoundError as exc:
            self.missing.append({"section": name, "source": str(exc), "reason": "required source absent"})
        except (Invalid, KeyError, TypeError, ValueError) as exc:
            self.errors.append({"section": name, "reason": f"{type(exc).__name__}: {exc}"})
        return None

    def incomplete(self, condition, name):
        if not condition:
            self.missing.append({"section": name, "reason": "incomplete scored/generated item coverage; never zero-filled"})

    def load_raw(self, label, bench):
        from scripts.experiments.nbpo_repair_20260909.generate_eval import items_for
        files = {"ifeval": ["data/ifeval.jsonl"], "gsm8k": ["data/gsm8k_test.jsonl"],
            "harmbench": ["data/harmbench_behaviors.jsonl"], "alpaca_eval": ["data/alpaca_eval.jsonl"],
            "arena_hard": ["data/arena_hard.jsonl"], "xstest": ["data/xstest_prompts.csv"],
            "saferlhf": ["splits/dev.jsonl", "splits/test.jsonl"]}[bench]
        for path in files:
            self.consume(path)
        planned = items_for(bench, self.root)
        require(len(planned) == COUNTS[bench] and len({u for u, _, _ in planned}) == len(planned), "Unexpected planned benchmark panel")
        expected = {u: (p, meta) for u, p, meta in planned}
        folder = f"responses/{label}"
        manifest_path = f"{folder}/{bench}.manifest.json"
        manifest = self.json(manifest_path)
        settings = self.json(f"{folder}/settings.json", manifest["settings_sha256"])
        rows = keyed(self.jsonl(f"{folder}/{bench}.jsonl", manifest["response_file_sha256"]))
        require(not set(rows)-set(expected), "Unexpected saved response UID")
        require(manifest["prompt_sha256"] == object_hash([(u, p) for u, p, _ in planned]), "Planned prompt hash mismatch")
        require(manifest["n_planned"] == len(planned) and manifest["n_generated"] == len(rows)
                and manifest["n_failed"] == len(planned)-len(rows), "Response manifest counts mismatch")
        for uid, row in rows.items():
            require(row["prompt"] == expected[uid][0] and row["prompt_sha256"] == digest(row["prompt"]), "Response prompt binding mismatch")
            require(row["model_sha256"] == object_hash(settings["model_weights"]), "Response model identity mismatch")
            require(len(row["output_token_ids"]) == row["n_output_tokens"] > 0, "Raw response token count mismatch")
            require(row["finish_reason"] in ("stop", "length"), "Unknown response finish reason")
            if row["finish_reason"] == "stop":
                require(row["output_token_ids"][-1] in settings["terminal_ids"], "Stopped response lacks raw terminal token")
            else:
                require(row["n_output_tokens"] == settings["max_new_tokens"], "Length-capped response disagrees with budget")
        source = {"responses_sha256": manifest["response_file_sha256"],
            "generation_manifest_sha256": self.ledger[str(self.path(manifest_path))]["sha256"],
            "generation_settings": settings, "n_planned": len(planned), "n_generated": len(rows)}
        self.raw[label, bench], self.expected[bench], self.provenance[label, bench] = rows, expected, source
        self.report["diagnostics_only"].setdefault(label, {})[bench] = diagnostics(rows.values())
        self.incomplete(len(rows) == len(planned), f"raw:{label}:{bench}")
        return source

    def bind_summary(self, summary, label, bench, nested=None):
        reference = self.provenance.get((label, bench))
        if reference is None:
            raise FileNotFoundError(f"verified responses for {label}/{bench}")
        source = summary[nested] if nested else summary
        require(all(source[k] == v for k, v in reference.items()), f"Summary response provenance mismatch: {label}/{bench}")

    def scored_rows(self, path, label, bench):
        if (label, bench) not in self.raw:
            raise FileNotFoundError(f"verified saved responses for {label}/{bench}")
        rows = self.jsonl(path)
        expected = self.expected.get(bench)
        if expected is None:
            raise FileNotFoundError(f"verified prompt plan for {bench}")
        require(set(keyed(rows)) == set(expected), "Scored per-item panel differs from planned IDs")
        raw = self.raw.get((label, bench), {})
        for row in rows:
            if row["status"] == "ok":
                require(row["uid"] in raw, "Scored row has no saved response")
        return rows

    def scalar_benchmark(self, label, bench):
        directory = "harmbench_primary_v1" if bench == "harmbench" else "deterministic_primary_v1"
        stem = f"evaluations/{directory}/{bench}_{label}"
        summary, rows = self.json(stem+".summary.json"), self.scored_rows(stem+".per_item.jsonl", label, bench)
        self.bind_summary(summary, label, bench)
        check_diagnostics(summary["response_diagnostics"], diagnostics(self.raw[label, bench].values()))
        self.items[label, bench] = rows
        for row in rows:
            if row["status"] != "ok":
                continue
            if bench == "gsm8k":
                require(row["correct"] in (0, 1) and row["parse_failure"] in (0, 1), "Nonbinary GSM metric")
                require(not row["parse_failure"] or row["correct"] == 0, "GSM parse failures must count incorrect, not disappear")
                from scripts.experiments.nbpo_downstream_local_v1.score_gsm8k import SPEC, norm
                gold = norm(self.expected[bench][row["uid"]][1]["gold"].split("####")[-1].strip())
                matches = SPEC.findall(self.raw[label, bench][row["uid"]]["output"])
                pred = norm(matches[-1]) if matches else None
                require(row["gold"] == gold and row["prediction"] == pred
                        and row["correct"] == int(pred is not None and pred == gold)
                        and row["parse_failure"] == int(pred is None), "GSM raw-response parser/EM mismatch")
            elif bench == "ifeval":
                count = row["n_instructions"]
                require(type(count) is int and count > 0, "Invalid IFEval instruction count")
                require(count == len(self.expected[bench][row["uid"]][1]["instruction_id_list"]), "IFEval source instruction count mismatch")
                for kind in ("strict", "loose"):
                    flags = row[f"{kind}_instruction_flags"]
                    require(len(flags) == count and sum(flags) == row[f"{kind}_instructions_correct"]
                            and int(all(flags)) == row[f"{kind}_prompt"], "IFEval flags/aggregate mismatch")
            else:
                require(row["harmful"] in (0, 1), "Nonbinary HarmBench label")
                metadata = self.expected[bench][row["uid"]][1]
                require(row["functional_category"] == metadata["functional_category"]
                        and row["semantic_category"] == metadata["semantic_category"], "HarmBench source category mismatch")
                require(row["generation_sha256"] == digest(self.raw[label,bench][row["uid"]]["output"])
                        and 0 <= row["evaluated_tokens"] <= 512, "HarmBench evaluated response binding mismatch")
        metrics = {}
        for name, (key, den) in DEFINITIONS[bench].items():
            saved = summary[name] if bench == "harmbench" else summary["metrics"][name]
            metrics[name] = verify_metric(saved, rows, key, den)
            self.incomplete(saved["n_failed"] == 0, f"{label}:{bench}:{name}")
        result = {"metrics": metrics, "summary_source": stem+".summary.json", "status_counts": dict(Counter(r["status"] for r in rows))}
        if label != "base":
            base_rows = self.items.get(("base", bench))
            if base_rows is None:
                raise FileNotFoundError(f"verified base per-item scores for {bench}")
            saved_pairs = summary["paired_delta_vs_base"]
            if bench == "harmbench":
                result["paired_delta_vs_base"] = verify_metric(saved_pairs, rows, "harmful", paired_base=base_rows)
            else:
                result["paired_delta_vs_base"] = {name: verify_metric(saved_pairs[name], rows, key, den,
                    paired_base=base_rows) for name, (key, den) in DEFINITIONS[bench].items()}
        if bench == "harmbench":
            result["categories"] = {}
            for field, groups in summary["categories"].items():
                names = {r[field] for r in rows}
                require(set(groups) == names, "HarmBench category inventory mismatch")
                result["categories"][field] = {name: verify_metric(groups[name], [r for r in rows if r[field] == name], "harmful") for name in sorted(names)}
            require(summary["invalid_labels"] == sum(r["status"] == "classifier_invalid" for r in rows), "Invalid label count mismatch")
            result["evaluator"] = {k: summary[k] for k in ("official_repo_commit", "classifier_weights_sha256",
                "classifier_tokenizer_files_sha256", "generation_truncation")}
        return result

    def skywork(self, label, bench):
        directory = "skywork_base_v1" if label == "base" else f"skywork_{label}_v1"
        stem = f"evaluations/{directory}/{bench}_{label}"
        summary, rows = self.json(stem+".summary.json"), self.scored_rows(stem+".per_item.jsonl", label, bench)
        self.bind_summary(summary, label, bench)
        self.bind_summary(summary, "base", bench, "base_provenance")
        raw, base = self.raw[label, bench], self.raw["base", bench]
        check_diagnostics(summary["candidate_length"], diagnostics(raw.values()))
        check_diagnostics(summary["base_length"], diagnostics(base.values()))
        for row in rows:
            if row["status"] == "ok":
                a, b = row["candidate_score"], row["base_score"]
                require(math.isfinite(a) and math.isfinite(b), "Nonfinite successful Skywork score")
                require(row["tie"] == (a == b) and row["win"] == (0.5 if a == b else float(a > b)), "Skywork exact-tie/win rule mismatch")
                if label == "base":
                    require(a == b and row["win"] == .5, "Base self-comparison must be a measured exact tie")
                require(row["candidate_tokens"] == raw[row["uid"]]["n_output_tokens"]
                        and row["base_tokens"] == base[row["uid"]]["n_output_tokens"], "Proxy pair response lengths differ")
        from scripts.experiments.nbpo_repair_20260909.evaluate_responses import arena_subset
        group_names = {"all805": 805} if bench == "alpaca_eval" else {"hard500": 500, "creative250": 250}
        subsets = {}
        for group, count in group_names.items():
            items = rows if bench == "alpaca_eval" else [r for r in rows if arena_subset(self.expected[bench][r["uid"]][1]) == group]
            require(len(items) == count, "Incorrect Skywork subset size")
            saved = summary["subsets"][group]
            metric = verify_metric(saved, items, "win")
            valid = [r for r in items if r["status"] == "ok"]
            for key, value in {"wins": sum(r["win"] == 1 for r in valid), "ties": sum(r["tie"] for r in valid),
                               "losses": sum(r["win"] == 0 for r in valid)}.items():
                require(saved[key] == value, "Skywork win/tie/loss counts mismatch")
            failures = dict(Counter(reason for r in items for reason in (r.get("base_failure"), r.get("candidate_failure")) if reason))
            require(saved["failure_reasons"] == failures, "Skywork failure reason mismatch")
            ids = {r["uid"] for r in items}
            check_diagnostics(saved["candidate_diagnostics"], diagnostics(r for uid, r in raw.items() if uid in ids))
            check_diagnostics(saved["base_diagnostics"], diagnostics(r for uid, r in base.items() if uid in ids))
            subsets[group] = metric
            self.incomplete(saved["n_failed"] == 0, f"{label}:{bench}:{group}")
        return {"subsets": subsets, "metric": summary["metric"], "not_official": summary["not_official"],
            "tie_rule": summary["tie_rule"], "rm_weights_sha256": summary["rm_weights_sha256"],
            "rm_tokenizer_sha256": summary["rm_tokenizer_sha256"], "summary_source": stem+".summary.json"}

    def xstest(self, label):
        stem = f"evaluations/xstest_primary_unadjudicated_v1/{label}"
        summary, rows = self.json(stem+".summary.json"), self.scored_rows(stem+".per_item.jsonl", label, "xstest")
        self.bind_summary(summary, label, "xstest", "source")
        require(summary["method"] == "not_adjudicated" and summary["model"] is None, "Frozen XSTest plan is explicitly unadjudicated")
        require(summary["appropriate_handling"] is None, "Unadjudicated appropriateness cannot become a score")
        raw = self.raw[label, "xstest"]
        for row in rows:
            require(row["status"] in ("not_adjudicated", "missing_response", "empty_response"), "Unexpected XSTest adjudication")
            require(all(row.get(k) is None for k in ("response_refusal", "prompt_harmfulness", "response_harmfulness", "appropriate_handling")), "Missing XSTest labels must remain null")
            require(row["source_label"] == self.expected["xstest"][row["uid"]][1]["label"], "XSTest source stratum mismatch")
            if row["uid"] in raw:
                require(row["response"] == raw[row["uid"]]["output"] and row["response_sha256"] == digest(row["response"]), "XSTest raw response binding mismatch")
        for group, count in (("all", 450), ("safe", 250), ("unsafe", 200)):
            selected = [r for r in rows if group == "all" or r["source_label"] == group]
            require(len(selected) == count and summary["results"][group]["n_planned"] == count, "XSTest planned stratum count mismatch")
            require(summary["results"][group]["status_counts"] == dict(Counter(r["status"] for r in selected)), "XSTest status counts mismatch")
            for metric in summary["results"][group]["metrics"].values():
                require(metric["estimate"] is None and metric["ci95"] is None and metric["n_scored"] == 0,
                        "XSTest null metric changed")
        return {"status": "UNADJUDICATED", "results": summary["results"], "appropriate_handling": None,
            "reason": summary["unadjudicated_reason"], "summary_source": stem+".summary.json"}

    def original_disagreement(self, fixed):
        names = ("source_artifacts", "comparator_occurrences_sha256", "reference_ensemble_probabilities_sha256", "prompt_order_sha256", "teacher")
        require(object_hash({k: fixed[k] for k in names}) == fixed["sha256"], "Fixed-game identity hash mismatch")
        require(fixed["beta"] == [.25, .25] and fixed["comparator_occurrence_mass"] == [.125]*8,
                "Changed finite-game regularization or comparator mass")
        result = {}
        for record in fixed["source_artifacts"]:
            shard = record["shard"]
            manifest = self.json(f"scores/shard{shard}/manifest.json", record["score_manifest_sha256"])
            require(manifest.get("objectives") == ["helpfulness","harmlessness"] and manifest.get("seeds") == [41,42,43], "Original reference objective/seed order differs")
            path = self.consume(f"scores/shard{shard}/probabilities.npz", record["probabilities_sha256"])
            require(manifest["probabilities_sha256"] == record["probabilities_sha256"], "Reference array provenance mismatch")
            with np.load(path) as arrays:
                ref = arrays["reference"]
            require(ref.shape == (3, 2, len(manifest["prompt_ids"]), 8, 8), "Reference tensor shape mismatch")
            margins = (ref.mean(0)-.5).mean(2)
            values = -.25 * np.log(np.exp(-margins/.25).mean(-1))
            for index, pid in enumerate(manifest["prompt_ids"]):
                require(pid not in result, "Duplicate original reference prompt")
                result[pid] = values[:, index]
        return result

    def saferlhf(self, label):
        directory = "evaluations/saferlhf_primary_v1"
        folder = f"{directory}/{label}"
        summary, manifest = self.json(folder+"/report.json"), self.json(folder+"/scoring_manifest.json")
        records = self.scored_rows(folder+"/per_prompt.jsonl", label, "saferlhf")
        self.bind_summary(summary, label, "saferlhf", "response_provenance")
        self.bind_summary(manifest, label, "saferlhf", "response_provenance")
        fixed = self.json(directory+"/fixed_game.json")
        require(summary["fixed_game_sha256"] == manifest["fixed_game_sha256"] == fixed["sha256"], "SafeRLHF fixed game differs")
        require(summary["objectives"] == manifest["objectives"] == ["helpfulness", "harmlessness"]
                and manifest["seeds"] == [41, 42, 43], "SafeRLHF axis/seed order differs")
        reference = self.original_disagreement(fixed)
        def probabilities(name):
            m = self.json(f"{directory}/{name}/scoring_manifest.json")
            require(m["prompt_ids"] == manifest["prompt_ids"] and m["fixed_game_sha256"] == fixed["sha256"], "Greedy base probability alignment differs")
            self.bind_summary(m,name,"saferlhf","response_provenance")
            cached=m.get("cached_probability_source")
            if cached is not None:
                fixed_cache=self.root/"evaluations/saferlhf_base_v1/base"
                require(name=="base" and Path(cached["source_directory"])==fixed_cache,"Unexpected cached base probability source")
                self.json(fixed_cache/"scoring_manifest.json",cached["scoring_manifest_sha256"])
                self.consume(fixed_cache/"probabilities.npz",cached["probabilities_sha256"])
            path = self.consume(f"{directory}/{name}/probabilities.npz", m["probabilities_sha256"])
            with np.load(path) as arrays:
                p = arrays["probabilities"]
            require(p.shape == (3, 2, 1500, 8), "SafeRLHF probability tensor shape mismatch")
            valid = np.isfinite(p).all(axis=(0, 1, 3))
            require(((p[:, :, valid] >= 0) & (p[:, :, valid] <= 1)).all(), "Out-of-range calibrated probabilities")
            values = np.full((1500, 2), np.nan)
            values[valid] = (-.25*np.log(np.exp(-(p[:, :, valid].mean(0)-.5)/.25).mean(-1))).T
            return valid, values
        valid, values = probabilities(label)
        base_valid, base_values = probabilities("base")
        index = keyed(records)
        ordered = [index["saferlhf:"+str(pid)] for pid in manifest["prompt_ids"]]
        require(len(ordered) == 1500 and len(set(manifest["prompt_ids"])) == 1500, "SafeRLHF prompt inventory differs")
        for x, row in enumerate(ordered):
            uid = row["uid"]
            require(row["split"] == self.expected["saferlhf"][uid][1]["split"], "SafeRLHF per-item split mismatch")
            close(row["d_original_reference"], reference[row["prompt_id"]], "original independent-reference d")
            require((row["status"] == "ok") == bool(valid[x]), "SafeRLHF finite probabilities/status mismatch")
            if valid[x]:
                close(row["V"], values[x], "SafeRLHF V from calibrated probability tensor")
                close(row["s_original_reference"], values[x]-reference[row["prompt_id"]], "SafeRLHF s=V-d")
            if valid[x] and base_valid[x]:
                close(row["V_base_greedy"], base_values[x], "SafeRLHF greedy base V")
                close(row["delta_vs_base_greedy"], values[x]-base_values[x], "SafeRLHF candidate-minus-greedy-base")
        for split, count in (("dev", 500), ("test", 1000)):
            selected = [r for r in ordered if r["split"] == split]
            scored = [r for r in selected if r["status"] == "ok"]
            paired = [r for r in scored if "V_base_greedy" in r]
            saved = summary["splits"][split]
            generated = sum(r["uid"] in self.raw[label,"saferlhf"] for r in selected)
            require(len(selected) == count and saved["n_planned"] == count
                    and saved["n_scored"] == len(scored) and saved["n_failed"] == count-len(scored)
                    and saved["n_generated"] == generated and saved["n_paired_vs_base"] == len(paired)
                    and saved["complete"] == (len(scored) == count), "SafeRLHF split coverage mismatch")
            for name, subset in (("original_reference_comparison", scored), ("paired_greedy_base_comparison", paired)):
                source = saved[name]
                require(source["n_prompts"] == len(subset), "SafeRLHF vector denominator mismatch")
                if subset:
                    require(source["bootstrap_seed"] == SEED and source["bootstrap_repetitions"] == REPS, "SafeRLHF CI protocol differs")
                for key, metric in source["metrics"].items():
                    calculated = bootstrap([r[key] for r in subset])
                    close(metric["estimate"], calculated["estimate"], f"SafeRLHF {split}/{name}/{key} estimate")
                    close(metric["ci95"], calculated["ci95"], f"SafeRLHF {split}/{name}/{key} CI")
            calculated = bootstrap([r["d_original_reference"] for r in selected])
            for field in ("estimate", "ci95"):
                close(saved["original_disagreement_all_planned"][field], calculated[field], "all-planned original d")
            require(saved["acceptance_estimate"]["algorithm1_accepted_next_stage_policy"] is False
                    and saved["fresh_test_claim"] is False, "Greedy evaluation cannot certify Algorithm1/fresh test")
            self.incomplete(len(scored) == count and len(paired) == count, f"{label}:saferlhf:{split}")
        truncations = self.jsonl(directory+"/teacher_truncation.jsonl")
        selected = [r for r in truncations if r["label"] == label]
        trunc = summary["teacher_truncation"]
        require(trunc["n_evaluated"] == len(selected), "Teacher truncation count mismatch")
        for field in ("prompt_truncated", "response_truncated"):
            close(trunc[field+"_fraction"], float(np.mean([r[field] for r in selected])) if selected else None, "teacher truncation fraction")
        diag = diagnostics(self.raw[label, "saferlhf"].values())
        for source_name, dest in (("mean_tokens", "mean_response_tokens"), ("median_tokens", "median_response_tokens"), ("max_length_hits", "max_length_hits")):
            close(summary["response_length"][source_name], diag[dest], "SafeRLHF response length")
        return {"splits": summary["splits"], "objectives": summary["objectives"],
            "fixed_game_sha256": fixed["sha256"], "teacher_truncation": trunc,
            "reference_interpretation": summary["reference_interpretation"], "scope": summary["scope"],
            "test_exposure": summary["test_exposure"], "fresh_test_claim": False,
            "summary_source": folder+"/report.json", "independently_recomputed_probability_values": True}

    def teacher_rm(self):
        directory = "teacher_rm/nash_repair_v2_v1"
        summary = self.json(directory+"/aggregate/report.json")
        rows = self.jsonl(directory+"/aggregate/per_prompt.jsonl", summary["per_prompt_sha256"])
        require(object_hash(summary["protocol"]) == summary["protocol_sha256"], "Teacher-RM protocol mismatch")
        train = self.jsonl("splits/train.jsonl")
        require(set(keyed(rows, "prompt_id")) == {r["prompt_id"] for r in train} and len(rows) == 2000,
                "Teacher-RM prompt inventory mismatch")
        n_candidates, successful, failures, candidates = 0, 0, Counter(), {}
        for shard in range(4):
            manifest = self.json(f"{directory}/shard{shard}/manifest.json", summary["shard_manifest_sha256"][str(shard)])
            require(manifest["protocol_sha256"] == summary["protocol_sha256"], "Teacher-RM shard protocol differs")
            settings=self.json(f"{directory}/shard{shard}/settings.json",manifest["settings_sha256"])
            require(settings["protocol_sha256"]==summary["protocol_sha256"] and object_hash(settings["protocol"])==summary["protocol_sha256"],"Teacher-RM shard settings differ")
            for name, expected in manifest["chunks_sha256"].items():
                require(Path(name).name == name and name.startswith("chunk") and name.endswith(".json"), "Invalid chunk path")
                chunk = self.json(f"{directory}/shard{shard}/{name}", expected)
                require(object_hash(chunk["records"]) == chunk["records_sha256"] and chunk["protocol_sha256"]==summary["protocol_sha256"], "Teacher-RM chunk records/protocol hash differs")
                for record in chunk["records"]:
                    key=(record["prompt_id"],record["sample_index"])
                    require(key not in candidates,"Duplicate teacher-RM candidate score")
                    candidates[key]=record
                n_candidates += len(chunk["records"])
                successful += sum(r["status"] == "ok" for r in chunk["records"])
                failures.update(r.get("failure_reason", "unspecified") for r in chunk["records"] if r["status"] != "ok")
        valid = [r for r in rows if r["status"] == "ok"]
        require(set(candidates)=={(r["prompt_id"],i) for r in train for i in range(8)},"Teacher-RM candidate occurrence inventory differs")
        for row in rows:
            group=[candidates[row["prompt_id"],i] for i in range(8)]
            require(row["n_candidates_present"]==8 and row["n_candidates_ok"]==sum(r["status"]=="ok" for r in group),"Teacher-RM per-prompt candidate coverage differs")
        require(summary["n_planned_prompts"] == 2000 and summary["n_planned_candidates"] == 16000
                and summary["n_present_candidates"] == n_candidates == sum(r["n_candidates_present"] for r in rows)
                and successful == sum(r["n_candidates_ok"] for r in rows)
                and summary["n_complete_prompts"] == len(valid)
                and summary["n_incomplete_prompts"] == 2000-len(valid), "Teacher-RM coverage mismatch")
        require(summary["failure_reasons"] == dict(failures), "Teacher-RM failure counts differ")
        for key, metric in summary["metrics"].items():
            values = [r[key] for r in valid if r[key] is not None]
            computed = bootstrap(values)
            close(metric["estimate"], computed["estimate"], f"teacher-RM {key} estimate")
            close(metric["ci95"], computed["ci95"], f"teacher-RM {key} CI")
            require(metric["n_prompts"] == len(values) and metric["n_undefined_among_complete_prompts"] == len(valid)-len(values), "Undefined teacher-RM metrics must not become zero")
        self.incomplete(len(valid) == 2000 and n_candidates == 16000, "teacher_rm")
        return {**summary, "independently_reaggregated_from_per_prompt": True,
                "summary_source": directory+"/aggregate/report.json"}

    def controller_contract(self):
        from scripts.experiments.nbpo_repair_20260909.evaluation_chain import plan
        saved = self.json("controllers/evaluation_chain_v1/plan.json")
        require(saved == plan(self.root), "Evaluation controller plan differs from fixed collection paths")
        source = self.json("controllers/evaluation_chain_v1/source_contract.json")
        for name, expected in source.items():
            self.consume("code/"+name, expected)
        primary = self.json("controllers/primary_chain_v1/complete.json")
        finished = self.json("controllers/evaluation_chain_v1/complete.json")
        require(finished["labels"] == list(LABELS[1:]), "Completed evaluation identity mismatch")
        require(len(primary["arms"])==2 and {r["arm"] for r in primary["arms"]}=={"wbc","mse"},"Expected exactly the two fixed primary exports")
        inventory=self.json("provenance/inventory.json")
        base_weights={Path(r["path"]).name:r["sha256"] for r in inventory["assets"]
            if Path(r["path"]).parent.name=="Llama-3.1-8B-Instruct" and r["path"].endswith(".safetensors")}
        require(bool(base_weights),"Captured base weight inventory missing")
        for bench in COUNTS:
            if ("base",bench) in self.provenance:
                require(self.provenance["base",bench]["generation_settings"]["model_weights"]==base_weights,"Base evaluation checkpoint differs from pinned inventory")
        for record in primary["arms"]:
            require(record["global_step"] == 1750, "Wrong controller checkpoint horizon")
            label = record["arm"]+"_primary_v1"
            require(label in LABELS[1:], "Unplanned controller arm")
            validation = self.json(f"probes/export_validation_{record['arm']}_v1/validation.json")
            require(validation["passed"] and validation["stage"] == "full" and validation["gpu_forward_executed"], "Final export did not pass full validation")
            require(validation["controller"]["weight_sha256"] == record["weight_sha256"], "Export probe/controller weight disagreement")
            require(validation["pinned_base"]["inventory_sha256"]==self.ledger[str(self.path("provenance/inventory.json"))]["sha256"],"Export probe used a different captured base inventory")
            for bench in COUNTS:
                if (label, bench) in self.provenance:
                    require(self.provenance[label, bench]["generation_settings"]["model_weights"] == record["weight_sha256"], "Evaluation did not use fixed final1750 weights")
        specs = list(saved["validation"])+list(saved["generation"])
        specs += [entry for lane in saved["scoring_lanes"].values() for entry in lane]
        specs.append(saved["teacher_aggregate"])
        for spec in specs:
            directory = f"jobs/{spec['name']}"
            job = self.json(directory+"/spec.json")
            exit_record = self.json(directory+"/exit.json")
            require(job["command"] == spec["command"] and exit_record["exit_code"] == 0, "Fixed evaluation job command/exit mismatch")
            self.consume(directory+"/source.zip", job["source_archive_sha256"])
            self.consume(directory+"/stdout.log", exit_record["log_sha256"])
        return {"fixed_plan_verified": True, "completed_labels": finished["labels"],
                "checkpoint_selection": saved["checkpoint_selection"]}

    def collect(self):
        for label in LABELS:
            for bench in COUNTS:
                self.attempt(f"raw:{label}:{bench}", lambda l=label,b=bench:self.load_raw(l,b))
        for label in LABELS[1:]:
            for bench in COUNTS:
                if (label, bench) in self.provenance and ("base", bench) in self.provenance:
                    self.attempt(f"compatibility:{label}:{bench}", lambda l=label,b=bench:
                        assert_matched_protocol(self.provenance[l,b]["generation_settings"], self.provenance["base",b]["generation_settings"]))
        self.report["controller"] = self.attempt("controller_contract", self.controller_contract)
        for bench in COUNTS:
            self.report["benchmarks"][bench] = {}
            for label in LABELS:
                function = (lambda l=label,b=bench:self.scalar_benchmark(l,b)) if bench in DEFINITIONS else (
                    (lambda l=label,b=bench:self.skywork(l,b)) if bench in ("alpaca_eval", "arena_hard") else
                    (lambda l=label:self.saferlhf(l)) if bench == "saferlhf" else (lambda l=label:self.xstest(l)))
                self.report["benchmarks"][bench][label] = self.attempt(f"scores:{label}:{bench}", function)
        self.report["teacher_rm"] = self.attempt("teacher_rm", self.teacher_rm)
        for bench in ("alpaca_eval", "arena_hard", "harmbench", "saferlhf"):
            available = [self.report["benchmarks"][bench][l] for l in LABELS if self.report["benchmarks"][bench][l] is not None]
            if available:
                key = "evaluator" if bench == "harmbench" else "fixed_game_sha256" if bench == "saferlhf" else "rm_weights_sha256"
                self.attempt(f"common_evaluator:{bench}", lambda entries=available,k=key:
                    require(all(r[k] == entries[0][k] for r in entries), "Evaluator identity differs across methods"))
        self.report["status"] = "INVALID" if self.errors else "INCOMPLETE" if self.missing else "COMPLETE"
        self.report["data_complete"] = self.report["status"] == "COMPLETE"
        self.report["source_sha256"] = file_hash(__file__)
        return self.report


def metric_counts(bench, value, path, metric):
    if bench == "saferlhf" and len(path)>2 and path[0] == "splits":
        split = value["splits"][path[1]]
        n = split["n_paired_vs_base"] if "paired_greedy_base_comparison" in path else (
            split["n_planned"] if "original_disagreement_all_planned" in path else split["n_scored"])
        return n, split["n_planned"]-n
    return metric.get("n_scored",metric.get("n_paired",metric.get("n_prompts"))), metric.get("n_failed",metric.get("n_unpaired",0))


def point(metric, *, percent=True, interval=True):
    if metric is None or metric.get("estimate") is None:
        return "not available"
    factor = 100 if percent else 1
    value = metric["estimate"] * factor
    if not interval:
        return f"{value:.2f}"
    lo, hi = metric["ci95"]
    return f"{value:.2f} [{lo*factor:.2f}, {hi*factor:.2f}]" if percent else f"{value:.5f} [{lo:.5f}, {hi:.5f}]"


def main_cells(report, label, panel):
    data = report["benchmarks"]
    def metric(bench, path):
        value = data[bench].get(label)
        for key in path:
            if value is None:
                return None
            value = value[key]
        return value
    if panel == "A":
        return [metric("ifeval", ("metrics", "strict_prompt_accuracy")),
            metric("ifeval", ("metrics", "strict_instruction_accuracy")),
            metric("gsm8k", ("metrics", "exact_match")), metric("harmbench", ("metrics", "overall"))]
    return [metric("alpaca_eval", ("subsets", "all805")), metric("arena_hard", ("subsets", "hard500")),
            metric("arena_hard", ("subsets", "creative250"))]


def metric_leaves(value, path=()):
    if isinstance(value, dict):
        if "estimate" in value and "ci95" in value:
            yield path, value
        else:
            for key, child in value.items():
                if key not in ("protocol", "provenance", "source", "controller"):
                    yield from metric_leaves(child, (*path, key))


def markdown(report):
    lines = ["# Primary benchmark report", "", f"Status: **{report['status']}**.", "",
        "Only base, fixed1750 WBC, and fixed1750 MSE are included. Missing is not zero. Brackets are95% prompt-bootstrap CIs; one training seed does not support training-seed significance claims.", "",
        "## Panel A: task accuracy and harmful-completion rate", "",
        "| Method | IF strict prompt (541) ↑ | IF strict instruction (834) ↑ | GSM8K EM (1319) ↑ | HarmBench (320) ↓ |",
        "|---|---:|---:|---:|---:|"]
    for label in LABELS:
        lines.append("| "+label+" | "+" | ".join(point(m) for m in main_cells(report,label,"A"))+" |")
    lines += ["", "GSM8K uses the frozen specified-answer parser; parse failures remain incorrect in EM. HarmBench uses official classifier/hash routes, not a refusal detector.", "",
        "## Panel B: local Skywork proxy win rates versus common base", "",
        "| Method | Alpaca805 ↑ | Hard500 ↑ | Creative250 ↑ |", "|---|---:|---:|---:|"]
    for label in LABELS:
        lines.append("| "+label+" | "+" | ".join(point(m) for m in main_cells(report,label,"B"))+" |")
    lines += ["", "Exact scalar ties count0.5; base-versus-itself is a measured tie check. These are NOT official AlpacaEval LC or Arena-Hard judge scores.", "",
        "## Full verified metrics, paired CIs and denominators", ""]
    for bench, arms in report["benchmarks"].items():
        lines += [f"### {bench}", ""]
        for label, value in arms.items():
            if value is None:
                lines += [f"{label}: not available.", ""]
                continue
            if bench == "xstest":
                lines += [f"{label}: UNADJUDICATED; refusal and appropriateness remain null (all450, safe250, unsafe200).", ""]
                continue
            if bench == "saferlhf":
                lines += [f"{label} split coverage: "+"; ".join(f"{s}: planned={v['n_planned']}, generated={v['n_generated']}, scored={v['n_scored']}, failed={v['n_failed']}, paired={v['n_paired_vs_base']}" for s,v in value["splits"].items()),
                    "", "Teacher input truncation (not policy response length): "+json.dumps(value["teacher_truncation"]), ""]
            lines += [f"{label}:", "", "| Metric | Estimate [95% CI] | n / failures |", "|---|---|---|"]
            for path, metric in metric_leaves(value):
                estimate = metric.get("estimate")
                n, failures = metric_counts(bench,value,path,metric)
                rendered = json.dumps({"estimate": estimate, "ci95": metric["ci95"]}) if isinstance(estimate, list) else point(metric, percent=bench != "saferlhf")
                lines.append(f"| {' / '.join(path)} | {rendered} | {n} / {failures} |")
            lines += [""]
    lines += ["## Length/cap/keyword diagnostics only", "", "| Method / benchmark | Mean tokens | Median | Length caps | Empty | Keyword hits |",
        "|---|---:|---:|---:|---:|---:|"]
    for label, benchmarks in report["diagnostics_only"].items():
        for bench, row in benchmarks.items():
            lines.append(f"| {label} / {bench} | {row['mean_response_tokens']} | {row['median_response_tokens']} | {row['max_length_hits']} | {row['empty_text']} | {row['keyword_refusal_count_diagnostic_only']} |")
    for bench in ("alpaca_eval","arena_hard"):
        for label,value in report["benchmarks"][bench].items():
            if value is not None:
                for subset,metric in value["subsets"].items():
                    row=metric["candidate_diagnostics"]
                    lines.append(f"| {label} / {subset} | {row['mean_response_tokens']} | {row['median_response_tokens']} | {row['max_length_hits']} | {row['empty_text']} | {row['keyword_refusal_count_diagnostic_only']} |")
    lines += ["", "Keyword hits are lexical diagnostics, not refusal labels. Length caps are generation limits, not early-EOS evidence. SafeRLHF V,d,s compare frozen finite-pool greedy point masses; they do not certify raw stochastic-policy acceptance, and the test split is not claimed globally untouched.", "",
        "## Finite-pool teacher / independent RM diagnostic", ""]
    teacher = report.get("teacher_rm")
    if teacher is None:
        lines += ["Not available.", ""]
    else:
        lines += [teacher["metric_scope"], "", "| Metric | Estimate [95% CI] | Defined prompts |", "|---|---|---:|"]
        for name, metric in teacher["metrics"].items():
            lines.append(f"| {name} | {point(metric, percent=False)} | {metric['n_prompts']} |")
        lines += [""]
    lines += ["## Missing or invalid evidence", ""]
    lines += [f"- {r['section']}: {r.get('source','')} {r['reason']}" for r in report["missing"]+report["errors"]] or ["None."]
    lines += ["", "Every consumed input is SHA256-bound in primary_results.json. No inferred or outcome-selected replacement paths are used. Training-precision/exposure comparisons remain in the independent training audit.", ""]
    return "\n".join(lines)


def tex_escape(text):
    return str(text).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")


def compact_metric_path(path):
    replacements={"original_reference_comparison":"original", "paired_greedy_base_comparison":"paired",
        "d_original_reference":"d", "s_original_reference":"s", "delta_vs_base_greedy":"delta V",
        "V_base_greedy":"base V", "original_disagreement_all_planned":"d (all)",
        "paired_delta_vs_base":"delta vs base", "functional_category":"functional",
        "semantic_category":"semantic", "misinformation_disinformation":"mis/disinformation",
        "cybercrime_intrusion":"cybercrime", "harassment_bullying":"harassment", "chemical_biological":"chemical/biological"}
    return "/".join(replacements.get(k,k) for k in path if k not in ("metrics","splits","categories"))


def table_tex(headers, rows):
    return "\n".join([r"\begin{center}\small", r"\begin{tabular}{"+"l"*len(headers)+"}", r"\toprule",
        " & ".join(headers)+r" \\", r"\midrule", *[" & ".join(row)+r" \\" for row in rows],
        r"\bottomrule", r"\end{tabular}", r"\end{center}", ""])


def paper_fragments(report):
    require(report["status"] == "COMPLETE" and report["data_complete"], "Incomplete evidence cannot produce paper fragments")
    require(all(report["benchmarks"][b].get(l) is not None for b in COUNTS for l in LABELS)
            and report.get("teacher_rm") is not None, "Complete fragments require every fixed source section")
    names = {"base": "Base", "wbc_primary_v1": "NBPO--WBC", "mse_primary_v1": "NBPO--MSE"}
    main = [r"% Generated only from complete provenance-checked fixed1750 results.",
        r"\begin{table}[t]\centering\small", r"\caption{Fixed-final-checkpoint evaluation (percent). Panel A: task metrics and HarmBench harmful-completion rate. Panel B: local Skywork proxy win rates against the same base, with exact ties worth $1/2$; not official AlpacaEval LC or Arena-Hard scores. Full paired confidence intervals, counts and diagnostics are in Appendix~\ref{sec:nbpo-primary-generated-details}.}",
        r"\label{tab:nbpo-primary-generated}", r"\textbf{A. Accuracy and harmful completion}"]
    main.append(table_tex(["Method", "IF strict P", "IF strict I", "GSM EM", r"HB $\downarrow$"],
        [[names[l]]+[point(m,interval=False) for m in main_cells(report,l,"A")] for l in LABELS]))
    main += [r"\textbf{B. Local scalar-RM proxy win rates}", table_tex(["Method", "Alpaca805", "Hard500", "Creative250"],
        [[names[l]]+[point(m,interval=False) for m in main_cells(report,l,"B")] for l in LABELS]),
        r"\par\footnotesize IF: 541 prompts/834 instructions; GSM8K: 1,319; HarmBench: 320. One fixed training seed. No checkpoint selection by these outcomes.", r"\end{table}", ""]
    appendix = [r"% Requires booktabs; generated independently of manuscript insertion.",
        r"\subsection{Primary evaluation details}\label{sec:nbpo-primary-generated-details}",
        r"Intervals are percentile 95\% bootstrap intervals from 2,000 prompt resamples (seed 20260909), conditional on the frozen pools and evaluators; they are not training-seed intervals. Differences are candidate minus greedy base. GSM parse failures remain incorrect. XSTest450 (safe250/unsafe200) is unadjudicated; no refusal or appropriateness rates are assigned."]
    for bench, arms in report["benchmarks"].items():
        if bench == "xstest":
            continue
        appendix.append(r"\paragraph{"+tex_escape(bench)+"}")
        for label, value in arms.items():
            rows = []
            for path, metric in metric_leaves(value):
                estimate = metric["estimate"]
                n,failed=metric_counts(bench,value,path,metric)
                if isinstance(estimate, list):
                    for k, objective in enumerate(("helpfulness", "harmlessness")):
                        rows.append([tex_escape(compact_metric_path(path)+" "+objective), point({"estimate":estimate[k],"ci95":metric["ci95"][k]},percent=False),str(n),str(failed)])
                else:
                    rows.append([tex_escape(compact_metric_path(path)), point(metric,percent=bench!="saferlhf"),str(n),str(failed)])
            appendix.append(r"\textit{"+names[label]+"}")
            # Split long category/vector tables into bounded groups without dropping rows.
            for start in range(0,len(rows),18):
                appendix.append(table_tex(["Metric", r"Estimate [95\% CI]","n","Failed"],rows[start:start+18]))
    appendix += [r"\paragraph{Length and lexical diagnostics only}"]
    length_rows = []
    for label, benchmarks in report["diagnostics_only"].items():
        for bench,row in benchmarks.items():
            length_rows.append([names[label],tex_escape(bench),f"{row['mean_response_tokens']:.1f}",f"{row['median_response_tokens']:.1f}",str(row['max_length_hits']),str(row['empty_text']),str(row['keyword_refusal_count_diagnostic_only'])])
    for bench in ("alpaca_eval","arena_hard"):
        for label,value in report["benchmarks"][bench].items():
            for subset,metric in value["subsets"].items():
                row=metric["candidate_diagnostics"]
                length_rows.append([names[label],tex_escape(subset),f"{row['mean_response_tokens']:.1f}",f"{row['median_response_tokens']:.1f}",str(row['max_length_hits']),str(row['empty_text']),str(row['keyword_refusal_count_diagnostic_only'])])
    appendix.append(table_tex(["Method","Benchmark","Mean","Median","Caps","Empty","Keywords"],length_rows))
    appendix += [r"Keywords are lexical diagnostics, not semantic refusals. Length caps do not establish early EOS. SafeRLHF surplus uses the original independent-reference disagreement, while the greedy-base difference uses an actually generated baseline. Neither certifies Algorithm~1 acceptance of the stochastic neural policy; the test is not claimed globally untouched.",
        r"\paragraph{Frozen finite-pool teacher: independent RM diagnostic}"]
    for start in range(0,len(report["teacher_rm"]["metrics"]),18):
        items = list(report["teacher_rm"]["metrics"].items())[start:start+18]
        appendix.append(table_tex(["Metric",r"Estimate [95\% CI]","Defined prompts"],
            [[tex_escape(k),point(v,percent=False),str(v["n_prompts"])] for k,v in items]))
    return "\n".join(main), "\n".join(appendix)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--paper-fragments", action="store_true")
    args = parser.parse_args()
    require(not(args.allow_incomplete and args.paper_fragments), "--allow-incomplete never writes paper fragments")
    require(args.output.resolve().is_relative_to(args.root.resolve()) and args.output.resolve()!=args.root.resolve(), "Use a fresh report child directory in task root")
    if args.output.exists():
        raise FileExistsError("Preserve previous reports; choose a new output directory")
    collector = Collector(args.root)
    report = collector.collect()
    report["allow_incomplete_mode"] = args.allow_incomplete
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/"primary_results.json",report)
    (args.output/"primary_results.md").write_text(markdown(report))
    if args.paper_fragments and report["status"] == "COMPLETE":
        main_tex, appendix_tex = paper_fragments(report)
        (args.output/"primary_main_table.tex").write_text(main_tex)
        (args.output/"primary_appendix.tex").write_text(appendix_tex)
    write_json(args.output/"manifest.json",{"status":report["status"],"source_sha256":file_hash(__file__),
        "outputs_sha256":{p.name:file_hash(p) for p in sorted(args.output.iterdir()) if p.is_file()}})
    print(json.dumps({"status":report["status"],"missing":len(report["missing"]),"errors":len(report["errors"]),"output":str(args.output)}))
    raise SystemExit(2 if report["errors"] or (report["missing"] and not args.allow_incomplete) else 0)


if __name__ == "__main__":
    main()
