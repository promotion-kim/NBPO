"""Build MaxMin-RLHF pair data from an existing merged-scores file.

MaxMin-RLHF (Chakraborty et al., 2024) trains on the reward mixture
sum_k w_k r_k whose weights follow a multiplicative-weights update on the
policy's objective-level performance, so the worst-served objective gains
mass between rounds. Round 1 with uniform w is exactly the averaged-oracle
stage-1 baseline; this script produces the round-2 (stage-2) pair labels:
per-prompt weighted average of the min-max-normalized objective scores, with
w_k proportional to exp(-mw_eta * g_k) for the measured stage-1 objective-level
normalized reward g_k. Output schema matches build_multi_objective_dataset's
--mnpo_output so the standard precompute/train path consumes it unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mnpo_scripts.build_multi_objective_dataset import (
    add_pair_fields,
    average_win_probs,
    normalize_scores,
)


def build_split(merged_path: Path, out_path: Path, weights: dict[str, float],
                scale: float, mw_eta: float, perf: dict[str, float]) -> int:
    n = 0
    with merged_path.open() as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            record = json.loads(line)
            names = list(record["objective_names"])
            norm = normalize_scores(record["objective_scores"], "minmax")
            num = len(record["all_generated_responses"])
            wavg = [sum(weights[k] * norm[k][i] for k in names) for i in range(num)]
            win_probs = average_win_probs(wavg, scale)
            best = max(range(num), key=wavg.__getitem__)
            worst = min(range(num), key=wavg.__getitem__)
            if best == worst:
                continue
            base = dict(record)
            base["normalized_objective_scores"] = norm
            base["avg_objective_scores"] = wavg
            base["min_objective_scores"] = [min(norm[k][i] for k in names) for i in range(num)]
            base["avg_oracle_win_probs"] = win_probs
            base["homogeneous_oracle"] = "maxmin_weighted_minmax_objectives"
            base["homogeneous_oracle_objectives"] = names
            base["homogeneous_oracle_preference_scale"] = float(scale)
            base["maxmin_weights"] = {k: weights[k] for k in names}
            base["maxmin_mw_eta"] = float(mw_eta)
            base["maxmin_stage1_perf"] = {k: perf[k] for k in names}
            pair = add_pair_fields(
                base,
                chosen_idx=best,
                rejected_idx=worst,
                target=1.0,
                pair_source="maxmin_weighted_objective",
                objective_name=None,
                objective_index=None,
                objective_gap=wavg[best] - wavg[worst],
            )
            pair["chosen_probs"] = float(win_probs[best])
            pair["rejected_probs"] = float(win_probs[worst])
            pair["avg_oracle_chosen_score"] = float(wavg[best])
            pair["avg_oracle_rejected_score"] = float(wavg[worst])
            pair["avg_oracle_score_gap"] = float(wavg[best] - wavg[worst])
            pair["avg_oracle_chosen_win_prob"] = float(win_probs[best])
            pair["avg_oracle_rejected_win_prob"] = float(win_probs[worst])
            fout.write(json.dumps(pair, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_merged", required=True)
    ap.add_argument("--test_merged", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--objective_perf", required=True,
                    help="stage-1 objective-level normalized reward, e.g. skywork=0.5962,athene=0.6768,armo=0.6208")
    ap.add_argument("--mw_eta", type=float, default=20.0,
                    help="MW temperature; 20 = 1/kappa, matching the entropic adversary scale")
    ap.add_argument("--preference_scale", type=float, default=8.0)
    args = ap.parse_args()

    perf = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in args.objective_perf.split(",")}
    import math
    raw = {k: math.exp(-args.mw_eta * v) for k, v in perf.items()}
    z = sum(raw.values())
    weights = {k: v / z for k, v in raw.items()}
    print("[maxmin] stage-1 perf:", perf)
    print("[maxmin] MW weights (eta=%g):" % args.mw_eta, {k: round(v, 4) for k, v in weights.items()})

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    n_tr = build_split(Path(args.train_merged), out / "train_maxmin_oracle.jsonl",
                       weights, args.preference_scale, args.mw_eta, perf)
    n_te = build_split(Path(args.test_merged), out / "test_maxmin_oracle.jsonl",
                       weights, args.preference_scale, args.mw_eta, perf)
    (out / "maxmin_summary.json").write_text(json.dumps({
        "weights": weights, "mw_eta": args.mw_eta, "stage1_perf": perf,
        "preference_scale": args.preference_scale,
        "train_examples": n_tr, "test_examples": n_te,
    }, indent=2) + "\n")
    print(f"[maxmin] wrote {n_tr} train / {n_te} test pairs -> {out}")


if __name__ == "__main__":
    main()
