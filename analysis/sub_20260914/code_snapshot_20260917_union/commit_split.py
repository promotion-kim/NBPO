"""Commit the train split from measured throughput, before any policy is trained."""
import json
from pathlib import Path

R = Path("/work/sub_20260914")
freeze = json.loads((R / "splits/freeze.json").read_text())
pilot = json.loads((R / "judgments/safe_pilot10_panelA/complete.json").read_text())
gen = json.loads((R / "responses/screen200/complete.json").read_text())

rate = pilot["judgments_per_second"]
per_prompt = 92 * 2 * 2 * 2          # 64 Y-Z + 28 Z-Z pairs, K=2, 2 orders, 2 repeats
full = (2000 + 500) * per_prompt
reduced = (1000 + 500) * per_prompt
freeze["committed_train_split"] = "train2000"
freeze["commitment_evidence"] = {
    "decided_utc_before_any_policy_training": True,
    "measured_judge_rate_verdicts_per_second_per_gpu": rate,
    "measured_on": "safe_pilot10_panelA, 2240 verdicts, parse rate %.4f" % pilot["valid_fraction"],
    "measured_generation_responses_per_second": gen["responses_per_second"],
    "verdicts_per_prompt": per_prompt,
    "train_dev_verdicts_full_plan": full,
    "train_dev_verdicts_reduced_plan": reduced,
    "gpu_hours_full_plan": full / rate / 3600.0,
    "gpu_hours_reduced_plan": reduced / rate / 3600.0,
    "wall_hours_full_plan_on_four_cards": full / rate / 3600.0 / 4.0,
    "reason": ("the pilot measured %.1f verdicts/s/GPU, so the full declared plan costs "
               "%.1f GPU-hours of labelling, about %.1f hours on the four cards. The "
               "contract permits the 1,000-prompt reduction only when throughput does not "
               "fit; it does fit, so the declared 2,000/500/1,000 plan is kept and the "
               "reduction is not used." % (rate, full / rate / 3600.0,
                                           full / rate / 3600.0 / 4.0)),
}
(R / "splits/freeze.json").write_text(json.dumps(freeze, indent=2, ensure_ascii=False) + "\n")
print(json.dumps({"committed": freeze["committed_train_split"],
                  **{k: freeze["commitment_evidence"][k] for k in
                     ("measured_judge_rate_verdicts_per_second_per_gpu",
                      "train_dev_verdicts_full_plan", "gpu_hours_full_plan",
                      "wall_hours_full_plan_on_four_cards")}}, indent=1))
