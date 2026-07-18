import math
from pathlib import Path


OBJECTIVES = ["major_quality", "minor_decoy_sensitive"]
RESPONSES = ["policy_main", "policy_alt", "fixed_ref", "decoy_a_star"]
SCORES = {
    "major_quality": {
        "policy_main": 1.0,
        "policy_alt": 0.9,
        "fixed_ref": 0.0,
        "decoy_a_star": 0.1,
    },
    "minor_decoy_sensitive": {
        "policy_main": 0.1,
        "policy_alt": 0.2,
        "fixed_ref": 0.0,
        "decoy_a_star": 1.0,
    },
}
PI = {"policy_main": 0.8, "policy_alt": 0.2}
SCALE = 8.0


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def win_prob(objective: str, adversary_response: str) -> float:
    return sum(
        prob
        * sigmoid(
            SCALE
            * (
                SCORES[objective][policy_response]
                - SCORES[objective][adversary_response]
            )
        )
        for policy_response, prob in PI.items()
    )


def main() -> None:
    fixed_ref = "fixed_ref"
    k_only = [(obj, fixed_ref, win_prob(obj, fixed_ref)) for obj in OBJECTIVES]
    full_atom = [
        (obj, resp, win_prob(obj, resp))
        for obj in OBJECTIVES
        for resp in RESPONSES
    ]
    k_only_worst = min(k_only, key=lambda row: row[2])
    full_worst = min(full_atom, key=lambda row: row[2])

    out = Path("toy/ronpo_atom_adversary_separation.md")
    lines = [
        "# RONPO Atom-Adversary Separation Toy",
        "",
        "This construction makes the failure depend on a specific decoy response `a*`, not only on the objective weight.",
        "The current policy puts mass on `policy_main` and `policy_alt`; `fixed_ref` is the response used by a weight-only ablation.",
        "",
        "| Objective | policy_main | policy_alt | fixed_ref | decoy_a_star |",
        "|---|---:|---:|---:|---:|",
    ]
    for obj in OBJECTIVES:
        lines.append(
            "| {obj} | {pm:.2f} | {pa:.2f} | {fr:.2f} | {decoy:.2f} |".format(
                obj=obj,
                pm=SCORES[obj]["policy_main"],
                pa=SCORES[obj]["policy_alt"],
                fr=SCORES[obj]["fixed_ref"],
                decoy=SCORES[obj]["decoy_a_star"],
            )
        )
    lines.extend(
        [
            "",
            "| Adversary | Objective | Response atom | E_pi P_k(y beats a) |",
            "|---|---|---|---:|",
        ]
    )
    for obj, resp, value in k_only:
        lines.append(f"| k-only fixed-a | {obj} | {resp} | {value:.4f} |")
    lines.append(f"| full (k,a) worst | {full_worst[0]} | {full_worst[1]} | {full_worst[2]:.4f} |")
    lines.extend(
        [
            "",
            f"Weight-only worst case with fixed reference: `{k_only_worst[0]}`, value `{k_only_worst[2]:.4f}`.",
            f"Full atom adversary worst case: `({full_worst[0]}, {full_worst[1]})`, value `{full_worst[2]:.4f}`.",
            "",
            "The fixed-reference adversary concludes that every objective is protected because the policy beats `fixed_ref` under both objectives.",
            "The full atom adversary instead selects `(minor_decoy_sensitive, decoy_a_star)`, exposing a low floor that objective-only weighting cannot see.",
        ]
    )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
