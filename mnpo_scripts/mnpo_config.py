from dataclasses import dataclass, field
from scripts.simpo_config import SimPOConfig
from typing import List, Optional, Tuple, Union


@dataclass
class MNPOConfig(SimPOConfig):
    """
    Configuration class for MNPOTrainer.

    A single `loss_type` switch selects which preference objective to optimize.
    All variants share the same data pipeline (split -> on-policy gen -> annotate
    -> precompute), so only the loss head differs.

        loss_type in {"mnpo", "ht_mnpo", "inpo", "dpo", "ipo", "simpo", "sppo", "ronpo"}

      - "mnpo" / "inpo":  squared loss to the constant target 1/(2*eta)  (INPO = single history opponent)
      - "ht_mnpo":        heterogeneous MNPO Eq. 18: squared loss between
                          policy-vs-opponent log-ratio and eta * ht_target.
      - "dpo":            logistic loss vs. reference  (length-normalized here; see note below)
      - "ipo":            TRL IPO squared loss vs. reference
      - "simpo":          reference-free length-normalized reward with margin
      - "sppo":           squared loss vs. previous policy with a per-response win-rate target
                          (requires `chosen_probs`/`rejected_probs` columns from annotation)
      - "ronpo":          partition-free robust objective-adversarial regression
                          (requires history0 logps for pi_t and a `ronpo_target` column
                          containing z_y - z_y_prime in {-1, 0, 1})
    """

    # --- which preference loss to optimize ---
    loss_type: str = "mnpo"

    # --- OMD / NLHF params (mnpo, inpo, sppo) ---
    # ratio = tau/eta : weight on the reference log-ratio in the INPO/MNPO logit.
    ratio: float = 0.3333
    # eta : OMD step; also the SPPO target scale eta*(p - 1/2).
    eta: float = 0.0075
    # number of historical opponents to read from the precomputed columns (history0, history1, ...)
    max_history_t: int = 2
    # weights for historical opponents [t-1, t-2, ...].
    # The Qwen runner uses this explicit name.
    history_weights: List[float] = field(default_factory=lambda: [1.0, 0.0])
    # Legacy upstream config name. Some released Gemma configs use `weights:`.
    weights: Optional[List[float]] = None

    # --- DPO ---
    # Logistic temperature. Because this codebase uses length-normalized logps
    # (average_log_prob=True), this is a *length-normalized* DPO; tune accordingly.
    # For an exact match to the paper's DPO number, use TRL's DPOTrainer instead.
    dpo_beta: float = 0.01

    # --- SimPO ---
    # The base SimPOTrainer already implements SimPO faithfully; these let the
    # unified trainer reproduce it on the same pipeline.
    simpo_beta: float = 2.0
    simpo_gamma: float = 1.0

    # --- metric logging only (NOT used by any loss) ---
    beta: float = 10.0

    # --- RONPO ---
    # Practical RONPO loss:
    #   h = rho_pi - alpha*tau*rho_ref - (1-alpha*tau)*rho_prev
    #   loss = (h - alpha*(z_y - z_y_prime))^2
    # where history0 is the previous/current behavior policy pi_t used for sampling.
    ronpo_alpha: float = 1.0
    ronpo_tau: float = 0.05
    ronpo_target_column: str = "ronpo_target"
    # Optional piecewise target schedule for kappa annealing.  Boundaries are
    # optimizer steps at which the corresponding target column becomes active.
    # Empty lists preserve the original single-column behavior exactly.
    ronpo_target_schedule_columns: List[str] = field(default_factory=list)
    ronpo_target_schedule_boundaries: List[int] = field(default_factory=list)
    # Pairwise log-ratio objectives leave a common-mode degree of freedom: both
    # response log-probabilities can drift together without changing the RONPO
    # residual.  Anchor observed-response logps to the frozen reference to keep
    # the policy in the local trust region.
    reference_anchor_weight: float = 0.0
    # Optional behavior-cloning term on the response preferred by the signed
    # RONPO target.  This is computed from response-only average log-probability,
    # so it works for pairs whose `chosen`/`rejected` names are not ordered.
    preference_sft_weight: float = 0.0

    # --- HT-MNPO reward-gap target ---
    # Eq. 18 uses eta * delta_i*, where delta_i* is the player-specific
    # reward-model gap. The dataset column should store delta_i* itself.
    ht_target_column: str = "ht_target"
    ht_target_scale: float = 1.0

    # --- Legacy compatibility ---
    het_loss_weight: float = 0.0
