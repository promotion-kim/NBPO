import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from scripts.simpo_trainer import SimPOTrainer
from mnpo_scripts.mnpo_config import MNPOConfig
from transformers import AutoModelForCausalLM, DataCollator, PreTrainedModel, PreTrainedTokenizerBase, Trainer


def validate_nbpo_args(
    args,
    dataset_columns=None,
    precompute_meta=None,
    tokenizer_hash: Optional[str] = None,
    chat_template_hash: Optional[str] = None,
    expected_parent_fingerprint: Optional[str] = None,
) -> None:
    """Hard-error validation for the paper-exact ``loss_type: nbpo`` branch.

    Called with only ``args`` from ``MNPOTrainer.__init__`` (config-level
    invariants) and again from ``run_mnpo`` with the dataset columns, the
    precompute sidecar (``precompute_meta.json``), and the training tokenizer's
    canonical hashes. A trainer constructed directly, outside ``run_mnpo``,
    gets only the config-level checks -- the sidecar lives next to the dataset,
    which the trainer object cannot locate on its own.

    No-op for every other loss type. For ``nbpo`` it enforces the Eq. (26)
    contract: no auxiliary anchor loss, no preference-SFT loss, sequence-sum
    log probabilities, exactly one center/history policy (``pi_t``), the
    ``nbpo_weighted_z`` target column, and a precompute artifact whose logps
    were also computed with ``reduction="sum"`` under the same tokenizer and
    chat template. When ``expected_parent_fingerprint`` (the content fingerprint
    of the checkpoint being trained, i.e. pi_t) is given, the artifact's
    ``history_fingerprints`` must equal ``[expected_parent_fingerprint]``:
    tokenizer equality is not weight equality, and history0 must be the true
    proximal centre of Eq. (15).
    """
    if str(getattr(args, "loss_type", "")).lower() != "nbpo":
        return
    problems = []
    if float(getattr(args, "reference_anchor_weight", 0.0)) != 0.0:
        problems.append("reference_anchor_weight must be 0 (Eq. (26) has no auxiliary anchor loss)")
    if float(getattr(args, "preference_sft_weight", 0.0)) != 0.0:
        problems.append("preference_sft_weight must be 0 (Eq. (26) has no preference-SFT loss)")
    if str(getattr(args, "logp_reduction", "mean")).lower() != "sum":
        problems.append(
            "logp_reduction must be 'sum' (h_t of Eq. (22) uses sequence-sum "
            "response log-probabilities, not token means)"
        )
    if int(getattr(args, "max_history_t", 0)) != 1:
        problems.append("max_history_t must be 1 (exactly one proximal center pi_t)")
    hw = [float(w) for w in (getattr(args, "history_weights", None) or [])]
    legacy_w = [float(w) for w in (getattr(args, "weights", None) or [])]
    if (hw and hw != [1.0]) or (legacy_w and legacy_w != [1.0]):
        problems.append(
            "history_weights must be [1.0]; NBPO never mixes multiple history models"
        )
    target_column = str(getattr(args, "nbpo_target_column", "nbpo_weighted_z"))
    if dataset_columns is not None:
        cols = set(dataset_columns)
        if target_column not in cols:
            problems.append(f"dataset lacks the `{target_column}` target column")
        if not {"history0_chosen_logps", "history0_rejected_logps"} <= cols:
            problems.append("dataset lacks history0 logp columns (the proximal center pi_t)")
        if {"history1_chosen_logps", "history1_rejected_logps"} & cols:
            problems.append(
                "dataset carries more than one history model; NBPO uses exactly one center"
            )
        if precompute_meta is None:
            problems.append(
                "precompute_meta.json is missing from the dataset artifact -- re-run "
                "mnpo_scripts.precompute with --logp_reduction sum (legacy artifacts "
                "recorded no reduction and cannot be trusted for nbpo)"
            )
        else:
            meta_reduction = str(precompute_meta.get("logp_reduction", "")).lower()
            if meta_reduction != "sum":
                problems.append(
                    f"precomputed logps were stored with reduction={meta_reduction!r}; "
                    "nbpo requires reduction='sum'"
                )
            if tokenizer_hash is not None and precompute_meta.get("tokenizer_hash") != tokenizer_hash:
                problems.append(
                    "tokenizer hash mismatch between the precompute artifact and the "
                    "training tokenizer"
                )
            if (
                chat_template_hash is not None
                and precompute_meta.get("chat_template_hash") != chat_template_hash
            ):
                problems.append(
                    "chat-template hash mismatch between the precompute artifact and "
                    "the training tokenizer"
                )
            if expected_parent_fingerprint is not None:
                hist = precompute_meta.get("history_fingerprints")
                if hist != [expected_parent_fingerprint]:
                    problems.append(
                        "history0 is not the proximal centre pi_t: precompute artifact "
                        f"history_fingerprints={hist} but the checkpoint being trained has "
                        f"fingerprint [{expected_parent_fingerprint}] (weights differ even if "
                        "the tokenizer matches)"
                    )
    if problems:
        raise ValueError("invalid configuration for loss_type=nbpo: " + "; ".join(problems))


class MNPOTrainer(SimPOTrainer):
    def __init__(
            self,
            model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
            args: Optional[MNPOConfig] = None,
            **kwargs,
    ):

        super().__init__(model=model, args=args, **kwargs)

        requested_loss_type = str(getattr(args, "loss_type", "mnpo")).lower()
        self.loss_type = requested_loss_type

        # MNPO / INPO / SPPO parameters
        self.ratio = float(args.ratio)
        self.eta = float(args.eta)
        self.beta = float(args.beta)              # metric logging only
        self.max_history_t = int(args.max_history_t)

        # DPO / SimPO parameters
        self.dpo_beta = float(getattr(args, "dpo_beta", 0.01))
        self.simpo_beta = float(getattr(args, "simpo_beta", 2.0))
        self.simpo_gamma = float(getattr(args, "simpo_gamma", 1.0))

        # RONPO parameters
        self.ronpo_alpha = float(getattr(args, "ronpo_alpha", 1.0))
        self.ronpo_tau = float(getattr(args, "ronpo_tau", 0.05))
        self.ronpo_target_column = str(getattr(args, "ronpo_target_column", "ronpo_target"))
        self.ronpo_target_schedule_columns = list(
            getattr(args, "ronpo_target_schedule_columns", None) or []
        )
        self.ronpo_target_schedule_boundaries = [
            int(value) for value in (getattr(args, "ronpo_target_schedule_boundaries", None) or [])
        ]
        if self.ronpo_target_schedule_columns:
            if len(self.ronpo_target_schedule_columns) != len(self.ronpo_target_schedule_boundaries):
                raise ValueError("RONPO target schedule columns and boundaries must have equal length")
            if self.ronpo_target_schedule_boundaries != sorted(self.ronpo_target_schedule_boundaries):
                raise ValueError("RONPO target schedule boundaries must be sorted")
            if not self.ronpo_target_schedule_boundaries or self.ronpo_target_schedule_boundaries[0] != 0:
                raise ValueError("RONPO target schedule must start at optimizer step 0")
        self.reference_anchor_weight = float(getattr(args, "reference_anchor_weight", 0.0))
        self.preference_sft_weight = float(getattr(args, "preference_sft_weight", 0.0))

        self.ht_target_column = str(getattr(args, "ht_target_column", "ht_target"))
        self.ht_target_scale = float(getattr(args, "ht_target_scale", 1.0))

        # NBPO (paper-exact finite-temperature branch)
        self.nbpo_target_column = str(getattr(args, "nbpo_target_column", "nbpo_weighted_z"))
        self.logp_reduction = str(getattr(args, "logp_reduction", "mean")).lower()
        if self.logp_reduction not in ("mean", "sum"):
            raise ValueError(f"logp_reduction must be 'mean' or 'sum', got {self.logp_reduction!r}")
        validate_nbpo_args(args)  # config-level invariants; run_mnpo re-validates with the dataset

        # Accept both the Qwen runner name (`history_weights`) and the released
        # Gemma config name (`weights`) so opponent mixtures are actually used.
        hw = getattr(args, "history_weights", None)
        legacy_weights = getattr(args, "weights", None)
        if hw:
            self.weights = list(hw)
        elif legacy_weights:
            self.weights = list(legacy_weights)
        else:
            self.weights = [1.0 for _ in range(self.max_history_t)]

    # ------------------------------------------------------------------ #
    #  History opponent mixing (shared by mnpo / inpo)                   #
    # ------------------------------------------------------------------ #
    def _history_mix(self, like: torch.Tensor, history_logps_list, device) -> torch.Tensor:
        """Weighted average of historical-policy log-ratios:  sum_j lambda_j * log[pi_{t-j}(yw)/pi_{t-j}(yl)]."""
        weighted = torch.zeros_like(like, device=device)
        t = self.max_history_t
        if history_logps_list and t > 0:
            effective_t = min(len(history_logps_list), t)
            if effective_t > 0:
                weights = [float(w) for w in self.weights] if self.weights else []
                if not weights:
                    weights = [1.0] * effective_t
                weights = weights[:effective_t]
                total_weight = float(sum(weights)) if sum(weights) != 0 else 0.0
                for j, (chosen_j, rejected_j) in enumerate(history_logps_list[:effective_t]):
                    chosen_j = torch.as_tensor(chosen_j, device=device, dtype=torch.float32)
                    rejected_j = torch.as_tensor(rejected_j, device=device, dtype=torch.float32)
                    lambda_j = weights[j] / total_weight if total_weight > 0 else 1.0 / float(effective_t)
                    weighted = weighted + lambda_j * (chosen_j - rejected_j)
        return weighted

    # ------------------------------------------------------------------ #
    #  Unified preference loss (dispatch on self.loss_type)              #
    #  NOTE: kept the public name `mnpo_loss` for backward compatibility.#
    # ------------------------------------------------------------------ #
    def mnpo_loss(
            self,
            policy_chosen_logps: torch.FloatTensor,
            policy_rejected_logps: torch.FloatTensor,
            reference_chosen_logps: torch.FloatTensor,
            reference_rejected_logps: torch.FloatTensor,
            history_logps_list: List[Tuple[torch.FloatTensor, torch.FloatTensor]],
            chosen_probs: Optional[torch.FloatTensor] = None,
            rejected_probs: Optional[torch.FloatTensor] = None,
            ronpo_target: Optional[torch.FloatTensor] = None,
            ht_target: Optional[torch.FloatTensor] = None,
            nbpo_target: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:

        device = self.accelerator.device

        pcl = policy_chosen_logps.to(device=device, dtype=torch.float32)
        prl = policy_rejected_logps.to(device=device, dtype=torch.float32)
        rcl = reference_chosen_logps.to(device=device, dtype=torch.float32)
        rrl = reference_rejected_logps.to(device=device, dtype=torch.float32)

        pi_logratios = pcl - prl       # log[pi(yw)/pi(yl)]
        ref_logratios = rcl - rrl      # log[mu(yw)/mu(yl)]

        lt = self.loss_type

        if lt == "simpo":
            # reference-free, length-normalized reward with a target margin gamma
            logits = self.simpo_beta * pi_logratios - self.simpo_gamma
            losses = -F.logsigmoid(logits)

        elif lt == "dpo":
            # logistic loss vs. the reference policy (length-normalized here)
            logits = self.dpo_beta * (pi_logratios - ref_logratios)
            losses = -F.logsigmoid(logits)

        elif lt == "ipo":
            # TRL DPOTrainer loss_type="ipo" on the same length-normalized
            # log-ratios used by this trainer's data collator.
            logits = pi_logratios - ref_logratios
            losses = (logits - 1.0 / (2.0 * self.dpo_beta)) ** 2

        elif lt == "sppo":
            # opponent = previous policy = history0 (pi_t); reference-free.
            # target uses the estimated per-response win rate against pi_t.
            if not history_logps_list:
                raise ValueError("SPPO requires history0 (previous policy) logps via --history_paths.")
            prev_chosen, prev_rejected = history_logps_list[0]
            prev_chosen = torch.as_tensor(prev_chosen, device=device, dtype=torch.float32)
            prev_rejected = torch.as_tensor(prev_rejected, device=device, dtype=torch.float32)
            if chosen_probs is None or rejected_probs is None:
                raise ValueError(
                    "SPPO requires `chosen_probs` and `rejected_probs` columns "
                    "(estimated win rates P_hat(y > pi_t)) produced during annotation."
                )
            pw = chosen_probs.to(device=device, dtype=torch.float32)
            pl = rejected_probs.to(device=device, dtype=torch.float32)
            eta = self.eta
            loss_c = (pcl - prev_chosen - eta * (pw - 0.5)) ** 2
            loss_r = (prl - prev_rejected - eta * (pl - 0.5)) ** 2
            losses = loss_c + loss_r

        elif lt == "ronpo":
            # Practical RONPO pairwise regression:
            #   h = rho_pi - alpha*tau*rho_ref - (1-alpha*tau)rho_pi_t
            #   target = alpha * (z_y - z_y_prime)
            # The training pair is not necessarily preference ordered; `chosen` and
            # `rejected` simply name the two sampled responses y and y_prime.
            if not history_logps_list:
                raise ValueError("RONPO requires history0 (the sampling policy pi_t) via --history_paths.")
            if ronpo_target is None:
                raise ValueError(
                    f"RONPO requires a `{self.ronpo_target_column}` column containing "
                    "z_y - z_y_prime values in {-1, 0, 1}."
                )
            prev_chosen, prev_rejected = history_logps_list[0]
            prev_chosen = torch.as_tensor(prev_chosen, device=device, dtype=torch.float32)
            prev_rejected = torch.as_tensor(prev_rejected, device=device, dtype=torch.float32)
            target = ronpo_target.to(device=device, dtype=torch.float32)

            alpha = self.ronpo_alpha
            tau = self.ronpo_tau
            prev_logratios = prev_chosen - prev_rejected
            logits = pi_logratios - alpha * tau * ref_logratios - (1.0 - alpha * tau) * prev_logratios
            losses = (logits - alpha * target) ** 2

        elif lt == "nbpo":
            # Paper-exact finite-temperature NBPO regression, manuscript Eq. (26)
            # `eq:regression-loss`:
            #   h_t = [log pi(y) - log pi(y')] - [log pi_t(y) - log pi_t(y')]  (Eq. (22))
            #   loss = (h_t - eta * nbpo_weighted_z)^2
            # history0 is the proximal center pi_t of Eq. (15). The target column
            # already carries the RAW dual weights (sum_k lambda_k Z_k, Eq. (24));
            # eta is applied exactly once, here. Log-probabilities are sequence
            # sums over non-masked response tokens (prompt tokens stay masked by
            # the tokenization path); validate_nbpo_args enforces
            # logp_reduction="sum" and forbids reference interpolation, alpha/tau,
            # multi-history mixing, and every auxiliary loss.
            if not history_logps_list:
                raise ValueError("NBPO requires history0 (the proximal center pi_t) via --history_paths.")
            if nbpo_target is None:
                raise ValueError(
                    f"NBPO requires a `{self.nbpo_target_column}` column containing the "
                    "unscaled weighted binary target sum_k lambda_k Z_k."
                )
            prev_chosen, prev_rejected = history_logps_list[0]
            prev_chosen = torch.as_tensor(prev_chosen, device=device, dtype=torch.float32)
            prev_rejected = torch.as_tensor(prev_rejected, device=device, dtype=torch.float32)
            target = self.eta * nbpo_target.to(device=device, dtype=torch.float32)
            h = pi_logratios - (prev_chosen - prev_rejected)
            losses = (h - target) ** 2

        elif lt in ("mnpo", "inpo"):
            # squared loss to the constant target 1/(2 eta).
            # INPO = this path with a single history opponent (pi_{t-1}); MNPO = >=1 weighted opponents.
            weighted_logratios = self._history_mix(pi_logratios, history_logps_list, device)
            logits = pi_logratios - self.ratio * ref_logratios - (1.0 - self.ratio) * weighted_logratios
            logits_shift = 1.0 / (2.0 * self.eta)
            losses = (logits - logits_shift) ** 2

        elif lt == "mopo":
            # Agnihotri et al., Eq. (7): the MOPO policy step is importance-weighted behaviour cloning
            # against the lagged reference, not a pairwise regression. The row carries
            # rho(y) = exp(tau^{-1}[p(y > y') + lambda^T q(y > y')] - 1) from Eq. (3), computed offline
            # from the judged pool, and the loss is -rho(y) log pi(y) on the sampled response.
            if ronpo_target is None:
                raise ValueError(
                    f"MOPO requires a `{self.ronpo_target_column}` column holding the importance weight "
                    "rho(y) of Eq. (3)."
                )
            rho = ronpo_target.to(device=device, dtype=torch.float32)
            losses = -rho * pcl

        elif lt == "ht_mnpo":
            # Eq. 18: policy log-ratio against the opponent mixture regresses to eta * delta_i*.
            if not history_logps_list:
                raise ValueError("HT-MNPO requires opponent logps via --history_paths.")
            if ht_target is None:
                raise ValueError(
                    f"HT-MNPO requires a `{self.ht_target_column}` column containing "
                    "the player-specific reward-model gap delta_i*."
                )
            opponent_logratios = self._history_mix(pi_logratios, history_logps_list, device)
            target = self.eta * self.ht_target_scale * ht_target.to(device=device, dtype=torch.float32)
            logits = pi_logratios - opponent_logratios
            losses = (logits - target) ** 2

        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type!r}")

        # logged reward metric (not part of any loss)
        chosen_rewards = self.beta * (pcl - rcl).detach()
        rejected_rewards = self.beta * (prl - rrl).detach()

        return losses, chosen_rewards, rejected_rewards

    def pack_history_logps_from_dataset(self, batch: Dict[str, torch.Tensor]) -> List[
        Tuple[torch.FloatTensor, torch.FloatTensor]]:
        """Extract historical logps (history0, history1, ...) from the precomputed batch."""
        history_logps_list = []
        for j in range(self.max_history_t):
            key_c = f"history{j}_chosen_logps"
            key_r = f"history{j}_rejected_logps"
            if key_c in batch and key_r in batch:
                history_logps_list.append((batch[key_c], batch[key_r]))
            else:
                break
        return history_logps_list

    def get_batch_loss_metrics(
            self,
            model,
            batch: Dict[str, Union[List, torch.LongTensor]],
            train_eval: str = "train",
    ):
        """Compute the selected preference loss and metrics for the batch."""
        metrics = {}
        prefix = "eval_" if train_eval == "eval" else ""

        # 1. Policy logps via the efficient concatenated_forward from SimPOTrainer
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            chosen_labels,
        ) = self.concatenated_forward(model, batch)

        # 2. Reference / history logps from the precomputed batch columns
        reference_chosen_logps = batch['reference_chosen_logps'].to(self.accelerator.device)
        reference_rejected_logps = batch['reference_rejected_logps'].to(self.accelerator.device)

        history_logps_list = self.pack_history_logps_from_dataset(batch)
        history_logps_list = [
            (c.to(self.accelerator.device), r.to(self.accelerator.device))
            for c, r in history_logps_list
        ]

        # 2b. SPPO per-response win-rate columns (optional; only required for loss_type="sppo")
        chosen_probs = batch.get("chosen_probs", None)
        rejected_probs = batch.get("rejected_probs", None)
        if chosen_probs is not None:
            chosen_probs = torch.as_tensor(chosen_probs, dtype=torch.float32, device=self.accelerator.device)
        if rejected_probs is not None:
            rejected_probs = torch.as_tensor(rejected_probs, dtype=torch.float32, device=self.accelerator.device)

        # 2c. RONPO target column: z_y - z_y_prime, usually in {-1, 0, 1}.
        active_target_column = self.ronpo_target_column
        if self.ronpo_target_schedule_columns:
            step = int(getattr(self.state, "global_step", 0))
            active_index = max(
                index for index, boundary in enumerate(self.ronpo_target_schedule_boundaries)
                if step >= boundary
            )
            active_target_column = self.ronpo_target_schedule_columns[active_index]
        ronpo_target = batch.get(active_target_column, None)
        if ronpo_target is not None:
            ronpo_target = torch.as_tensor(ronpo_target, dtype=torch.float32, device=self.accelerator.device)

        ht_target = batch.get(self.ht_target_column, None)
        if ht_target is not None:
            ht_target = torch.as_tensor(ht_target, dtype=torch.float32, device=self.accelerator.device)

        # 2d. NBPO target column: unscaled sum_k lambda_k Z_k (raw dual weights).
        nbpo_target = batch.get(self.nbpo_target_column, None)
        if nbpo_target is not None:
            nbpo_target = torch.as_tensor(nbpo_target, dtype=torch.float32, device=self.accelerator.device)

        ronpo_weight = batch.get("ronpo_weight", None)
        if ronpo_weight is not None:
            ronpo_weight = torch.as_tensor(ronpo_weight, dtype=torch.float32, device=self.accelerator.device)

        # 3. Compute loss
        losses, chosen_rewards, rejected_rewards = self.mnpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            history_logps_list,
            chosen_probs,
            rejected_probs,
            ronpo_target,
            ht_target,
            nbpo_target,
        )

        if self.loss_type == "ronpo" and ronpo_weight is not None:
            weight = ronpo_weight / ronpo_weight.mean().clamp_min(1e-8)
            losses = losses * weight

        core_loss = losses.mean()
        loss = core_loss

        reference_anchor_loss = None
        if self.reference_anchor_weight > 0.0:
            policy_chosen = policy_chosen_logps.to(self.accelerator.device, dtype=torch.float32)
            policy_rejected = policy_rejected_logps.to(self.accelerator.device, dtype=torch.float32)
            reference_chosen = reference_chosen_logps.to(self.accelerator.device, dtype=torch.float32)
            reference_rejected = reference_rejected_logps.to(self.accelerator.device, dtype=torch.float32)
            reference_anchor_loss = 0.5 * (
                (policy_chosen - reference_chosen).square()
                + (policy_rejected - reference_rejected).square()
            ).mean()
            loss = loss + self.reference_anchor_weight * reference_anchor_loss

        preference_sft_loss = None
        if self.preference_sft_weight > 0.0:
            policy_chosen = policy_chosen_logps.to(self.accelerator.device, dtype=torch.float32)
            policy_rejected = policy_rejected_logps.to(self.accelerator.device, dtype=torch.float32)
            if ronpo_target is None:
                preferred_nll = -policy_chosen
            else:
                target = ronpo_target.to(self.accelerator.device, dtype=torch.float32)
                preferred_nll = torch.where(
                    target > 0,
                    -policy_chosen,
                    torch.where(target < 0, -policy_rejected, -0.5 * (policy_chosen + policy_rejected)),
                )
            preference_sft_loss = preferred_nll.mean()
            loss = loss + self.preference_sft_weight * preference_sft_loss

        # 4. Metrics
        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.cpu().mean()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.cpu().mean()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.cpu().mean()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).cpu().mean()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().cpu().mean()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().cpu().mean()
        reference_pair_logratio = reference_chosen_logps - reference_rejected_logps
        policy_pair_logratio = policy_chosen_logps - policy_rejected_logps
        metrics[f"{prefix}drift/pair_logratio_to_reference_abs"] = (
            policy_pair_logratio.detach().float() - reference_pair_logratio.detach().float()
        ).abs().mean().cpu()
        metrics[f"{prefix}drift/offpolicy_reverse_kl_reference_proxy"] = 0.5 * (
            reference_chosen_logps.detach().float() - policy_chosen_logps.detach().float()
            + reference_rejected_logps.detach().float() - policy_rejected_logps.detach().float()
        ).mean().cpu()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().cpu().mean()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().cpu().mean()
        metrics[f"{prefix}loss/core"] = core_loss.detach().cpu()
        if reference_anchor_loss is not None:
            metrics[f"{prefix}loss/reference_anchor"] = reference_anchor_loss.detach().cpu()
            metrics[f"{prefix}drift/chosen_abs"] = (
                policy_chosen_logps.detach().float() - reference_chosen_logps.detach().float()
            ).abs().mean().cpu()
            metrics[f"{prefix}drift/rejected_abs"] = (
                policy_rejected_logps.detach().float() - reference_rejected_logps.detach().float()
            ).abs().mean().cpu()
        if preference_sft_loss is not None:
            metrics[f"{prefix}loss/preference_sft"] = preference_sft_loss.detach().cpu()

        if self.loss_type == "ronpo" and ronpo_target is not None and history_logps_list:
            prev_chosen, prev_rejected = history_logps_list[0]
            prev_chosen = prev_chosen.to(self.accelerator.device, dtype=torch.float32)
            prev_rejected = prev_rejected.to(self.accelerator.device, dtype=torch.float32)
            ref_logratios = reference_chosen_logps - reference_rejected_logps
            prev_logratios = prev_chosen - prev_rejected
            policy_logratios = policy_chosen_logps - policy_rejected_logps
            ronpo_logits = (
                policy_logratios
                - self.ronpo_alpha * self.ronpo_tau * ref_logratios
                - (1.0 - self.ronpo_alpha * self.ronpo_tau) * prev_logratios
            )
            ronpo_targets = self.ronpo_alpha * ronpo_target
            metrics[f"{prefix}ronpo/logit"] = ronpo_logits.detach().mean().cpu()
            metrics[f"{prefix}ronpo/target"] = ronpo_targets.detach().mean().cpu()
            metrics[f"{prefix}ronpo/target_abs"] = ronpo_targets.detach().abs().mean().cpu()
            metrics[f"{prefix}ronpo/residual"] = (ronpo_logits - ronpo_targets).detach().mean().cpu()
            metrics[f"{prefix}ronpo/residual_abs"] = (ronpo_logits - ronpo_targets).detach().abs().mean().cpu()
            if ronpo_weight is not None:
                metrics[f"{prefix}ronpo/weight"] = ronpo_weight.detach().mean().cpu()
                metrics[f"{prefix}ronpo/weight_max"] = ronpo_weight.detach().max().cpu()

        if self.loss_type == "nbpo" and nbpo_target is not None and history_logps_list:
            prev_chosen, prev_rejected = history_logps_list[0]
            prev_chosen = prev_chosen.to(self.accelerator.device, dtype=torch.float32)
            prev_rejected = prev_rejected.to(self.accelerator.device, dtype=torch.float32)
            policy_logratios = (policy_chosen_logps - policy_rejected_logps).to(
                self.accelerator.device, dtype=torch.float32
            )
            nbpo_h = policy_logratios - (prev_chosen - prev_rejected)
            nbpo_scaled_target = self.eta * nbpo_target
            metrics[f"{prefix}nbpo/h"] = nbpo_h.detach().mean().cpu()
            metrics[f"{prefix}nbpo/target"] = nbpo_scaled_target.detach().mean().cpu()
            metrics[f"{prefix}nbpo/target_abs"] = nbpo_scaled_target.detach().abs().mean().cpu()
            metrics[f"{prefix}nbpo/residual"] = (nbpo_h - nbpo_scaled_target).detach().mean().cpu()
            metrics[f"{prefix}nbpo/residual_abs"] = (
                (nbpo_h - nbpo_scaled_target).detach().abs().mean().cpu()
            )

        if self.loss_type == "ht_mnpo" and ht_target is not None and history_logps_list:
            policy_logratios = (policy_chosen_logps - policy_rejected_logps).to(
                self.accelerator.device,
                dtype=torch.float32,
            )
            opponent_logratios = self._history_mix(
                policy_logratios,
                history_logps_list,
                self.accelerator.device,
            )
            ht_logits = policy_logratios - opponent_logratios
            ht_targets = self.eta * self.ht_target_scale * ht_target.to(self.accelerator.device, dtype=torch.float32)
            metrics[f"{prefix}ht_mnpo/logit"] = ht_logits.detach().mean().cpu()
            metrics[f"{prefix}ht_mnpo/target"] = ht_targets.detach().mean().cpu()
            metrics[f"{prefix}ht_mnpo/target_abs"] = ht_targets.detach().abs().mean().cpu()
            metrics[f"{prefix}ht_mnpo/residual"] = (ht_logits - ht_targets).detach().mean().cpu()
            metrics[f"{prefix}ht_mnpo/residual_abs"] = (ht_logits - ht_targets).detach().abs().mean().cpu()

        metrics[f"{prefix}loss"] = loss.detach().cpu()
        return loss, metrics
