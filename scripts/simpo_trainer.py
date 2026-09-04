import csv
import inspect
import json
import os
import random
import shutil
import subprocess
import sys
import warnings
from collections import defaultdict
from contextlib import nullcontext
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import PartialState
from datasets import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, DataCollator, PreTrainedModel, PreTrainedTokenizerBase, Trainer, is_wandb_available
try:
    from trl.trainer import CPOTrainer  # noqa: F401
except ImportError:
    CPOTrainer = None
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput
from transformers.utils import is_torch_fx_proxy, is_peft_available

from scripts.simpo_config import SimPOConfig

from dataclasses import dataclass
from typing import Dict, Literal, Optional

from transformers import TrainingArguments

from mnpo_scripts.pair_tokenization import (
    build_tokenized_answer as canonical_build_tokenized_answer,
    tokenize_preference_pair,
)

try:
    from trl.trainer.utils import (
        DPODataCollatorWithPadding,
        disable_dropout_in_model,
        pad_to_length,
        peft_module_casting_to_bf16,
    )
except ImportError:
    def disable_dropout_in_model(model):
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0

    def pad_to_length(tensor: torch.Tensor, length: int, pad_value: int, dim: int = -1) -> torch.Tensor:
        if tensor.size(dim) >= length:
            return tensor
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat(
            [tensor, torch.full(pad_size, pad_value, dtype=tensor.dtype, device=tensor.device)],
            dim=dim,
        )

    def peft_module_casting_to_bf16(model):
        return model

    class DPODataCollatorWithPadding:
        """Compatibility replacement for TRL versions that removed this collator.

        SimPOTrainer tokenizes rows before collation, so this collator only pads
        tokenized prompt/chosen/rejected fields and preserves auxiliary numeric
        columns such as reference/history log-probabilities and ronpo_target.
        """

        def __init__(
            self,
            pad_token_id: int,
            label_pad_token_id: int = -100,
            is_encoder_decoder: bool = False,
        ):
            self.pad_token_id = 0 if pad_token_id is None else int(pad_token_id)
            self.label_pad_token_id = int(label_pad_token_id)
            self.is_encoder_decoder = bool(is_encoder_decoder)

        def _pad_tensor_list(self, values: List[Any], key: str) -> torch.Tensor:
            if key.endswith("_input_ids"):
                padding_value = self.pad_token_id
            elif key.endswith("_attention_mask"):
                padding_value = 0
            elif key.endswith("_labels"):
                padding_value = self.label_pad_token_id
            else:
                raise ValueError(f"Unexpected tokenized key for padding: {key}")

            tensors = [torch.as_tensor(value, dtype=torch.long) for value in values]
            if not self.is_encoder_decoder and key.startswith("prompt_"):
                tensors = [tensor.flip(0) for tensor in tensors]
                padded = pad_sequence(tensors, batch_first=True, padding_value=padding_value)
                return padded.flip(1)
            return pad_sequence(tensors, batch_first=True, padding_value=padding_value)

        @staticmethod
        def _is_scalar_number(value: Any) -> bool:
            return isinstance(value, (bool, int, float, np.integer, np.floating))

        def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
            if not features:
                return {}

            batch: Dict[str, Any] = {}
            keys = features[0].keys()
            for key in keys:
                if not all(key in feature for feature in features):
                    continue
                values = [feature[key] for feature in features]

                if key.endswith("_input_ids") or key.endswith("_attention_mask") or key.endswith("_labels"):
                    batch[key] = self._pad_tensor_list(values, key)
                elif all(self._is_scalar_number(value) for value in values):
                    if any(isinstance(value, (float, np.floating)) for value in values):
                        batch[key] = torch.tensor(values, dtype=torch.float32)
                    elif any(isinstance(value, bool) for value in values):
                        batch[key] = torch.tensor(values, dtype=torch.bool)
                    else:
                        batch[key] = torch.tensor(values, dtype=torch.long)
                else:
                    batch[key] = values

            return batch
from trl.models.utils import unwrap_model_for_generation

if is_peft_available():
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training

if is_wandb_available():
    import wandb


class SimPOTrainer(Trainer):
    r"""
    Initialize SimPOTrainer.

    Args:
        model (`transformers.PreTrainedModel`):
            The model to train, preferably an `AutoModelForSequenceClassification`.
        args (`SimPOConfig`):
            The SimPO config arguments to use for training.
        data_collator (`transformers.DataCollator`):
            The data collator to use for training. If None is specified, the default data collator (`DPODataCollatorWithPadding`) will be used
            which will pad the sequences to the maximum length of the sequences in the batch, given a dataset of paired sequences.
        train_dataset (`datasets.Dataset`):
            The dataset to use for training.
        eval_dataset (`datasets.Dataset`):
            The dataset to use for evaluation.
        tokenizer (`transformers.PreTrainedTokenizerBase`):
            The tokenizer to use for training. This argument is required if you want to use the default data collator.
        model_init (`Callable[[], transformers.PreTrainedModel]`):
            The model initializer to use for training. If None is specified, the default model initializer will be used.
        callbacks (`List[transformers.TrainerCallback]`):
            The callbacks to use for training.
        optimizers (`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`):
            The optimizer and scheduler to use for training.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`):
            The function to use to preprocess the logits before computing the metrics.
        peft_config (`Dict`, defaults to `None`):
            The PEFT configuration to use for training. If you pass a PEFT configuration, the model will be wrapped in a PEFT model.
        compute_metrics (`Callable[[EvalPrediction], Dict]`, *optional*):
            The function to use to compute the metrics. Must take a `EvalPrediction` and return
            a dictionary string to metric values.
    """

    _tag_names = ["trl", "simpo"]

    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        args: Optional[SimPOConfig] = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        peft_config: Optional[Dict] = None,
        compute_metrics: Optional[Callable[[EvalLoopOutput], Dict]] = None,
    ):
        if args.model_init_kwargs is None:
            model_init_kwargs = {}
        elif not isinstance(model, str):
            raise ValueError("You passed model_kwargs to the SimPOTrainer. But your model is already instantiated.")
        else:
            model_init_kwargs = args.model_init_kwargs
            requested_dtype = model_init_kwargs.get("torch_dtype")
            if requested_dtype not in ["auto", None] and not isinstance(requested_dtype, torch.dtype):
                requested_dtype = getattr(torch, requested_dtype)
            model_init_kwargs["torch_dtype"] = requested_dtype

        if isinstance(model, str):
            warnings.warn(
                "You passed a model_id to the SimPOTrainer. This will automatically create an "
                "`AutoModelForCausalLM` or a `PeftModel` (if you passed a `peft_config`) for you."
            )
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)

        # Initialize this variable to False. This helps tracking the case when `peft_module_casting_to_bf16`
        # has been called in order to properly call autocast if needed.
        self._peft_has_been_casted_to_bf16 = False

        if not is_peft_available() and peft_config is not None:
            raise ValueError(
                "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use the PEFT models"
            )
        elif is_peft_available() and peft_config is not None:
            # if model is a peft model and we have a peft_config, we merge and unload it first
            if isinstance(model, PeftModel):
                model = model.merge_and_unload()

            if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
                _support_gc_kwargs = hasattr(
                    args, "gradient_checkpointing_kwargs"
                ) and "gradient_checkpointing_kwargs" in list(
                    inspect.signature(prepare_model_for_kbit_training).parameters
                )

                prepare_model_kwargs = {"use_gradient_checkpointing": args.gradient_checkpointing}

                if _support_gc_kwargs:
                    prepare_model_kwargs["gradient_checkpointing_kwargs"] = args.gradient_checkpointing_kwargs

                model = prepare_model_for_kbit_training(model, **prepare_model_kwargs)
            elif getattr(args, "gradient_checkpointing", False):
                # For backward compatibility with older versions of transformers
                if hasattr(model, "enable_input_require_grads"):
                    model.enable_input_require_grads()
                else:

                    def make_inputs_require_grad(module, input, output):
                        output.requires_grad_(True)

                    model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

            # get peft model with the given config
            model = get_peft_model(model, peft_config)
            if args.bf16 and getattr(model, "is_loaded_in_4bit", False):
                peft_module_casting_to_bf16(model)
                # If args.bf16 we need to explicitly call `generate` with torch amp autocast context manager
                self._peft_has_been_casted_to_bf16 = True

        # For models that use gradient_checkpointing, we need to attach a hook that enables input
        # to explicitly have `requires_grad=True`, otherwise training will either silently
        # fail or completely fail.
        elif getattr(args, "gradient_checkpointing", False):
            # For backward compatibility with older versions of transformers
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        if args.generate_during_eval:
            backend = str(getattr(args, "eval_generation_backend", "checkpoint"))
            if backend not in {"checkpoint", "in_memory"}:
                raise ValueError(
                    "`eval_generation_backend` must be one of {'checkpoint', 'in_memory'}, "
                    f"got {backend!r}."
                )

        if model is not None:
            self.is_encoder_decoder = model.config.is_encoder_decoder
        elif args.is_encoder_decoder is None:
            raise ValueError("When no model is provided, you need to pass the parameter is_encoder_decoder.")
        else:
            self.is_encoder_decoder = args.is_encoder_decoder

        if self.is_encoder_decoder:
            self.decoder_start_token_id = model.config.decoder_start_token_id
            self.pad_token_id = model.config.pad_token_id

        if tokenizer is None:
            raise ValueError("tokenizer must be specified to tokenize a SimPO dataset.")
        if args.max_length is None:
            warnings.warn(
                "`max_length` is not set in the SimPOConfig's init"
                " it will default to `512` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            max_length = 512
        else:
            max_length = args.max_length
        if args.max_prompt_length is None:
            warnings.warn(
                "`max_prompt_length` is not set in the SimPOConfig's init"
                " it will default to `128` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            max_prompt_length = 128
        else:
            max_prompt_length = args.max_prompt_length

        if args.max_target_length is None and self.is_encoder_decoder:
            warnings.warn(
                "When using an encoder decoder architecture, you should set `max_target_length` in the SimPOConfig's init"
                " it will default to `128` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            max_target_length = 128
        else:
            max_target_length = args.max_target_length

        if data_collator is None:
            data_collator = DPODataCollatorWithPadding(
                pad_token_id=tokenizer.pad_token_id,
                label_pad_token_id=args.label_pad_token_id,
                is_encoder_decoder=self.is_encoder_decoder,
            )

            if args.remove_unused_columns:
                args.remove_unused_columns = False
                # warn users
                warnings.warn(
                    "When using DPODataCollatorWithPadding, you should set `remove_unused_columns=False` in your TrainingArguments"
                    " we have set it for you, but you should do it yourself in the future.",
                    UserWarning,
                )

            self.use_dpo_data_collator = True
        else:
            self.use_dpo_data_collator = False

        if args.disable_dropout:
            disable_dropout_in_model(model)

        self.max_length = max_length
        self.generate_during_eval = args.generate_during_eval
        self.label_pad_token_id = args.label_pad_token_id
        self.padding_value = args.padding_value if args.padding_value is not None else tokenizer.pad_token_id
        self.max_prompt_length = max_prompt_length
        self.truncation_mode = args.truncation_mode
        self.max_target_length = max_target_length
        self.tokenizer = tokenizer

        if args.loss_type in ["hinge"] and args.label_smoothing > 0:
            warnings.warn(
                "You are using a loss type that does not support label smoothing. Ignoring label_smoothing parameter."
            )

        self.beta = args.beta
        self.gamma_beta_ratio = args.gamma_beta_ratio
        self.sft_weight = args.sft_weight
        self.label_smoothing = args.label_smoothing
        self.loss_type = args.loss_type

        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        # Compute that only on the main process for faster data processing.
        # see: https://github.com/huggingface/trl/pull/1255
        with PartialState().local_main_process_first():
            # tokenize the dataset
            train_dataset = train_dataset.map(
                self.tokenize_row,
                num_proc=args.dataset_num_proc,
                load_from_cache_file=True,
            )
            if eval_dataset is not None:
                eval_dataset = eval_dataset.map(
                    self.tokenize_row,
                    num_proc=args.dataset_num_proc,
                    load_from_cache_file=True,
                )

        trainer_init_kwargs = {
            "model": model,
            "args": args,
            "data_collator": data_collator,
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset,
            "model_init": model_init,
            "compute_metrics": compute_metrics,
            "callbacks": callbacks,
            "optimizers": optimizers,
            "preprocess_logits_for_metrics": preprocess_logits_for_metrics,
        }
        trainer_init_params = inspect.signature(Trainer.__init__).parameters
        if "tokenizer" in trainer_init_params:
            trainer_init_kwargs["tokenizer"] = tokenizer
        elif "processing_class" in trainer_init_params:
            trainer_init_kwargs["processing_class"] = tokenizer

        super().__init__(**trainer_init_kwargs)

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        if not hasattr(self, "accelerator"):
            raise AttributeError(
                "Your `Trainer` does not have an `accelerator` object. Consider upgrading `transformers`."
            )

    def build_tokenized_answer(self, prompt, answer):
        """Delegates to the canonical implementation (mnpo_scripts.pair_tokenization).

        Kept as a method because subclasses and tests call it; the logic lives in
        one place so the offline precompute path cannot drift from this one.
        """
        return canonical_build_tokenized_answer(self.tokenizer, prompt, answer)

    def tokenize_row(self, feature, model: Optional[Union[PreTrainedModel, nn.Module]] = None) -> Dict:
        """Tokenize one row through the CANONICAL implementation.

        The decoder-only path is ``mnpo_scripts.pair_tokenization.tokenize_preference_pair``
        -- the same function ``mnpo_scripts.precompute`` uses -- so the online
        policy and the offline proximal centre score identical token ids under
        identical attention and label masks, which is what Eq. (22) requires.
        """
        if not self.is_encoder_decoder:
            batch = tokenize_preference_pair(
                self.tokenizer,
                feature["prompt"], feature["chosen"], feature["rejected"],
                max_length=self.max_length,
                max_prompt_length=self.max_prompt_length,
                truncation_mode=self.truncation_mode,
                label_pad_token_id=self.label_pad_token_id,
            )
            return batch
        else:
            # Encoder-decoder path, unchanged and unused by NBPO.
            batch = {}
            prompt, chosen, rejected = feature["prompt"], feature["chosen"], feature["rejected"]
            chosen_tokens = self.tokenizer(
                chosen, truncation=True, max_length=self.max_target_length, add_special_tokens=True
            )
            rejected_tokens = self.tokenizer(
                rejected, truncation=True, max_length=self.max_target_length, add_special_tokens=True
            )
            prompt_tokens = self.tokenizer(
                prompt, truncation=True, max_length=self.max_prompt_length, add_special_tokens=True
            )

            batch["chosen_labels"] = chosen_tokens["input_ids"]
            batch["rejected_labels"] = rejected_tokens["input_ids"]
            batch["prompt_input_ids"] = prompt_tokens["input_ids"]
            batch["prompt_attention_mask"] = prompt_tokens["attention_mask"]

            if model is not None and hasattr(model, "prepare_decoder_input_ids_from_labels"):
                batch["rejected_decoder_input_ids"] = model.prepare_decoder_input_ids_from_labels(
                    labels=torch.tensor(batch["rejected_labels"])
                )
                batch["chosen_decoder_input_ids"] = model.prepare_decoder_input_ids_from_labels(
                    labels=torch.tensor(batch["chosen_labels"])
                )

        return batch

    @staticmethod
    def concatenated_inputs(
        batch: Dict[str, Union[List, torch.LongTensor]],
        is_encoder_decoder: bool = False,
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.LongTensor]:
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).
            is_encoder_decoder: Whether the model is an encoder-decoder model.
            label_pad_token_id: The label pad token id.
            padding_value: The padding value to use for the concatenated inputs_ids.
            device: The device for the concatenated inputs.

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """
        concatenated_batch = {}

        if is_encoder_decoder:
            max_length = max(batch["chosen_labels"].shape[1], batch["rejected_labels"].shape[1])
        else:
            max_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])

        for k in batch:
            if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
                if "labels" in k or is_encoder_decoder:
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                else:
                    continue
                concatenated_key = k.replace("chosen", "concatenated")
                concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
        for k in batch:
            if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
                if "labels" in k or is_encoder_decoder:
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                else:
                    continue
                concatenated_key = k.replace("rejected", "concatenated")
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key],
                        pad_to_length(batch[k], max_length, pad_value=pad_value),
                    ),
                    dim=0,
                ).to(device=device)

        if is_encoder_decoder:
            concatenated_batch["concatenated_input_ids"] = batch["prompt_input_ids"].repeat(2, 1).to(device=device)
            concatenated_batch["concatenated_attention_mask"] = (
                batch["prompt_attention_mask"].repeat(2, 1).to(device=device)
            )

        return concatenated_batch

    def simpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the SimPO loss for a batch of policy model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the SimPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        pi_logratios = pi_logratios.to(self.accelerator.device)
        logits = pi_logratios - self.gamma_beta_ratio

        if self.loss_type == "sigmoid":
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)
        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge']"
            )

        chosen_rewards = self.beta * policy_chosen_logps.to(self.accelerator.device).detach()
        rejected_rewards = self.beta * policy_rejected_logps.to(self.accelerator.device).detach()

        return losses, chosen_rewards, rejected_rewards

    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = self.concatenated_inputs(
            batch,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=self.accelerator.device,
        )
        len_chosen = batch["chosen_labels"].shape[0]

        model_kwargs = (
            {
                "labels": concatenated_batch["concatenated_labels"],
                "decoder_input_ids": concatenated_batch.pop("concatenated_decoder_input_ids", None),
            }
            if self.is_encoder_decoder
            else {}
        )

        all_logits = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            use_cache=False,
            **model_kwargs,
        ).logits

        all_logps = self.get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            # "mean" (the historical default for every legacy loss) length-normalizes;
            # loss_type=nbpo requires "sum" (sequence-sum response log-probabilities),
            # enforced by mnpo_scripts.mnpo_trainer.validate_nbpo_args.
            average_log_prob=(str(getattr(self.args, "logp_reduction", "mean")).lower() == "mean"),
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        chosen_labels = concatenated_batch["concatenated_labels"][:len_chosen]

        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_labels)

    @staticmethod
    def get_batch_logps(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = True,
        label_pad_token_id: int = -100,
        is_encoder_decoder: bool = False,
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.
            label_pad_token_id: The label pad token id.
            is_encoder_decoder: Whether the model is an encoder-decoder model.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        if not is_encoder_decoder:
            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]
        loss_mask = labels != label_pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == label_pad_token_id] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the SimPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}
        prefix = "eval_" if train_eval == "eval" else ""

        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            chosen_labels,
        ) = self.concatenated_forward(model, batch)

        losses, chosen_rewards, rejected_rewards = self.simpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
        )

        loss = losses.mean()

        if self.sft_weight > 0.0:
            if not self.is_encoder_decoder:
                policy_chosen_logits = policy_chosen_logits[..., :-1, :].contiguous()
                chosen_labels = chosen_labels[..., 1:].clone()
            loss_func = nn.CrossEntropyLoss()
            sft_loss = loss_func(policy_chosen_logits.view(-1, policy_chosen_logits.shape[-1]), chosen_labels.view(-1))
            loss = self.sft_weight * sft_loss + loss
            metrics[f"{prefix}sft_loss"] = sft_loss.detach().cpu()
        
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().cpu()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().cpu()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().cpu()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().cpu()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().mean().cpu()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().mean().cpu()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().mean().cpu()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().mean().cpu()

        return loss, metrics

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
        num_items_in_batch: Optional[int] = None,  # add this to adapt to transformers update
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if not self.use_dpo_data_collator:
            warnings.warn(
                "compute_loss is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )

        compute_loss_context_manager = torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext

        with compute_loss_context_manager():
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train")

        # force log the metrics
        self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return (loss, metrics)
        return loss

    def generate_samples_for_eval(self, model, batch: Dict[str, torch.LongTensor]) -> Tuple[str, str]:
        """Generate samples from the model and reference model for the given batch of inputs."""

        # If one uses `generate_during_eval` with peft + bf16, we need to explicitly call generate with
        # the torch cuda amp context manager as some hidden states are silently casted to full precision.
        generate_context_manager = nullcontext if not self._peft_has_been_casted_to_bf16 else torch.cuda.amp.autocast

        with generate_context_manager():
            generation_kwargs = {
                "input_ids": batch["prompt_input_ids"],
                "attention_mask": batch["prompt_attention_mask"],
                "max_new_tokens": int(getattr(self.args, "eval_generation_max_new_tokens", 256)),
                "do_sample": bool(getattr(self.args, "eval_generation_do_sample", False)),
                "pad_token_id": self.tokenizer.pad_token_id,
            }
            if generation_kwargs["do_sample"]:
                generation_kwargs["temperature"] = float(getattr(self.args, "eval_generation_temperature", 0.7))
                generation_kwargs["top_p"] = float(getattr(self.args, "eval_generation_top_p", 0.9))
            generation_model = getattr(self, "model_wrapped", None) or model
            with unwrap_model_for_generation(generation_model, self.accelerator) as unwrapped_model:
                policy_output = unwrapped_model.generate(
                    **generation_kwargs,
                )

        prompt_width = batch["prompt_input_ids"].shape[1]
        policy_output_decoded = self.tokenizer.batch_decode(
            policy_output[:, prompt_width:],
            skip_special_tokens=True,
        )

        return policy_output_decoded

    @staticmethod
    def _eval_generation_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False)

    @staticmethod
    def _max_token_run(text: str) -> int:
        best = 0
        prev = None
        cur = 0
        for token in str(text).lower().split():
            if token == prev:
                cur += 1
            else:
                prev = token
                cur = 1
            best = max(best, cur)
        return best

    @staticmethod
    def _safe_path_name(value: str) -> str:
        safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
        return safe.strip("._") or "run"

    def _eval_generation_step_dir(self) -> Path:
        root_arg = getattr(self.args, "eval_generation_output_dir", None)
        root = Path(root_arg) if root_arg else Path(self.args.output_dir) / "eval_generations"
        run_name = self._safe_path_name(str(getattr(self.args, "run_name", None) or "run"))
        return root / run_name / f"step-{int(self.state.global_step):08d}"

    def _select_eval_generation_records(self, dataset: Dataset) -> Tuple[List[Dict[str, Any]], bool]:
        num_samples = len(dataset)
        sample_count = min(int(getattr(self.args, "eval_generation_samples", 5)), num_samples)
        rng = random.Random(int(getattr(self.args, "eval_generation_seed", 42)))
        indices = sorted(rng.sample(range(num_samples), k=sample_count))

        records = []
        formatted_prompts = True
        for eval_index, dataset_index in enumerate(indices):
            example = dataset[int(dataset_index)]
            formatted_prompts = formatted_prompts and bool(example.get("chat_template_applied", False))
            records.append(
                {
                    "prompt_id": example.get("prompt_id", dataset_index),
                    "eval_index": eval_index,
                    "dataset_index": int(dataset_index),
                    "prompt": self._eval_generation_text(example.get("prompt", "")),
                    "chosen": self._eval_generation_text(example.get("chosen", "")),
                    "rejected": self._eval_generation_text(example.get("rejected", "")),
                }
            )
        return records, formatted_prompts

    def _write_eval_generation_prompts(self, records: List[Dict[str, Any]], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _write_eval_generation_csv(self, records: List[Dict[str, Any]], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "step",
            "eval_index",
            "dataset_index",
            "prompt_id",
            "prompt",
            "response",
            "response_chars",
            "max_token_run",
            "model_path",
            "chosen",
            "rejected",
        ]
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                writer.writerow({name: record.get(name, "") for name in fieldnames})

    def _print_eval_generation_records(self, records: List[Dict[str, Any]], jsonl_path: Path, csv_path: Path) -> None:
        max_chars = int(getattr(self.args, "eval_generation_print_max_chars", 1200))
        print(
            f"\n=== Eval policy generations | step={self.state.global_step} | "
            f"jsonl={jsonl_path} | csv={csv_path} ===",
            flush=True,
        )
        for idx, record in enumerate(records, start=1):
            prompt = str(record.get("prompt", ""))
            response = str(record.get("response", ""))
            if len(prompt) > max_chars:
                prompt = prompt[:max_chars] + "..."
            if len(response) > max_chars:
                response = response[:max_chars] + "..."
            print(
                f"\n[{idx}/{len(records)}] prompt_id={record.get('prompt_id')} "
                f"dataset_index={record.get('dataset_index')} "
                f"max_token_run={record.get('max_token_run')}\n"
                f"PROMPT:\n{prompt}\n"
                f"POLICY:\n{response}",
                flush=True,
            )
        print("=== End eval policy generations ===\n", flush=True)

    def _save_eval_generation_records(self, records: List[Dict[str, Any]], jsonl_path: Path, csv_path: Path) -> None:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with jsonl_path.open("w") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._write_eval_generation_csv(records, csv_path)
        self._print_eval_generation_records(records, jsonl_path, csv_path)

    def _run_checkpoint_eval_generation(
        self,
        prompt_records: List[Dict[str, Any]],
        formatted_prompts: bool,
        step_dir: Path,
    ) -> None:
        prompt_path = step_dir / "prompts.jsonl"
        raw_jsonl_path = step_dir / "policy_generations.raw.jsonl"
        jsonl_path = step_dir / "policy_generations.jsonl"
        csv_path = step_dir / "policy_generations.csv"
        snapshot_dir = step_dir / "policy_snapshot"
        done_path = step_dir / ".generation_done"
        error_path = step_dir / ".generation_error"

        if self.accelerator.is_main_process:
            step_dir.mkdir(parents=True, exist_ok=True)
            if snapshot_dir.exists():
                shutil.rmtree(snapshot_dir)
            for marker_path in (done_path, error_path):
                marker_path.unlink(missing_ok=True)
            self._write_eval_generation_prompts(prompt_records, prompt_path)

        # DeepSpeed ZeRO-3 model saving is collective.  Every rank must enter
        # save_model before rank 0 can run the standalone generation process.
        self.accelerator.wait_for_everyone()
        self.save_model(str(snapshot_dir))
        self.accelerator.wait_for_everyone()

        if not self.accelerator.is_main_process:
            self.accelerator.wait_for_everyone()
            if error_path.exists():
                raise RuntimeError(error_path.read_text(errors="replace"))
            return

        script_path = Path(__file__).resolve().parent / "generate_checkpoint_samples.py"
        cmd = [
            sys.executable,
            str(script_path),
            "--model-path",
            str(snapshot_dir),
            "--prompts-file",
            str(prompt_path),
            "--output",
            str(raw_jsonl_path),
            "--num-prompts",
            str(len(prompt_records)),
            "--max-new-tokens",
            str(int(getattr(self.args, "eval_generation_max_new_tokens", 256))),
            "--device",
            str(getattr(self.args, "eval_generation_device", "cuda")),
            "--dtype",
            str(getattr(self.args, "eval_generation_dtype", "bfloat16")),
        ]
        if bool(getattr(self.args, "eval_generation_local_files_only", True)):
            cmd.append("--local-files-only")
        if formatted_prompts:
            cmd.append("--formatted-prompts")
        if bool(getattr(self.args, "eval_generation_do_sample", False)):
            cmd.extend(
                [
                    "--do-sample",
                    "--temperature",
                    str(float(getattr(self.args, "eval_generation_temperature", 0.7))),
                    "--top-p",
                    str(float(getattr(self.args, "eval_generation_top_p", 0.9))),
                    "--top-k",
                    str(int(getattr(self.args, "eval_generation_top_k", 20))),
                ]
            )

        env = os.environ.copy()
        eval_cuda_visible_devices = getattr(self.args, "eval_generation_cuda_visible_devices", None)
        if eval_cuda_visible_devices:
            env["CUDA_VISIBLE_DEVICES"] = str(eval_cuda_visible_devices)

        try:
            result = subprocess.run(cmd, env=env, text=True, capture_output=True)
            if result.stdout:
                print(result.stdout, end="", flush=True)
            if result.stderr:
                print(result.stderr, end="", flush=True)
            if result.returncode != 0:
                raise RuntimeError(f"Eval generation subprocess failed with exit code {result.returncode}: {' '.join(cmd)}")

            generated_records = []
            prompt_by_id = {record["prompt_id"]: record for record in prompt_records}
            for line in raw_jsonl_path.read_text(errors="replace").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                prompt_meta = prompt_by_id.get(record.get("prompt_id"), {})
                record.update(
                    {
                        "step": int(self.state.global_step),
                        "eval_index": prompt_meta.get("eval_index"),
                        "dataset_index": prompt_meta.get("dataset_index"),
                        "chosen": prompt_meta.get("chosen", ""),
                        "rejected": prompt_meta.get("rejected", ""),
                    }
                )
                generated_records.append(record)

            self._save_eval_generation_records(generated_records, jsonl_path, csv_path)
            done_path.write_text("ok\n")
        except Exception as exc:
            error_path.write_text(f"{type(exc).__name__}: {exc}\n")

        if not bool(getattr(self.args, "eval_generation_keep_snapshot", False)):
            shutil.rmtree(snapshot_dir, ignore_errors=True)

        self.accelerator.wait_for_everyone()
        if error_path.exists():
            raise RuntimeError(error_path.read_text(errors="replace"))

    def _run_in_memory_eval_generation(
        self,
        prompt_records: List[Dict[str, Any]],
        random_batch_dataset: Dataset,
        step_dir: Path,
    ) -> None:
        random_batch = self.data_collator(random_batch_dataset)
        random_batch = self._prepare_inputs(random_batch)
        policy_output_decoded = self.generate_samples_for_eval(self.model, random_batch)
        jsonl_path = step_dir / "policy_generations.jsonl"
        csv_path = step_dir / "policy_generations.csv"
        records = []
        for prompt_record, response in zip(prompt_records, policy_output_decoded):
            records.append(
                {
                    **prompt_record,
                    "step": int(self.state.global_step),
                    "response": response,
                    "response_chars": len(response),
                    "max_token_run": self._max_token_run(response),
                    "model_path": "in_memory_policy",
                }
            )
        self._save_eval_generation_records(records, jsonl_path, csv_path)

    def _run_local_eval_generation(self, dataloader: DataLoader) -> None:
        if len(dataloader.dataset) == 0:
            return

        step_dir = self._eval_generation_step_dir()

        backend = str(getattr(self.args, "eval_generation_backend", "checkpoint"))
        if backend == "checkpoint":
            prompt_records: List[Dict[str, Any]] = []
            formatted_prompts = True
            if self.accelerator.is_main_process:
                prompt_records, formatted_prompts = self._select_eval_generation_records(dataloader.dataset)
            self._run_checkpoint_eval_generation(prompt_records, formatted_prompts, step_dir)
        elif backend == "in_memory":
            if self.accelerator.is_main_process:
                prompt_records, _ = self._select_eval_generation_records(dataloader.dataset)
                indices = [record["dataset_index"] for record in prompt_records]
                random_batch_dataset = dataloader.dataset.select(indices)
                step_dir.mkdir(parents=True, exist_ok=True)
                self._run_in_memory_eval_generation(prompt_records, random_batch_dataset, step_dir)
            self.accelerator.wait_for_everyone()
        else:
            raise ValueError(f"Unsupported eval generation backend: {backend}")

    def get_batch_samples(self, epoch_iterator, num_batches, device):
        """
        Adapter for HF Trainer's internal API.

        This is used by `Trainer._inner_training_loop`, we don't want to change
        its behavior, so we just defer to the parent implementation.
        """
        return super().get_batch_samples(epoch_iterator, num_batches, device)

    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        if not self.use_dpo_data_collator:
            warnings.warn(
                "prediction_step is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )
        if ignore_keys is None:
            if hasattr(model, "config"):
                ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        prediction_context_manager = torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext

        with torch.no_grad(), prediction_context_manager():
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="eval")

        # force log the metrics
        self.store_metrics(metrics, train_eval="eval")

        if prediction_loss_only:
            return (loss.detach(), None, None)

        # logits for the chosen and rejected samples from model
        logits_dict = {
            "eval_logits/chosen": metrics["eval_logits/chosen"],
            "eval_logits/rejected": metrics["eval_logits/rejected"],
        }
        logits = tuple(v.unsqueeze(dim=0) for k, v in logits_dict.items() if k not in ignore_keys)
        logits = torch.stack(logits).mean(axis=1).to(self.accelerator.device)
        labels = torch.zeros(logits.shape[0], device=self.accelerator.device)

        return (loss.detach(), logits, labels)

    def store_metrics(self, metrics: Dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Overriding built-in evaluation loop to store metrics for each batch.
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """

        # Sample from the current policy if requested.  The default checkpoint
        # backend avoids the ZeRO in-memory generation path that produced
        # repeated-token artifacts during HT-MNPO smoke tests.
        if self.generate_during_eval:
            self._run_local_eval_generation(dataloader)

        # Base evaluation
        initial_output = super().evaluation_loop(
            dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix
        )

        return initial_output

    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        """
        Log `logs` on the various objects watching training, including stored metrics.

        """
        train_eval = "train" if "loss" in logs else "eval"

        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]

        return super().log(logs, *args, **kwargs)

    @wraps(Trainer.push_to_hub)
    def push_to_hub(self, commit_message: Optional[str] = "End of training", blocking: bool = True, **kwargs) -> str:
        """
        Overwrite the `push_to_hub` method in order to force-add the tag "simpo" when pushing the
        model on the Hub. Please refer to `~transformers.Trainer.push_to_hub` for more details.
        """

        return super().push_to_hub(commit_message=commit_message, blocking=blocking, **kwargs)
