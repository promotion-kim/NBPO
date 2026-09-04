from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from datasets import Dataset
from torch.utils.data import DataLoader
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput
from trl import DPOTrainer, DPOConfig
from accelerate.utils import is_deepspeed_available, tqdm

from mnpo_scripts.pair_tokenization import tokenize_preference_pair

@dataclass
class PreferenceDataCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[PreTrainedModel] = None
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_prompt_length: Optional[int] = None
    label_pad_token_id: int = -100
    padding_value: int = 0
    truncation_mode: str = "keep_end"
    is_encoder_decoder: Optional[bool] = False
    max_target_length: Optional[int] = None
    mask_prompt: Optional[bool] = False

    def tokenize_batch_element(
        self,
        prompt: str,
        chosen: str,
        rejected: str,
    ) -> Dict:
        """Tokenize one preference pair through the CANONICAL implementation.

        This used to be a second, divergent copy of the trainer's tokenization:
        it tokenized prompt/chosen/rejected separately (so a tokenizer that
        merges across the prompt/response boundary produced different ids than
        training), zeroed the attention mask at EOS positions inside the prompt,
        computed a chosen attention mask and discarded it, and appended EOS
        unconditionally. Every one of those made the offline pi_t log-probability
        incomparable with the online one, which is exactly what Eq. (22)
        subtracts. There is now one implementation, in
        ``mnpo_scripts.pair_tokenization``, and both paths call it.
        """
        if self.is_encoder_decoder:
            raise NotImplementedError
        if self.mask_prompt:
            # The canonical path masks the prompt in LABELS, never by removing it
            # from the attention mask. Honouring mask_prompt would hide real
            # context from the model on this path only, so it is refused rather
            # than silently ignored.
            raise ValueError(
                "mask_prompt is not supported by the canonical tokenization: prompt "
                "exclusion belongs in labels (label_pad_token_id), not in attention_mask")
        return tokenize_preference_pair(
            self.tokenizer, prompt, chosen, rejected,
            max_length=self.max_length,
            max_prompt_length=self.max_prompt_length,
            truncation_mode=self.truncation_mode,
            label_pad_token_id=self.label_pad_token_id,
        )

    def collate(self, batch):
        # first, pad everything to the same length
        padded_batch = {}
        # 'chosen_input_ids', 'chosen_attention_mask', 'chosen_labels', 'rejected_input_ids', 'rejected_attention_mask', 'rejected_labels', 'prompt_input_ids', 'prompt_attention_mask'
        # 'prompt', 'chosen', 'rejected', 'chosen_response_only', 'rejected_response_only'
        for k in batch[0].keys():
            if k.endswith("_input_ids") or k.endswith("_attention_mask") or k.endswith("_labels"):
                if self.is_encoder_decoder:
                    to_pad = [torch.LongTensor(ex[k]) for ex in batch]

                    if (k.startswith("prompt")) and (k.endswith("input_ids")):
                        padding_value = self.tokenizer.pad_token_id
                    elif k.endswith("_attention_mask"):
                        padding_value = 0
                    elif (k.startswith("chosen")) or (k.startswith("rejected")) or ("decoder" in k):
                        padding_value = self.label_pad_token_id
                    else:
                        raise ValueError(f"Unexpected key in batch '{k}'")
                    padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                else:
                    # adapted from https://stackoverflow.com/questions/73256206
                    if "prompt" in k:
                        to_pad = [torch.LongTensor(ex[k][::-1]) for ex in batch]
                    else:
                        to_pad = [torch.LongTensor(ex[k]) for ex in batch]
                    if k.endswith("_input_ids"):
                        padding_value = self.tokenizer.pad_token_id
                    elif k.endswith("_labels"):
                        padding_value = self.label_pad_token_id
                    elif k.endswith("_attention_mask"):
                        padding_value = self.padding_value
                    else:
                        raise ValueError(f"Unexpected key in batch '{k}'")

                    padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                    # for the prompt, flip back so padding is on left side
                    if "prompt" in k:
                        padded_batch[k] = padded_batch[k].flip(dims=[1])
            else:
                padded_batch[k] = [ex[k] for ex in batch]

        return padded_batch

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        tokenized_batch = []

        for feature in features:
            prompt = feature["prompt"]
            chosen = feature["chosen"]
            rejected = feature["rejected"]

            batch_element = self.tokenize_batch_element(prompt, chosen, rejected)
            tokenized_batch.append(batch_element)

        # return collated batch
        return self.collate(tokenized_batch)

class PreComputer(DPOTrainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        ref_model: Optional[Union[PreTrainedModel, nn.Module]] = None,
        args: DPOConfig = None,
        data_collator: Optional[DataCollator] = None,
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        truncation_mode: str = "keep_end",
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        max_target_length: Optional[int] = None,
        peft_config: Optional[Dict] = None,
        is_encoder_decoder: Optional[bool] = None,
        disable_dropout: bool = True,
        generate_during_eval: bool = False,
        compute_metrics: Optional[Callable[[EvalLoopOutput], Dict]] = None,
        mask_prompt: Optional[bool] = False,
        len_penalty: float = 0,
    ):

        if data_collator is None:
            # 2048, 1000, -100, 0, keep_end, None, False
            data_collator = PreferenceDataCollatorWithPadding(
                tokenizer,
                max_length=args.max_length,
                max_prompt_length=args.max_prompt_length,
                label_pad_token_id=label_pad_token_id,
                padding_value=padding_value,
                truncation_mode=truncation_mode,
                is_encoder_decoder=False,
                max_target_length=max_target_length,
                mask_prompt=mask_prompt,
            )
        super().__init__(
            model=model,
            ref_model=ref_model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator
        )
        self.use_dpo_data_collator = True
        # self.len_penalty = len_penalty
        self.precompute_ref_log_probs = True
        self._precomputed_train_ref_log_probs = True

    def precompute(self):
        dataloader_params = {
                "batch_size": self.args.per_device_train_batch_size,
                "collate_fn": self.data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "shuffle": False,
        }

        # prepare dataloader
        data_loader = self.accelerator.prepare(DataLoader(self.train_dataset, **dataloader_params))

        reference_chosen_logps = []
        reference_rejected_logps = []
        for padded_batch in tqdm(iterable=data_loader, desc="Train dataset reference log probs"):
            reference_chosen_logp, reference_rejected_logp = self.compute_reference_log_probs(padded_batch)
            reference_chosen_logp, reference_rejected_logp = self.accelerator.gather_for_metrics(
                (reference_chosen_logp, reference_rejected_logp)
            )
            reference_chosen_logps.append(reference_chosen_logp.cpu())
            reference_rejected_logps.append(reference_rejected_logp.cpu())

        all_reference_chosen_logps = torch.cat(reference_chosen_logps).float().numpy()
        all_reference_rejected_logps = torch.cat(reference_rejected_logps).float().numpy()

        print(all_reference_chosen_logps.shape, all_reference_rejected_logps.shape)
        return all_reference_chosen_logps, all_reference_rejected_logps