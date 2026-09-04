import json
import logging
import inspect
import os
import sys
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union, Tuple

import torch
import transformers
if os.environ.get("MNPO_DISABLE_APEX", "").lower() in {"1", "true", "yes"}:
    import transformers.utils.import_utils as _transformers_import_utils

    if hasattr(_transformers_import_utils, "_apex_available"):
        _transformers_import_utils._apex_available = False
from accelerate import Accelerator
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    HfArgumentParser,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)
try:
    from trl.trainer.utils import DPODataCollatorWithPadding, pad_to_length
except ImportError:
    DPODataCollatorWithPadding = None

    def pad_to_length(tensor: torch.Tensor, length: int, pad_value: int, dim: int = -1) -> torch.Tensor:
        if tensor.size(dim) >= length:
            return tensor
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat(
            [tensor, torch.full(pad_size, pad_value, dtype=tensor.dtype, device=tensor.device)],
            dim=dim,
        )
import torch.nn as nn
from mnpo_scripts.precompute_trainer import PreferenceDataCollatorWithPadding

logger = logging.getLogger(__name__)


def _apply_chat_template_non_thinking(
    tokenizer: PreTrainedTokenizerBase,
    messages: List[Dict[str, str]],
    **kwargs: Any,
) -> str:
    signature = inspect.signature(tokenizer.apply_chat_template)
    supports_extra_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if "enable_thinking" in signature.parameters or supports_extra_kwargs:
        kwargs.setdefault("enable_thinking", False)
    return tokenizer.apply_chat_template(messages, **kwargs)


def is_openai_format(messages: Any) -> bool:
    return (
        isinstance(messages, list)
        and all(isinstance(message, dict) for message in messages)
        and all("role" in message and "content" in message for message in messages)
    )


@dataclass
class ScriptArguments:
    """
    Arguments for the precompute script.
    """

    # loss parameters
    beta: Optional[float] = field(default=0.005, metadata={"help": "beta parameter for DPO loss"})

    # model parameters
    model_name_or_path: Optional[str] = field(
        default="sshleifer/tiny-gpt2",
        metadata={"help": "base model name or local path"},
    )
    ref_model: Optional[str] = field(
        default="",
        metadata={"help": "reference/SFT model name or local path"},
    )
    last_model: Optional[str] = field(
        default="",
        metadata={"help": "last-iteration model name or path (if any)"},
    )

    # data I/O
    train_dir: Optional[str] = field(
        default="./data/uf_split0_responses_K8_reward.json",
        metadata={"help": "train dataset path or HF hub id"},
    )
    # Backward-compatible alias: either pass --test_dir or --eval_dir
    test_dir: Optional[str] = field(
        default=None,
        metadata={"help": "test dataset path or HF hub id (optional; if provided, will produce a DatasetDict with 'train' and 'test')"},
    )
    eval_dir: Optional[str] = field(
        default=None,
        metadata={"help": "alias of --test_dir for backward compatibility"},
    )

    # optimization settings (some may be unused in this precompute phase but kept for parity)
    learning_rate: Optional[float] = field(default=5e-7, metadata={"help": "optimizer learning rate"})
    lr_scheduler_type: Optional[str] = field(
        default="constant_with_warmup", metadata={"help": "lr scheduler type"}
    )
    warmup_steps: Optional[int] = field(default=100, metadata={"help": "number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.01, metadata={"help": "weight decay"})
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "optimizer type"})

    per_device_train_batch_size: Optional[int] = field(default=1, metadata={"help": "train batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "eval batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(
        default=16, metadata={"help": "gradient accumulation steps"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "use gradient checkpointing"}
    )

    eos_padding: Optional[bool] = field(default=True, metadata={"help": "pad with eos token"})
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "LoRA alpha"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "LoRA dropout"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "LoRA rank"})

    margin_scale: Optional[float] = field(default=1.0, metadata={"help": "margin scale"})

    max_prompt_length: Optional[int] = field(default=1000, metadata={"help": "maximum prompt length"})
    max_length: Optional[int] = field(default=2048, metadata={"help": "maximum sequence length"})
    max_steps: Optional[int] = field(default=-1, metadata={"help": "max number of training steps"})
    num_train_epochs: Optional[int] = field(default=2, metadata={"help": "max number of training epochs"})
    logging_steps: Optional[int] = field(default=2, metadata={"help": "logging frequency"})
    save_strategy: Optional[str] = field(default="epoch", metadata={"help": "saving strategy"})
    save_steps: Optional[int] = field(default=50000, metadata={"help": "saving frequency"})
    eval_steps: Optional[int] = field(default=100, metadata={"help": "evaluation frequency"})
    run_name: Optional[str] = field(default="dpo_soft", metadata={"help": "run name"})
    loss_type: Optional[str] = field(default="sigmoid", metadata={"help": "loss type"})
    output_dir: Optional[str] = field(default="./dpo_soft", metadata={"help": "output directory"})
    log_freq: Optional[int] = field(default=1, metadata={"help": "logging frequency"})

    # instrumentation
    sanity_check: Optional[bool] = field(default=False, metadata={"help": "train on a small subset (e.g., 100 samples)"})
    max_training_samples: Optional[int] = field(default=-1, metadata={"help": "maximum sample size"})
    choose_type: Optional[str] = field(default="max_min", metadata={"help": "choose type"})

    report_to: Optional[str] = field(
        default="none",
        metadata={
            "help": 'Reporting destinations: "azure_ml", "comet_ml", "mlflow", "neptune", "tensorboard", "clearml", "wandb", "all", or "none".'
        },
    )

    # distributed training debug flag
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers; see https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )
    eot_token: Optional[str] = field(default="", metadata={"help": "end-of-text token override"})
    truncation_mode: str = field(
        default="keep_end",
        metadata={"help": "keep_start|keep_end; MUST equal the trainer's "
                          "SimPOConfig.truncation_mode, or pi_t and pi truncate "
                          "differently and Eq. (22) subtracts logps of different tokens"},
    )
    mask_prompt: Optional[bool] = field(default=False, metadata={"help": "whether to mask prompt tokens"})
    len_penalty: Optional[float] = field(default=0, metadata={"help": "length penalty"})
    history_paths: Optional[List[str]] = field(default_factory=list, metadata={"help": "list of historical model paths"})
    max_history_t: Optional[int] = field(default=2, metadata={"help": "maximum history length"})
    cache_dir: Optional[str] = field(default=None, metadata={"help": "cache directory for models and datasets"})
    ronpo_target_mode: Optional[str] = field(
        default="score_diff_sign",
        metadata={
            "help": (
                "How to add the RONPO relative label column. "
                "'score_diff_sign' matches chosen/rejected to all_generated_responses and stores sign(score_chosen-score_rejected); "
                "'ordered' stores +1 for every pair; 'none' skips target creation."
            )
        },
    )
    ronpo_target_column: Optional[str] = field(
        default="ronpo_target",
        metadata={"help": "Column name for the RONPO relative label z_y - z_y_prime."},
    )
    ronpo_tie_threshold: Optional[float] = field(
        default=0.0,
        metadata={"help": "Absolute score gap below which score_diff_sign emits a neutral 0 target."},
    )
    apply_chat_template: Optional[bool] = field(
        default=True,
        metadata={
            "help": (
                "Format prompt/chosen/rejected with the model chat template before "
                "precomputing logps. Disable only for legacy preformatted datasets."
            )
        },
    )
    auto_insert_empty_system_msg: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Insert an empty system message before applying the chat template. "
                "Defaults to false to match on_policy_data_gen.decode for Qwen."
            )
        },
    )
    solver_artifact_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional solution.json of the NBPO dual solve that produced the "
                          "pair targets; its sha256 is recorded in precompute_meta.json."},
    )
    logp_reduction: Optional[str] = field(
        default="mean",
        metadata={
            "help": (
                "Per-response log-probability reduction: 'mean' (token average; the "
                "historical default consumed by every legacy loss) or 'sum' (sequence "
                "sum over non-masked response tokens; REQUIRED by loss_type=nbpo). "
                "Recorded in precompute_meta.json and validated at training time."
            )
        },
    )


def get_batch_logps(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
    average_log_prob: bool = True,
    label_pad_token_id: int = -100,
    is_encoder_decoder: bool = False,
) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Shape (batch, seq_len, vocab)
        labels: Shape (batch, seq_len); tokens == label_pad_token_id are ignored
        average_log_prob: average per non-masked token if True, else sum
        label_pad_token_id: label pad id
        is_encoder_decoder: whether the model is encoder-decoder

    Returns:
        Shape (batch,)
    """
    if logits.shape[:-1] != labels.shape:
        raise ValueError("Logits (batch, seq_len) and labels must have the same shape on those dims.")

    if not is_encoder_decoder:
        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
    loss_mask = labels != label_pad_token_id

    # replace pad labels with a dummy id (ignored by loss via mask)
    labels[labels == label_pad_token_id] = 0

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)


def concatenated_inputs(
    batch: Dict[str, Union[List, torch.LongTensor]],
    padding_value: int = 0,
    label_pad_token_id: int = -100,
) -> Dict[str, torch.LongTensor]:
    """
    Take a batch with separate chosen/rejected tensors and concatenate them.
    """
    concatenated_batch = {}
    max_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])

    for k in batch:
        if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
            pad_value = label_pad_token_id if "labels" in k else padding_value
            concatenated_key = k.replace("chosen", "concatenated")
            concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)

    for k in batch:
        if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
            pad_value = label_pad_token_id if "labels" in k else padding_value
            concatenated_key = k.replace("rejected", "concatenated")
            concatenated_batch[concatenated_key] = torch.cat(
                (
                    concatenated_batch[concatenated_key],
                    pad_to_length(batch[k], max_length, pad_value=pad_value),
                ),
                dim=0,
            )

    return concatenated_batch


def concatenated_forward(
    model: nn.Module, batch: Dict, average_log_prob: bool = True
) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    """
    Core forward pass consistent with DPO Trainer behavior.
    Takes a batch processed by DPODataCollatorWithPadding and returns
    chosen/rejected log-probabilities (token-averaged by default; pass
    average_log_prob=False for sequence sums, as required by loss_type=nbpo).
    """
    # 1) concatenate chosen/rejected
    concatenated_batch = concatenated_inputs(batch)

    # 2) prepare model inputs
    input_ids = concatenated_batch["concatenated_input_ids"]
    labels = concatenated_batch["concatenated_labels"]
    attention_mask = concatenated_batch["concatenated_attention_mask"]

    # 3) forward
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    # 4) compute logps
    all_logps = get_batch_logps(logits, labels, average_log_prob=average_log_prob)

    # 5) split back
    bsz = batch["chosen_labels"].shape[0]
    chosen_logps = all_logps[:bsz]
    rejected_logps = all_logps[bsz:]

    return chosen_logps, rejected_logps


def transform_chat_to_str(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert 'chosen' and 'rejected' fields from a list of dicts to a single string.
    Assumes desired content is in the last message of the list.
    """
    if isinstance(example.get("chosen"), list) and example["chosen"]:
        example["chosen"] = example["chosen"][-1]["content"]
    if isinstance(example.get("rejected"), list) and example["rejected"]:
        example["rejected"] = example["rejected"][-1]["content"]
    return example


def _maybe_insert_empty_system(messages: List[Dict[str, str]], tokenizer: PreTrainedTokenizerBase) -> List[Dict[str, str]]:
    if messages and messages[0].get("role") == "system":
        return messages
    chat_template = tokenizer.chat_template or getattr(tokenizer, "default_chat_template", None) or ""
    if "system" in chat_template or "<|im_start|>" in chat_template:
        return [{"role": "system", "content": ""}] + messages
    return messages


def _split_preference_example(example: Dict[str, Any]) -> Tuple[List[Dict[str, str]], str, str]:
    prompt = example.get("prompt")
    chosen = example.get("chosen")
    rejected = example.get("rejected")

    if is_openai_format(chosen) and is_openai_format(rejected):
        if is_openai_format(prompt):
            prompt_messages = list(prompt)
        else:
            prompt_messages = list(chosen[:-1])
        if not chosen or not rejected:
            raise ValueError("chosen/rejected messages must be non-empty")
        chosen_response = chosen[-1]["content"]
        rejected_response = rejected[-1]["content"]
        return prompt_messages, chosen_response, rejected_response

    if isinstance(prompt, str) and isinstance(chosen, str) and isinstance(rejected, str):
        return [{"role": "user", "content": prompt}], chosen, rejected

    raise ValueError(
        "Expected either OpenAI-format chosen/rejected messages or string prompt/chosen/rejected fields."
    )


def _assistant_suffix_from_template(
    tokenizer: PreTrainedTokenizerBase,
    prompt_messages: List[Dict[str, str]],
    response: str,
) -> str:
    prompt_text = _apply_chat_template_non_thinking(
        tokenizer,
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = _apply_chat_template_non_thinking(
        tokenizer,
        prompt_messages + [{"role": "assistant", "content": response}],
        tokenize=False,
        add_generation_prompt=False,
    )
    if not full_text.startswith(prompt_text):
        raise ValueError(
            "Chat template full conversation does not start with the generation prompt prefix. "
            "Cannot split assistant suffix safely."
        )
    return full_text[len(prompt_text):].rstrip()


def apply_preference_chat_template(
    example: Dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    auto_insert_empty_system_msg: bool = False,
) -> Dict[str, Any]:
    prompt_messages, chosen_response, rejected_response = _split_preference_example(example)
    if auto_insert_empty_system_msg:
        prompt_messages = _maybe_insert_empty_system(prompt_messages, tokenizer)

    example["prompt"] = _apply_chat_template_non_thinking(
        tokenizer,
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    example["chosen"] = _assistant_suffix_from_template(tokenizer, prompt_messages, chosen_response)
    example["rejected"] = _assistant_suffix_from_template(tokenizer, prompt_messages, rejected_response)
    example["chat_template_applied"] = True
    return example


def _response_text(value: Any) -> Optional[str]:
    """Extract the assistant response text from common pair formats."""
    if isinstance(value, str):
        return value
    if isinstance(value, list) and value:
        last = value[-1]
        if isinstance(last, dict):
            return last.get("content")
        if isinstance(last, str):
            return last
    if isinstance(value, dict):
        return value.get("content")
    return None


def _find_response_score(response: Optional[str], responses: Any, scores: Any) -> Optional[float]:
    if response is None or not isinstance(responses, list) or not isinstance(scores, list):
        return None
    if len(responses) != len(scores):
        return None

    for candidate, score in zip(responses, scores):
        if candidate == response:
            return float(score)

    response_norm = response.strip()
    for candidate, score in zip(responses, scores):
        if isinstance(candidate, str) and candidate.strip() == response_norm:
            return float(score)
    return None


def add_ronpo_target(example: Dict[str, Any], mode: str, target_column: str, tie_threshold: float) -> Dict[str, Any]:
    """
    Add a practical RONPO relative target for the current pair.

    The full RONPO algorithm samples an adversarial atom (k, a) and queries two
    binary labels z_y and z_y'.  The current UltraFeedback-style data has one
    scalar RM score list per prompt, so the feasible first LLM experiment uses
    the sign of the chosen-vs-rejected score gap as a surrogate relative label.
    """
    mode = (mode or "none").lower()
    if mode == "none":
        return example
    if mode == "ordered":
        example[target_column] = 1.0
        return example
    if mode != "score_diff_sign":
        raise ValueError(f"Unsupported ronpo_target_mode={mode!r}")

    chosen = _response_text(example.get("chosen"))
    rejected = _response_text(example.get("rejected"))
    chosen_score = _find_response_score(chosen, example.get("all_generated_responses"), example.get("all_rm_scores"))
    rejected_score = _find_response_score(rejected, example.get("all_generated_responses"), example.get("all_rm_scores"))

    if chosen_score is None or rejected_score is None:
        # The annotation scripts write chosen=max score and rejected=min score.
        # Falling back to +1 keeps older scored files usable.
        target = 1.0
        gap = None
    else:
        gap = chosen_score - rejected_score
        if gap > tie_threshold:
            target = 1.0
        elif gap < -tie_threshold:
            target = -1.0
        else:
            target = 0.0

    example[target_column] = float(target)
    example["ronpo_chosen_score"] = chosen_score
    example["ronpo_rejected_score"] = rejected_score
    example["ronpo_score_gap"] = gap
    return example



def compute_and_add_logps(
    dataset: DatasetDict,
    model_path: str,
    tokenizer: PreTrainedTokenizerBase,
    args: ScriptArguments,
    accelerator: Accelerator,
    column_prefix: str,
) -> DatasetDict:
    logger.info(f"--- Processing model: {model_path} for columns with prefix: '{column_prefix}' ---")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        use_cache=False,
        cache_dir=args.cache_dir,
    ).eval()
    model = accelerator.prepare_model(model)

    data_collator = PreferenceDataCollatorWithPadding(
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        label_pad_token_id=-100,
        padding_value=0,
        truncation_mode=args.truncation_mode,
        is_encoder_decoder=False,
        max_target_length=None,
        mask_prompt=args.mask_prompt,
    )

    for split in dataset.keys():
        split_dataset = dataset[split]
        dataloader = DataLoader(
            split_dataset,
            batch_size=args.per_device_train_batch_size,
            shuffle=False,
            collate_fn=data_collator,
        )
        dataloader = accelerator.prepare(dataloader)

        all_chosen_logps, all_rejected_logps = [], []
        average_log_prob = str(getattr(args, "logp_reduction", "mean")).lower() == "mean"
        for batch in tqdm(dataloader, desc=f"Computing '{column_prefix}' logps for split={split}"):
            with torch.no_grad():
                chosen_logps, rejected_logps = concatenated_forward(
                    model, batch, average_log_prob=average_log_prob
                )

            chosen_logps, rejected_logps = accelerator.gather_for_metrics((chosen_logps, rejected_logps))
            all_chosen_logps.append(chosen_logps.cpu())
            all_rejected_logps.append(rejected_logps.cpu())

        chosen_arr = torch.cat(all_chosen_logps).float().numpy()
        rejected_arr = torch.cat(all_rejected_logps).float().numpy()

        assert len(split_dataset) == len(chosen_arr), "number of logps should be the same as number of samples"

        dataset[split] = split_dataset.add_column(f"{column_prefix}_chosen_logps", chosen_arr)
        dataset[split] = dataset[split].add_column(f"{column_prefix}_rejected_logps", rejected_arr)

    del model
    accelerator.free_memory()
    torch.cuda.empty_cache()
    return dataset


def copy_logp_columns(dataset: DatasetDict, src_prefix: str, dst_prefix: str) -> DatasetDict:
    """Reuse already-computed logps when two model paths are intentionally identical."""
    src_chosen = f"{src_prefix}_chosen_logps"
    src_rejected = f"{src_prefix}_rejected_logps"
    dst_chosen = f"{dst_prefix}_chosen_logps"
    dst_rejected = f"{dst_prefix}_rejected_logps"

    for split in dataset.keys():
        split_dataset = dataset[split]
        if dst_chosen in split_dataset.column_names and dst_rejected in split_dataset.column_names:
            continue
        if src_chosen not in split_dataset.column_names or src_rejected not in split_dataset.column_names:
            raise ValueError(f"Cannot copy missing {src_prefix} logp columns for split={split}")
        dataset[split] = split_dataset.add_column(dst_chosen, split_dataset[src_chosen])
        dataset[split] = dataset[split].add_column(dst_rejected, split_dataset[src_rejected])
    return dataset


def _same_model_path(lhs: Optional[str], rhs: Optional[str]) -> bool:
    if not lhs or not rhs:
        return False
    lhs_norm = os.path.abspath(os.path.expanduser(lhs)) if os.path.exists(os.path.expanduser(lhs)) else lhs
    rhs_norm = os.path.abspath(os.path.expanduser(rhs)) if os.path.exists(os.path.expanduser(rhs)) else rhs
    return lhs_norm == rhs_norm


def load_flexible_dataset(dataset_name_or_path: str, cache_dir: Optional[str] = None, split: str = "train"):
    """
    Load a dataset from a local file/directory or the Hugging Face Hub.

    Args:
        dataset_name_or_path: path to file/dir or a Hub dataset id
        cache_dir: HF cache dir
        split: split name if applicable
    """
    # Case 1: exact file path
    if os.path.isfile(dataset_name_or_path):
        print(f"Detected local file: {dataset_name_or_path}")
        file_type = dataset_name_or_path.split(".")[-1]
        if file_type == "jsonl":
            file_type = "json"  # jsonl uses the 'json' loader
        print(f"Inferred file type '{file_type}'. Loading...")
        try:
            return load_dataset(
                file_type,
                data_files=dataset_name_or_path,
                split=split,
                cache_dir=cache_dir,
            )
        except Exception:
            print("Failed with inferred type; retrying with 'json' loader...")
            return load_dataset(
                "json",
                data_files=dataset_name_or_path,
                split=split,
                cache_dir=cache_dir,
            )

    # Case 2: directory (saved HF dataset) or Hub name
    if os.path.isdir(dataset_name_or_path):
        print(f"Detected local directory: {dataset_name_or_path}; trying to load from disk...")
        loaded_object = load_from_disk(dataset_name_or_path)

        if isinstance(loaded_object, Dataset):
            print("   -> Loaded a single Dataset; returning it directly.")
            return loaded_object
        elif isinstance(loaded_object, DatasetDict):
            print("   -> Loaded a DatasetDict; selecting split...")
            if split in loaded_object:
                return loaded_object[split]
            else:
                available_splits = list(loaded_object.keys())
                raise ValueError(
                    f"Split '{split}' not found in the loaded dataset. Available splits: {available_splits}"
                )
        else:
            raise TypeError(f"Unexpected object type loaded from disk: {type(loaded_object)}")

    # Case 3: Hub id
    print(f"No local path found: {dataset_name_or_path}; loading from the Hugging Face Hub...")
    return load_dataset(dataset_name_or_path, split=split, cache_dir=cache_dir)



def require_nonempty_split_file(path: Optional[str], flag: str) -> None:
    """Reject an empty REQUESTED split by name, before `datasets` sees it.

    `datasets` cannot infer a schema from zero rows and raises SchemaInferenceError
    deep inside its builder, naming neither the split nor the flag; further
    downstream an empty split reaches torch.cat([]). Both are replaced by the
    message below. Only local files are inspected; a hub id or a directory is
    left to the loader.
    """
    if not path or not os.path.isfile(path):
        return
    empty = os.path.getsize(path) == 0
    if not empty and path.endswith((".jsonl", ".json")):
        with open(path) as f:
            head = f.read(2048).strip()
        empty = (not head) or head in ("[]", "{}")
    if empty:
        raise ValueError(
            f"{flag} {path} contains no records. An empty requested split is never "
            f"silently dropped: omit {flag} for a train-only artifact, or point it at "
            "a file with rows. (build_nbpo_pairs writes pairs_test.jsonl "
            "unconditionally, so the file exists even when no prompts were held out.)"
        )


def main():
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    if str(script_args.logp_reduction).lower() not in ("mean", "sum"):
        raise ValueError(f"--logp_reduction must be 'mean' or 'sum', got {script_args.logp_reduction!r}")

    logging.basicConfig(level=logging.INFO)
    accelerator = Accelerator()

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, cache_dir=script_args.cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ----- Load raw train (and optional test) datasets -----
    # Emptiness is checked BEFORE loading. `datasets` cannot infer a schema from
    # zero rows and raises SchemaInferenceError deep inside its builder, which
    # says nothing about which split was empty or why it was requested; further
    # downstream an empty split reaches torch.cat([]). Both are replaced by a
    # named error here. build_nbpo_pairs always writes pairs_test.jsonl, so this
    # is the common case for a train-only run: omit --test_dir (the stage runner
    # gates the flag on scripts/nbpo/nbpo_common.py:jsonl_has_records).
    require_nonempty_split_file(script_args.train_dir, "--train_dir")
    logger.info(f"Loading initial raw train dataset from: {script_args.train_dir}")
    raw_train = load_flexible_dataset(script_args.train_dir, cache_dir=script_args.cache_dir, split="train")
    if len(raw_train) == 0:
        raise ValueError(f"--train_dir {script_args.train_dir} loaded zero records")


    # accept --test_dir or legacy --eval_dir
    test_path = script_args.test_dir or script_args.eval_dir
    raw_test = None
    if test_path:
        # build_nbpo_pairs writes pairs_test.jsonl unconditionally, so the file
        # exists even when no prompts were held out. A REQUESTED split that turns
        # out empty is an error here, named and explained, rather than an opaque
        # torch.cat([]) failure hundreds of lines later; callers that do not want
        # a test split must omit --test_dir (see scripts/nbpo/nbpo_common.py's
        # jsonl_has_records, which is what run_nbpo_stage gates the flag on).
        require_nonempty_split_file(test_path, "--test_dir")
        logger.info(f"Loading raw test dataset from: {test_path}")
        raw_test = load_flexible_dataset(test_path, cache_dir=script_args.cache_dir, split="train")
        if len(raw_test) == 0:
            raise ValueError(
                f"--test_dir {test_path} was requested but contains no records. "
                "Omit --test_dir for a train-only artifact, or point it at a "
                "non-empty split; an empty requested split is never silently dropped."
            )

    # Build a DatasetDict so everything downstream works on both splits
    if raw_test is not None:
        raw_dataset = DatasetDict({"train": raw_train, "test": raw_test})
    else:
        raw_dataset = DatasetDict({"train": raw_train})
    logger.info(f"Precomputing splits: {sorted(raw_dataset.keys())}")

    if (script_args.ronpo_target_mode or "none").lower() != "none":
        logger.info(
            "Adding RONPO relative target column '%s' with mode=%s.",
            script_args.ronpo_target_column,
            script_args.ronpo_target_mode,
        )
        raw_dataset = raw_dataset.map(
            lambda ex: add_ronpo_target(
                ex,
                mode=script_args.ronpo_target_mode,
                target_column=script_args.ronpo_target_column,
                tie_threshold=float(script_args.ronpo_tie_threshold),
            ),
            num_proc=12,
        )
        logger.info("RONPO target column added.")

    # ----- Normalize preference format for tokenization -----
    if script_args.apply_chat_template:
        logger.info("Applying model chat template to prompt/chosen/rejected columns...")
        raw_dataset = raw_dataset.map(
            apply_preference_chat_template,
            fn_kwargs={
                "tokenizer": tokenizer,
                "auto_insert_empty_system_msg": bool(script_args.auto_insert_empty_system_msg),
            },
            num_proc=12,
        )
        logger.info("Chat template formatting complete.")
    else:
        logger.info("Transforming 'chosen'/'rejected' columns from list-of-dicts to strings...")
        raw_dataset = raw_dataset.map(transform_chat_to_str, num_proc=12)
        logger.info("String transformation complete.")

    if script_args.sanity_check:
        raw_dataset["train"] = raw_dataset["train"].select(range(min(100, len(raw_dataset["train"]))))
        if "test" in raw_dataset:
            raw_dataset["test"] = raw_dataset["test"].select(range(min(100, len(raw_dataset["test"]))))

    # We'll compute logps on the tokenized dataset and store columns on it.
    dataset_with_logps = raw_dataset

    # Reference model logps
    if not script_args.ref_model:
        raise ValueError("--ref_model must be provided for precompute.")
    dataset_with_logps = compute_and_add_logps(
        dataset=dataset_with_logps,
        model_path=script_args.ref_model,
        tokenizer=tokenizer,
        args=script_args,
        accelerator=accelerator,
        column_prefix="reference",
    )

    # Historical models logps (optional)
    if script_args.history_paths:
        for i, model_path in enumerate(script_args.history_paths):
            history_prefix = f"history{i}"
            if _same_model_path(model_path, script_args.ref_model):
                logger.info(
                    "--- Reusing reference logps for %s because model path matches ref_model: %s ---",
                    history_prefix,
                    model_path,
                )
                dataset_with_logps = copy_logp_columns(dataset_with_logps, "reference", history_prefix)
            else:
                dataset_with_logps = compute_and_add_logps(
                    dataset=dataset_with_logps,
                    model_path=model_path,
                    tokenizer=tokenizer,
                    args=script_args,
                    accelerator=accelerator,
                    column_prefix=history_prefix,
                )

    # Save final dataset (DatasetDict with train and optionally test)
    if accelerator.is_main_process:
        logger.info(f"Saving final dataset (with logps) to: {script_args.output_dir}")
        dataset_with_logps.save_to_disk(script_args.output_dir)
        # Provenance sidecar: records the logp reduction plus canonical
        # tokenizer/chat-template hashes so training can detect a mean/sum or
        # tokenization mismatch (mandatory check for loss_type=nbpo).
        from mnpo_scripts.pair_tokenization import (
            tokenization_config,
            tokenization_config_hash,
        )
        from mnpo_scripts.precompute_provenance import (
            checkpoint_fingerprint,
            sha256_file_hex,
            tokenizer_content_hashes,
            write_precompute_meta,
        )

        tok_cfg = tokenization_config(
            max_length=int(script_args.max_length),
            max_prompt_length=int(script_args.max_prompt_length),
            truncation_mode=str(script_args.truncation_mode),
        )

        def _fp(path):
            return checkpoint_fingerprint(path) if path and os.path.isdir(path) else None

        # Weight-level identity of every policy whose logps are stored: history0 IS
        # the proximal centre pi_t of Eq. (15), and loss_type=nbpo verifies it.
        history_paths = list(script_args.history_paths or [])
        solver_sha = (sha256_file_hex(script_args.solver_artifact_path)
                      if script_args.solver_artifact_path else None)
        if solver_sha is None:
            try:  # pairs from build_nbpo_pairs carry the solver hash on every row
                with open(script_args.train_dir) as f:
                    first = json.loads(f.readline())
                solver_sha = first.get("solver_hash")
            except Exception:
                solver_sha = None
        meta = {
            "logp_reduction": str(script_args.logp_reduction).lower(),
            **tokenizer_content_hashes(tokenizer),
            "tokenizer_source": script_args.model_name_or_path,
            "model_fingerprint": _fp(script_args.model_name_or_path),
            "ref_model": script_args.ref_model,
            "reference_fingerprint": _fp(script_args.ref_model),
            "history_paths": history_paths,
            "history_fingerprints": [_fp(h) for h in history_paths],
            "pair_artifact_path": script_args.train_dir,
            "pair_artifact_sha256": (sha256_file_hex(script_args.train_dir)
                                     if os.path.isfile(script_args.train_dir) else None),
            "test_pair_artifact_sha256": (sha256_file_hex(test_path)
                                          if test_path and os.path.isfile(test_path) else None),
            "solver_artifact_sha256": solver_sha,
            "max_length": int(script_args.max_length),
            "max_prompt_length": int(script_args.max_prompt_length),
            "apply_chat_template": bool(script_args.apply_chat_template),
            # Tokenization provenance: pi_t's logps here and pi's logps in the
            # trainer must come from identical token ids, attention and label
            # masks, so the settings that determine them are hashed and checked
            # (mnpo_scripts/pair_tokenization.py).
            **tok_cfg,
            "tokenization_config_sha256": tokenization_config_hash(tok_cfg),
        }
        # Hash every file the dataset actually consists of -- Arrow shards
        # included -- so an edited artifact cannot reach the trainer. The
        # manifest is written first, then hashed into the sidecar, so one value
        # (precompute_manifest_sha256) pins the whole precomputed dataset.
        from mnpo_scripts.precompute_provenance import (
            write_precompute_manifest,
        )

        meta["dataset_splits"] = sorted(dataset_with_logps.keys())
        meta["split_sizes"] = {k: len(v) for k, v in dataset_with_logps.items()}
        manifest_path, manifest_sha = write_precompute_manifest(
            script_args.output_dir, splits=meta["dataset_splits"])
        meta["precompute_manifest_sha256"] = manifest_sha
        meta_path = write_precompute_meta(script_args.output_dir, meta)
        logger.info(f"Wrote provenance sidecar: {meta_path} "
                    f"(dataset manifest {manifest_path}, sha {manifest_sha[:12]})")
        logger.info("Script finished successfully.")


if __name__ == "__main__":
    main()
