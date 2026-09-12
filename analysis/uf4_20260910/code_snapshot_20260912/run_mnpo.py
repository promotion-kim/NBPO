import logging
import inspect
import json
import os
import random
import sys
from tqdm import tqdm

import torch
import torch.nn.functional as F
import transformers
if os.environ.get("MNPO_DISABLE_APEX", "").lower() in {"1", "true", "yes"}:
    # Some NGC images expose an incomplete system `apex` package without
    # `apex.amp`.  Transformers 4.x treats its mere presence as availability
    # and then fails while importing Trainer.  This repair path is bf16/AdamW
    # and does not use Apex, so disable only the stale availability flag.
    import transformers.utils.import_utils as _transformers_import_utils

    if hasattr(_transformers_import_utils, "_apex_available"):
        _transformers_import_utils._apex_available = False
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, set_seed

from alignment import (
    DataArguments,
    H4ArgumentParser,
    ModelArguments,
    get_checkpoint,
    get_datasets,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    get_tokenizer,
    is_adapter_model,
)
from alignment.data import maybe_insert_system_message, is_openai_format
from peft import PeftConfig, PeftModel

from mnpo_scripts.mnpo_trainer import MNPOTrainer
from mnpo_scripts.mnpo_config import MNPOConfig
from datasets import load_from_disk
from mnpo_scripts.response_logps import response_logps
from mnpo_scripts.nbpo_neural import (select_training_splits,
                                      validate_canonical_pair_dataset,
                                      validate_mopo_rho_dataset)
# =====================================================================================


def _apply_chat_template_non_thinking(tokenizer, messages, **kwargs):
    signature = inspect.signature(tokenizer.apply_chat_template)
    supports_extra_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if "enable_thinking" in signature.parameters or supports_extra_kwargs:
        kwargs.setdefault("enable_thinking", False)
    return tokenizer.apply_chat_template(messages, **kwargs)

logger = logging.getLogger(__name__)

MISTRAL_CHAT_TEMPLATE = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'].strip() + '\n\n' %}{% else %}{% set loop_messages = messages %}{% set system_message = '' %}{% endif %}{% for message in loop_messages %}{% if loop.index0 == 0 %}{% set content = system_message + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"


# =====================================================================================
# NEW: Utility function for logps calculation (adapted from TRL's DPOTrainer)
# =====================================================================================
def get_batch_logps(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = True,
        label_pad_token_id: int = -100,
) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits."""
    return response_logps(logits, labels, average_log_prob, label_pad_token_id)


# =====================================================================================


def apply_chat_template(
        example,
        tokenizer,
        task: str,
        auto_insert_empty_system_msg: bool = True,
        change_template=None,
):
    # This function remains largely the same, but we will call it for "mnpo" task
    if change_template == "mistral":
        tokenizer.chat_template = MISTRAL_CHAT_TEMPLATE

    if task == "mnpo":  # We can reuse the same logic as simpo/dpo
        if all(k in example.keys() for k in ("chosen", "rejected")):
            if not is_openai_format(example["chosen"]) or not is_openai_format(example["rejected"]):
                raise ValueError(f"Require OpenAI format for all messages")

            if "prompt" in example and is_openai_format(example["prompt"]):
                prompt_messages = example["prompt"]
                chosen_messages = example["chosen"]
                rejected_messages = example["rejected"]
            else:
                prompt_messages = example["chosen"][:-1]
                chosen_messages = example["chosen"][-1:]
                rejected_messages = example["rejected"][-1:]

            if auto_insert_empty_system_msg:
                maybe_insert_system_message(prompt_messages, tokenizer)

            example["text_prompt"] = _apply_chat_template_non_thinking(tokenizer, prompt_messages, tokenize=False)
            example["text_chosen"] = _apply_chat_template_non_thinking(tokenizer, chosen_messages, tokenize=False)
            if example["text_chosen"].startswith(tokenizer.bos_token):
                example["text_chosen"] = example["text_chosen"][len(tokenizer.bos_token):]
            example["text_rejected"] = _apply_chat_template_non_thinking(tokenizer, rejected_messages, tokenize=False)
            if example["text_rejected"].startswith(tokenizer.bos_token):
                example["text_rejected"] = example["text_rejected"][len(tokenizer.bos_token):]
        else:
            raise ValueError(f"Could not format example for `{task}` task!")
    else:
        raise ValueError(f"Task {task} not supported.")
    return example


def main():
    # The B200 NGC image ships a newer ProcessGroupNCCL than the original
    # training environment.  Older Accelerate does not pass device_id to its
    # early dataset barriers, so make the torchrun local-rank mapping explicit
    # before any distributed synchronization.  This changes no training math.
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None and torch.cuda.is_available():
        local_rank_int = int(local_rank)
        torch.cuda.set_device(local_rank_int)
        # Accelerate 0.29 calls torch.distributed.barrier() without device_ids.
        # Torch/NCCL in the 2025.12 B200 image guesses a device in that case and
        # aborts in the allocator.  Preserve every explicit caller argument and
        # add only the missing local device for legacy calls.
        original_barrier = torch.distributed.barrier

        def _barrier_with_local_device(*args, **kwargs):
            kwargs.setdefault("device_ids", [local_rank_int])
            return original_barrier(*args, **kwargs)

        torch.distributed.barrier = _barrier_with_local_device

    if os.environ.get("MNPO_DISABLE_CUDNN_SDPA", "").lower() in {"1", "true", "yes"}:
        if not hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            raise RuntimeError("MNPO_DISABLE_CUDNN_SDPA was requested but this torch build lacks the backend switch")
        torch.backends.cuda.enable_cudnn_sdp(False)
        if torch.backends.cuda.cudnn_sdp_enabled():
            raise RuntimeError("failed to disable the cuDNN SDPA backend")
        print("[runtime] cuDNN SDPA disabled; using another enabled PyTorch SDPA backend", flush=True)

    # =====================================================================================
    # MODIFIED: Use MNPOConfig
    # =====================================================================================
    parser = H4ArgumentParser((ModelArguments, DataArguments, MNPOConfig))
    parsed = parser.parse()

    if isinstance(parsed, tuple):
        model_args, data_args, training_args = parsed
    else:
        raise RuntimeError("Expected 3 arguments (Model, Data, Training), got 1.")

    # =====================================================================================
    #######
    # Setup
    #######
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    set_seed(training_args.seed)

    ###############
    # Load datasets
    ###############
    if len(data_args.dataset_mixer) > 1:
        raise ValueError("Direct loading only supports a single dataset in dataset_mixer.")

    dataset_path = list(data_args.dataset_mixer.keys())[0]
    logger.info(f"Loading pre-processed dataset directly from disk: {dataset_path}")

    raw_datasets = load_from_disk(dataset_path)

    from datasets import DatasetDict
    if not isinstance(raw_datasets, DatasetDict):
        raw_datasets = DatasetDict({"train": raw_datasets})

    train_dataset, eval_dataset = select_training_splits(
        raw_datasets, data_args.train_split, data_args.eval_split)

    logger.info(f"Using '{next(k for k, v in raw_datasets.items() if v is train_dataset)}' for training.")
    if eval_dataset:
        logger.info(f"Using '{next(k for k, v in raw_datasets.items() if v is eval_dataset)}' for evaluation.")
    else:
        logger.info("No evaluation split found or selected.")

    logger.info(f"Loaded dataset splits: {list(raw_datasets.keys())}")
    #####################################
    # Load tokenizer
    #####################################
    data_args.truncation_side = "left"
    tokenizer = get_tokenizer(model_args, data_args)

    column_names = list(train_dataset.features)
    # Provenance pinning applies to every mode that declares a pinned dataset;
    # only the structural validator below is format-specific. Splitting the two
    # lets an arm whose dataset is not pair-shaped still have its manifest,
    # tokenizer, chat template and model revision pinned rather than exempted.
    if training_args.nbpo_target_mode in ("canonical_logratio", "mopo_rho"):
        from mnpo_scripts.precompute_provenance import verify_precompute_manifest, tokenizer_content_hashes
        expected_manifest = training_args.nbpo_expected_dataset_manifest_sha256
        if not expected_manifest:
            raise ValueError("Canonical training config must pin nbpo_expected_dataset_manifest_sha256")
        verify_precompute_manifest(dataset_path, expected_manifest_sha256=expected_manifest)
        with open(os.path.join(dataset_path, "nbpo_dataset_provenance.json")) as handle:
            provenance = json.load(handle)["provenance"]
        token_hashes = tokenizer_content_hashes(tokenizer)
        for key in ("tokenizer_hash", "chat_template_hash"):
            if provenance[key] != token_hashes[key]:
                raise ValueError(f"Immutable dataset {key} does not match the training tokenizer")
        if str(provenance["model_revision"]) != str(model_args.model_revision):
            raise ValueError("Dataset generation model revision differs from training model revision")
        for name, split in ((data_args.train_split, train_dataset), (data_args.eval_split, eval_dataset)):
            if split is None:
                continue
            expected_solver = (training_args.nbpo_expected_solver_artifact_sha256
                               if name == data_args.train_split else None)
            validator = (validate_mopo_rho_dataset
                         if training_args.nbpo_target_mode == "mopo_rho"
                         else validate_canonical_pair_dataset)
            report = validator(split, training_args.max_length,
                               training_args.max_prompt_length, expected_solver)
            logger.info("%s split %s validated: %s",
                        training_args.nbpo_target_mode, name, report)

    # Fail fast for the NBPO branch: validate the target column, the
    # single proximal center, and the precompute provenance sidecar (reduction +
    # tokenizer/chat-template hashes) BEFORE the policy model is loaded.
    if str(getattr(training_args, "loss_type", "")).lower() in ("nbpo", "nbpo_wbc"):
        from mnpo_scripts.mnpo_trainer import validate_nbpo_args
        from mnpo_scripts.precompute_provenance import (
            checkpoint_fingerprint,
            read_precompute_meta,
            tokenizer_content_hashes,
        )

        hashes = tokenizer_content_hashes(tokenizer)
        # The model being trained is initialised from pi_t, so its content
        # fingerprint is the proximal centre history0 must bind to.
        parent_fp = (checkpoint_fingerprint(model_args.model_name_or_path)
                     if os.path.isdir(model_args.model_name_or_path) else None)
        validate_nbpo_args(
            training_args,
            dataset_columns=column_names,
            precompute_meta=read_precompute_meta(dataset_path),
            tokenizer_hash=hashes["tokenizer_hash"],
            chat_template_hash=hashes["chat_template_hash"],
            expected_parent_fingerprint=parent_fp,
            dataset_dir=dataset_path,
        )
        logger.info("NBPO configuration validated against the precompute artifact.")

    # for index in random.sample(range(len(raw_datasets["train"])), 3):
    #     logger.info(f"Prompt sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['prompt']}")
    #     logger.info(f"Logps sample {index}: {raw_datasets['train'][index]['reference_chosen_logps']}")

    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        attn_implementation=model_args.attn_implementation,
    )

    model = model_args.model_name_or_path
    training_args.model_init_kwargs = model_kwargs

    # =====================================================================================
    # Instantiate MNPOTrainer
    # =====================================================================================
    trainer = MNPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    # The frozen proximal centre, loaded AFTER the trainer so it shares the
    # accelerator's device placement. It is a separate copy of pi_t held in eval
    # mode with gradients off: it is never the learner detached, and it is never
    # re-synced to the learner during training. Its only job is to be forwarded
    # through the same collated batch as the policy, so that h is a difference of
    # two log-probabilities computed on the identical kernel path.
    if (getattr(training_args, "nbpo_online_reference", False)
            or getattr(training_args, "nbpo_eval_online_reference", False)):
        ref_path = getattr(training_args, "nbpo_reference_model_path", "") or \
            model_args.model_name_or_path
        logger.info(f"*** Loading frozen NBPO reference (pi_t) from {ref_path} ***")
        ref = AutoModelForCausalLM.from_pretrained(
            ref_path, torch_dtype=next(trainer.model.parameters()).dtype, use_cache=False,
            revision=model_args.model_revision, attn_implementation=model_args.attn_implementation,
            trust_remote_code=model_args.trust_remote_code)
        ref.eval()
        for prm in ref.parameters():
            prm.requires_grad_(False)
        trainer.nbpo_reference_model = ref.to(trainer.accelerator.device)
        logger.info("*** NBPO online reference active: pi_t forwarded per batch ***")

    if training_args.nbpo_require_fp32_optimizer:
        from mnpo_scripts.nbpo_runtime import NBPOPrecisionCallback
        trainer.add_callback(NBPOPrecisionCallback(trainer))

    # =====================================================================================

    if os.environ.get("MNPO_EVAL_ONLY", "").lower() in {"1", "true", "yes"}:
        if eval_dataset is None:
            raise ValueError("MNPO_EVAL_ONLY requires the explicitly selected dev split")
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        return

    ###############
    # Training loop
    ###############
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training complete ***")
    if training_args.nbpo_profile_updates > 0:
        logger.info("Disposable runtime profile complete at %d updates; scheduler horizon was %d. No model export.",
                    trainer.state.global_step, training_args.max_steps)
        return

    skip_final_save = os.environ.get("MNPO_SKIP_FINAL_SAVE", "").lower() in {"1", "true", "yes"}
    if skip_final_save:
        logger.info("*** Skip final model save because MNPO_SKIP_FINAL_SAVE is set ***")
        if trainer.accelerator.is_main_process:
            trainer.tokenizer.save_pretrained(training_args.output_dir)
            trainer.model.config.use_cache = True
            trainer.model.config.save_pretrained(training_args.output_dir)
        logger.info("*** Training complete! ***")
        return

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Add this step to explicitly save the tokenizer.
    if trainer.accelerator.is_main_process:
        trainer.tokenizer.save_pretrained(training_args.output_dir)
        logger.info(f"Tokenizer saved to {training_args.output_dir}")

    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "dataset": list(data_args.dataset_mixer.keys()),
        "dataset_tags": list(data_args.dataset_mixer.keys()),
        "tags": ["alignment-handbook", "mnpo"],  # MODIFIED
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    # if training_args.do_eval:
    #     logger.info("*** Evaluate ***")
    #     metrics = trainer.evaluate()
    #     if "test" in raw_datasets:
    #         metrics["eval_samples"] = len(raw_datasets["test"])
    #     trainer.log_metrics("eval", metrics)
    #     trainer.save_metrics("eval", metrics)

    if training_args.push_to_hub is True:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)

    logger.info("*** Training complete! ***")


if __name__ == "__main__":
    main()
