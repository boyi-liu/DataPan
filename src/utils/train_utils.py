"""Training utilities (thin wrapper around HuggingFace ``Trainer``)."""

import inspect
import math
import os
import random

import numpy as np
import torch
from transformers import (
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

# ``Trainer``'s ``tokenizer`` argument was renamed to ``processing_class``
# (deprecated in transformers 4.46, removed in v5). Pick whichever the
# installed version accepts so we support both.
TRAINER_TOKENIZER_KW = (
    "processing_class"
    if "processing_class" in inspect.signature(Trainer.__init__).parameters
    else "tokenizer"
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def warmup_steps_from_ratio(cfg, num_examples):
    """Convert ``train.warmup_ratio`` into an absolute step count.

    Mirrors HuggingFace's own ratio->steps math (single-device; world size and
    DataLoader drop_last are not accounted for).
    """
    eff_batch = max(1, cfg.train.batch_size * cfg.train.grad_accum)
    steps_per_epoch = max(1, math.ceil(num_examples / eff_batch))
    total_steps = math.ceil(steps_per_epoch * cfg.train.epochs)
    return math.ceil((cfg.train.warmup_ratio or 0.0) * total_steps)


def build_training_args(cfg, num_train_examples=None):
    kwargs = dict(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.train.epochs,
        per_device_train_batch_size=cfg.train.batch_size,
        per_device_eval_batch_size=cfg.train.batch_size,
        gradient_accumulation_steps=cfg.train.grad_accum,
        learning_rate=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        logging_steps=cfg.train.logging_steps,
        save_strategy=cfg.train.save_strategy,
        bf16=bool(cfg.train.bf16),
        fp16=bool(cfg.train.fp16),
        gradient_checkpointing=bool(cfg.train.gradient_checkpointing),
        seed=cfg.seed,
        report_to=[],
    )
    # Prefer the non-deprecated `warmup_steps`; fall back to `warmup_ratio` only
    # when the dataset size is unknown.
    if cfg.train.warmup_ratio:
        if num_train_examples is not None:
            kwargs["warmup_steps"] = warmup_steps_from_ratio(cfg, num_train_examples)
        else:
            kwargs["warmup_ratio"] = cfg.train.warmup_ratio
    return TrainingArguments(**kwargs)


def build_trainer(cfg, model, tokenizer, train_dataset, eval_dataset=None,
                  training_args=None):
    # Honor `lora.enable`: fine-tune adapters instead of all ~Bn parameters.
    from utils.model_utils import maybe_wrap_lora

    model = maybe_wrap_lora(cfg, model)
    if getattr(model, "config", None) is not None:
        model.config.use_cache = False
    # Gradient checkpointing + LoRA needs the (frozen) inputs to require grad.
    if cfg.train.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, padding="longest", label_pad_token_id=-100
    )
    return Trainer(
        model=model,
        args=training_args or build_training_args(cfg, len(train_dataset)),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        **{TRAINER_TOKENIZER_KW: tokenizer},
    )


def train(cfg, model, tokenizer, train_dataset, eval_dataset=None):
    """Fine-tune ``model`` on ``train_dataset`` and persist the result."""
    trainer = build_trainer(cfg, model, tokenizer, train_dataset, eval_dataset)
    trainer.train()
    os.makedirs(cfg.output_dir, exist_ok=True)
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    return trainer
