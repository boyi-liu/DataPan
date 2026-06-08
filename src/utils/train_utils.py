"""Training utilities (thin wrapper around HuggingFace ``Trainer``)."""

import os
import random

import numpy as np
import torch
from transformers import (
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_training_args(cfg):
    return TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.train.epochs,
        per_device_train_batch_size=cfg.train.batch_size,
        per_device_eval_batch_size=cfg.train.batch_size,
        gradient_accumulation_steps=cfg.train.grad_accum,
        learning_rate=cfg.train.lr,
        warmup_ratio=cfg.train.warmup_ratio,
        weight_decay=cfg.train.weight_decay,
        logging_steps=cfg.train.logging_steps,
        save_strategy=cfg.train.save_strategy,
        bf16=bool(cfg.train.bf16),
        fp16=bool(cfg.train.fp16),
        gradient_checkpointing=bool(cfg.train.gradient_checkpointing),
        seed=cfg.seed,
        report_to=[],
    )


def build_trainer(cfg, model, tokenizer, train_dataset, eval_dataset=None,
                  training_args=None):
    collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, padding="longest", label_pad_token_id=-100
    )
    return Trainer(
        model=model,
        args=training_args or build_training_args(cfg),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
    )


def train(cfg, model, tokenizer, train_dataset, eval_dataset=None):
    """Fine-tune ``model`` on ``train_dataset`` and persist the result."""
    trainer = build_trainer(cfg, model, tokenizer, train_dataset, eval_dataset)
    trainer.train()
    os.makedirs(cfg.output_dir, exist_ok=True)
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    return trainer
