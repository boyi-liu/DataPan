"""Loader for dataset 'a'.

This is a template instruction-tuning loader. Point ``cfg.dataset.data_dir``
at a directory containing ``a.jsonl`` (or adapt ``_read_raw`` to your source).
Each raw record is expected to look like::

    {"instruction": "...", "input": "...", "output": "..."}

The loader formats each record with a prompt template, tokenizes it, and masks
the prompt tokens in the labels so loss is only computed on the response.
"""

import os

from datasets import Dataset, load_dataset

PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task"
    "{maybe_input_note}. Write a response that appropriately "
    "completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n{input_block}### Response:\n"
)


def _format_prompt(example):
    has_input = bool(example.get("input", "").strip())
    return PROMPT_TEMPLATE.format(
        maybe_input_note=", paired with an input" if has_input else "",
        instruction=example["instruction"].strip(),
        input_block=f"### Input:\n{example['input'].strip()}\n\n" if has_input else "",
    )


def _read_raw(cfg):
    """Return a ``datasets.Dataset`` of raw records.

    Replace the body to load from the Hub, a parquet file, etc. Here we read a
    local JSONL file so the framework is runnable out of the box.
    """
    path = os.path.join(cfg.dataset.data_dir, "a.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Expected dataset 'a' at {path}. Place an a.jsonl there or edit "
            "dataset/load_a.py:_read_raw to point at your source."
        )
    return load_dataset("json", data_files=path, split="train")


def _make_tokenize_fn(cfg, tokenizer):
    max_length = cfg.model.max_length

    def tokenize(example):
        prompt = _format_prompt(example)
        response = example["output"].strip() + tokenizer.eos_token

        prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        full_ids = tokenizer(prompt + response, add_special_tokens=True)["input_ids"]
        full_ids = full_ids[:max_length]

        labels = list(full_ids)
        # Mask the prompt portion so loss is computed only on the response.
        for i in range(min(len(prompt_ids), len(labels))):
            labels[i] = -100

        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
        }

    return tokenize


def load(cfg, tokenizer):
    raw = _read_raw(cfg)

    tokenized = raw.map(
        _make_tokenize_fn(cfg, tokenizer),
        remove_columns=raw.column_names,
        desc="Tokenizing dataset 'a'",
    )

    val_split = cfg.dataset.validation_split or 0.0
    if val_split and val_split > 0:
        split = tokenized.train_test_split(test_size=val_split, seed=cfg.seed)
        return {"train": split["train"], "validation": split["test"]}
    return {"train": tokenized, "validation": None}
