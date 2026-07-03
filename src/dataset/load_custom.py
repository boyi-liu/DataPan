"""Custom / uploaded dataset -- load a user-provided JSONL directly.

Unlike the curated loaders (alpaca, wizardlm, less) this one does no downloading
or reformatting: it reads the unified ``{"instruction", "input", "output"}``
JSONL pointed to by ``cfg.dataset.data_files`` and tokenizes it with the shared
pipeline. This is what the demo UI's "Uploaded dataset" option maps to.

    cfg.dataset.name       = custom
    cfg.dataset.data_files = /path/to/uploaded.jsonl

Select with ``--dataset custom -o dataset.data_files=./my.jsonl``.
"""

import os

from dataset._common import tokenize_split


def prepare(cfg):
    path = cfg.dataset.data_files
    if not path:
        raise ValueError(
            "dataset.name='custom' requires dataset.data_files to point at a "
            "unified JSONL (records of {instruction, input, output})."
        )
    if not os.path.exists(path):
        raise FileNotFoundError(f"custom dataset file not found: {path}")
    return path


def load(cfg, tokenizer):
    return tokenize_split(cfg, tokenizer, prepare(cfg))
