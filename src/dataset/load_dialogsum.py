"""DialogSum dataset -- prepare and load in one module.

As a library (the registry calls this): ``load(cfg, tokenizer)`` builds
``{data_dir}/dialogsum.jsonl`` on first use -- the DialogSum dialogue
summarization corpus (Chen et al., 2021) reformatted to ``{"instruction",
"input", "output"}`` -- then tokenizes it. A fixed summarization ``instruction``
is paired with the ``dialogue`` as ``input`` and the reference ``summary`` as
``output``. Select with ``cfg.dataset.name = dialogsum``.

As a script: ``python -m dataset.load_dialogsum [--dataset_name ...]
[--data_files ...] [--sample_percentage ...]`` (re)builds just the JSONL.

The source defaults to the Hub (``knkarthick/dialogsum``). Override via
``dataset.source`` (Hub id) or ``dataset.data_files`` (a local file).
"""

import argparse
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from dataset._common import subsample, tokenize_split, write_jsonl

DEFAULT_SOURCE = "knkarthick/dialogsum"
FILENAME = "dialogsum.jsonl"
INSTRUCTION = "Summarize the following dialogue."


def to_record(example):
    """Map one DialogSum row to ``{instruction, input, output}`` or ``None``."""
    dialogue = (example.get("dialogue") or "").strip()
    output = (example.get("summary") or "").strip()
    if not dialogue or not output:
        return None
    return {"instruction": INSTRUCTION, "input": dialogue, "output": output}


def build_records(source=DEFAULT_SOURCE, data_files=None, sample_percentage=1.0, seed=3):
    """Download/read the source and return unified, subsampled records."""
    from datasets import load_dataset

    if data_files:
        ext = os.path.splitext(data_files)[1].lstrip(".").lower()
        fmt = {"jsonl": "json", "": "json"}.get(ext, ext)
        raw = load_dataset(fmt, data_files=data_files, split="train")
    else:
        raw = load_dataset(source, split="train")
    records = [r for r in (to_record(ex) for ex in raw) if r is not None]
    return subsample(records, sample_percentage, seed)


def prepare(cfg):
    """Build ``{data_dir}/dialogsum.jsonl`` if missing; return its path."""
    path = os.path.join(cfg.dataset.data_dir, FILENAME)
    if not os.path.exists(path):
        records = build_records(
            source=cfg.dataset.source or DEFAULT_SOURCE,
            data_files=cfg.dataset.data_files,
            sample_percentage=cfg.dataset.sample_percentage or 1.0,
            seed=cfg.seed,
        )
        write_jsonl(records, path)
        print(f"[dialogsum] wrote {len(records)} examples to {path}")
    return path


def load(cfg, tokenizer):
    return tokenize_split(cfg, tokenizer, prepare(cfg))


def _cli():
    p = argparse.ArgumentParser(description="Build the DialogSum JSONL.")
    p.add_argument("--dataset_name", default=DEFAULT_SOURCE, help="HF Hub dataset id.")
    p.add_argument("--data_files", default=None, help="Local file to load instead of the Hub.")
    p.add_argument("--output", default=f"./data/{FILENAME}", help="Output JSONL path.")
    p.add_argument("--sample_percentage", type=float, default=1.0, help="Fraction to keep.")
    p.add_argument("--seed", type=int, default=3, help="Subsample seed.")
    a = p.parse_args()
    records = build_records(a.dataset_name, a.data_files, a.sample_percentage, a.seed)
    write_jsonl(records, a.output)
    print(f"Wrote {len(records)} examples to {a.output}")


if __name__ == "__main__":
    _cli()
