"""LESS four-dataset mixture -- prepare and load in one module.

Reproduces the FLAN V2 + CoT + Dolly + OpenAssistant1 training pool used by

  * LESS: Selecting Influential Data for Targeted Instruction Tuning
    (Xia et al., 2024)

It builds on the open-instruct / Tulu pre-processing, where each source is a
uniform ``messages`` record::

    {"dataset": "flan_v2", "id": "flan_v2_42",
     "messages": [{"role": "user", "content": "..."},
                  {"role": "assistant", "content": "..."}]}

This module consumes the four ``{name}_data.jsonl`` files, flattens each into
``{"instruction", "input", "output"}``, subsamples every source, shuffles the
union, and caches it as ``{data_dir}/less.jsonl``.

As a library (the registry calls this): ``load(cfg, tokenizer)`` builds that
cache on first use, then tokenizes it. Because the four source files have no
clean Hub mirror, point ``dataset.processed_dir`` at the directory holding them
(``-o dataset.processed_dir=./raw``); ``load`` raises if it is unset and the
cache is missing. Select with ``cfg.dataset.name = less``.

As a script: ``python -m dataset.load_less --processed_dir ./raw
[--sample_percentage 0.05] [--seed 3] [--output ...]``.

Getting the processed source files
----------------------------------
The four ``*_data.jsonl`` files are produced by open-instruct's
``reformat_datasets.py`` and ship with the LESS repo under
``data/train/processed`` (flat ``{name}_data.jsonl`` or ``{name}/{name}_data.jsonl``
layouts are both searched).
"""

import argparse
import json
import os
import random

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from dataset._common import subsample, tokenize_split, write_jsonl

DATASETS = ["flan_v2", "cot", "dolly", "oasst1"]
FILENAME = "less.jsonl"


# --------------------------------------------------------------------------- #
# messages -> {instruction, input, output}
# --------------------------------------------------------------------------- #
def messages_to_record(messages):
    """Flatten a Tulu ``messages`` conversation into a single SFT record.

    The response is the final assistant turn; any preceding turns (system /
    earlier user+assistant exchanges) are folded into ``instruction`` with role
    markers so multi-turn examples (common in OpenAssistant1) are preserved as
    context. ``input`` is left empty -- the conversation lives in ``instruction``.
    Returns ``None`` if there is no assistant turn to predict.
    """
    last_assistant = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "assistant" and messages[i].get("content", "").strip():
            last_assistant = i
            break
    if last_assistant is None:
        return None

    output = messages[last_assistant]["content"].strip()

    history = messages[:last_assistant]
    if len(history) == 1 and history[0].get("role") == "user":
        instruction = history[0]["content"].strip()
    else:
        # System prompt + multi-turn history -> a single tagged transcript.
        role_tag = {"system": "System", "user": "User", "assistant": "Assistant"}
        parts = []
        for m in history:
            content = m.get("content", "").strip()
            if not content:
                continue
            parts.append(f"{role_tag.get(m.get('role'), m.get('role', ''))}: {content}")
        instruction = "\n\n".join(parts)

    if not instruction:
        return None
    return {"instruction": instruction, "input": "", "output": output}


# --------------------------------------------------------------------------- #
# source reading & mixing
# --------------------------------------------------------------------------- #
def _find_processed_file(processed_dir, name):
    """Locate ``{name}_data.jsonl`` in a flat or per-dataset-subfolder layout."""
    candidates = [
        os.path.join(processed_dir, f"{name}_data.jsonl"),
        os.path.join(processed_dir, name, f"{name}_data.jsonl"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def read_processed(processed_dir, name):
    """Yield unified records from a Tulu-format ``{name}_data.jsonl`` file."""
    path = _find_processed_file(processed_dir, name)
    if path is None:
        raise FileNotFoundError(
            f"Could not find '{name}_data.jsonl' under {processed_dir!r}. Expected "
            f"{processed_dir}/{name}_data.jsonl or {processed_dir}/{name}/{name}_data.jsonl. "
            "See the module docstring for how to produce the processed files."
        )
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            record = messages_to_record(obj.get("messages", []))
            if record is not None:
                yield record


def build_records(processed_dir, datasets=DATASETS, sample_percentage=1.0, seed=3):
    """Read, subsample, tag and shuffle the four sources; return the mixture.

    Also returns per-source ``{name: (available, kept)}`` stats.
    """
    combined = []
    stats = {}
    for name in datasets:
        records = list(read_processed(processed_dir, name))
        kept = subsample(records, sample_percentage, seed)
        for r in kept:
            r["dataset"] = name
        stats[name] = (len(records), len(kept))
        combined.extend(kept)
    random.Random(seed).shuffle(combined)
    return combined, stats


def prepare(cfg):
    """Build ``{data_dir}/less.jsonl`` if missing; return its path."""
    path = os.path.join(cfg.dataset.data_dir, FILENAME)
    if not os.path.exists(path):
        processed_dir = cfg.dataset.processed_dir
        if not processed_dir:
            raise FileNotFoundError(
                f"{path} not found and dataset.processed_dir is unset. Point it at the "
                "four Tulu {name}_data.jsonl files (-o dataset.processed_dir=./raw), or "
                f"pre-build with `python -m dataset.load_less --processed_dir <dir> --output {path}`."
            )
        records, _ = build_records(
            processed_dir,
            sample_percentage=cfg.dataset.sample_percentage or 1.0,
            seed=cfg.seed,
        )
        write_jsonl(records, path)
        print(f"[less] wrote {len(records)} examples to {path}")
    return path


def load(cfg, tokenizer):
    return tokenize_split(cfg, tokenizer, prepare(cfg))


def _cli():
    p = argparse.ArgumentParser(description="Build the LESS mixture JSONL.")
    p.add_argument("--processed_dir", default="./raw",
                   help="Directory with the four Tulu-format {name}_data.jsonl files.")
    p.add_argument("--output", default=f"./data/{FILENAME}", help="Output JSONL path.")
    p.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS,
                   help="Subset of sources to mix (default: all four).")
    p.add_argument("--sample_percentage", type=float, default=1.0,
                   help="Fraction of each source to keep (LESS warmup uses 0.05).")
    p.add_argument("--seed", type=int, default=3, help="Subsample/shuffle seed (LESS uses 3).")
    a = p.parse_args()

    records, stats = build_records(a.processed_dir, a.datasets, a.sample_percentage, a.seed)
    write_jsonl(records, a.output)

    print(f"Wrote {len(records)} examples to {a.output}")
    print(f"{'dataset':<12}{'available':>12}{'kept':>10}")
    for name in a.datasets:
        avail, kept = stats[name]
        print(f"{name:<12}{avail:>12}{kept:>10}")


if __name__ == "__main__":
    _cli()
