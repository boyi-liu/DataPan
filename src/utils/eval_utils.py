"""Minimal benchmark evaluation.

Just enough to close the loop end-to-end. Each benchmark registers an example
builder plus the metric to score it with:

* ``exact_match`` -- compare the final numeric answer (GSM8K grade-school math).
* ``rouge_l``     -- LCS-based ROUGE-L F1 for open-ended generation tasks
                     (e.g. DialogSum summarization).

Add new benchmarks by adding an entry to ``BENCHMARKS``. For serious
leaderboard numbers use a dedicated harness (lm-evaluation-harness,
OpenCompass, ...); this is a lightweight sanity check.
"""

import re

import torch

from dataset.formatting import format_prompt

_NUM = re.compile(r"-?\d[\d,]*\.?\d*")
_WORD = re.compile(r"\w+")


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _last_number(text):
    matches = _NUM.findall(text)
    if not matches:
        return None
    return matches[-1].replace(",", "").rstrip(".")


def _exact_match(pred, gold):
    """1.0 if the final numbers match, else 0.0 (GSM8K-style)."""
    if gold is None:
        return 0.0
    return float(_last_number(pred) == gold)


def _lcs_length(a, b):
    """Length of the longest common subsequence of token lists ``a`` and ``b``."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        curr = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            curr[j] = prev[j - 1] + 1 if x == y else max(prev[j], curr[j - 1])
        prev = curr
    return prev[-1]


def _rouge_l(pred, gold):
    """ROUGE-L F1 between a prediction and a reference (whitespace tokenized)."""
    pred_toks = _WORD.findall(pred.lower())
    gold_toks = _WORD.findall((gold or "").lower())
    if not pred_toks or not gold_toks:
        return 0.0
    lcs = _lcs_length(pred_toks, gold_toks)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_toks)
    recall = lcs / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


METRICS = {"exact_match": _exact_match, "rouge_l": _rouge_l}


# --------------------------------------------------------------------------- #
# Benchmarks
# --------------------------------------------------------------------------- #
def _gsm8k_examples(limit):
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    prompts = [f"Question: {q}\nAnswer:" for q in ds["question"]]
    golds = [_last_number(a) for a in ds["answer"]]
    return prompts, golds


def _dialogsum_examples(limit):
    from datasets import load_dataset
    ds = load_dataset("knkarthick/dialogsum", split="test")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    prompts, golds = [], []
    for dialogue, summary in zip(ds["dialogue"], ds["summary"]):
        prompts.append(format_prompt({
            "instruction": "Summarize the following dialogue.",
            "input": dialogue or "",
        }))
        golds.append((summary or "").strip())
    return prompts, golds


# Each entry: builder producing (prompts, golds) + the metric key to score with.
BENCHMARKS = {
    "gsm8k": {"examples": _gsm8k_examples, "metric": "exact_match"},
    "dialogsum": {"examples": _dialogsum_examples, "metric": "rouge_l"},
}


@torch.no_grad()
def evaluate(cfg, model, tokenizer, benchmark="gsm8k", limit=100, max_new_tokens=256):
    if benchmark not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark {benchmark!r}; choices: {list(BENCHMARKS)}")
    spec = BENCHMARKS[benchmark]
    prompts, golds = spec["examples"](limit)
    metric_name = spec["metric"]
    score_fn = METRICS[metric_name]

    model.eval()
    total = 0.0
    for prompt, gold in zip(prompts, golds):
        enc = tokenizer(prompt, return_tensors="pt", truncation=True,
                        max_length=cfg.model.max_length).to(cfg.device)
        out = model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        gen = tokenizer.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        total += score_fn(gen, gold)

    score = total / max(1, len(golds))
    return {"benchmark": benchmark, "n": len(golds), "metric": metric_name, "score": score}
