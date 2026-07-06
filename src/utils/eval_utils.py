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


# --------------------------------------------------------------------------- #
# Generation backends
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _generate_hf(cfg, model, tokenizer, prompts, max_new_tokens):
    """Greedy generation via transformers, batched for throughput.

    Prompts are left-padded so the generated tokens line up at the right edge of
    every row, letting us slice them off with a single prompt-length offset.
    """
    model.eval()
    batch_size = cfg.get_path("eval.batch_size") or 16
    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    gens = []
    try:
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start:start + batch_size]
            enc = tokenizer(batch, return_tensors="pt", truncation=True,
                            max_length=cfg.model.max_length, padding=True).to(cfg.device)
            out = model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            gen_ids = out[:, enc["input_ids"].shape[1]:]
            gens.extend(tokenizer.batch_decode(gen_ids, skip_special_tokens=True))
    finally:
        tokenizer.padding_side = prev_side
    return gens


def _generate_vllm(llm, prompts, max_new_tokens, lora_request):
    """Greedy generation via vLLM. ``llm`` is a ``vllm.LLM`` built by
    ``model_utils.load_vllm``; outputs come back in input order."""
    from vllm import SamplingParams

    sampling = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    kwargs = {"lora_request": lora_request} if lora_request is not None else {}
    outputs = llm.generate(prompts, sampling, **kwargs)
    return [o.outputs[0].text for o in outputs]


def evaluate(cfg, model, tokenizer, benchmark="gsm8k", limit=100,
             max_new_tokens=None, lora_request=None):
    """Benchmark ``model`` on ``benchmark``.

    The generation engine is chosen by ``cfg.eval.backend`` (``hf`` | ``vllm``):
    for ``hf`` pass a transformers model; for ``vllm`` pass a ``vllm.LLM`` (see
    ``main.stage_eval``). Scoring is backend-agnostic.
    """
    if benchmark not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark {benchmark!r}; choices: {list(BENCHMARKS)}")
    spec = BENCHMARKS[benchmark]
    prompts, golds = spec["examples"](limit)
    metric_name = spec["metric"]
    score_fn = METRICS[metric_name]

    if max_new_tokens is None:
        max_new_tokens = cfg.get_path("eval.max_new_tokens") or 256
    backend = (cfg.get_path("eval.backend") or "hf").lower()

    if backend == "vllm":
        gens = _generate_vllm(model, prompts, max_new_tokens, lora_request)
    else:
        gens = _generate_hf(cfg, model, tokenizer, prompts, max_new_tokens)

    total = sum(score_fn(gen, gold) for gen, gold in zip(gens, golds))
    score = total / max(1, len(golds))
    return {"benchmark": benchmark, "n": len(golds), "metric": metric_name,
            "score": score, "backend": backend}
