"""Scorers: turn a dataset into per-example value scores (axes ①②③).

This is the **Scorer** layer of the executable design (Scorer -> Policy ->
Timing). A scorer fuses the three axes that don't vary independently in practice:

  * **① Scoring Metric** -- what signal defines data value (lexical relevance,
    representation similarity, perplexity, ...).
  * **② Comparison Target** -- what the example is measured *against* (a target /
    validation anchor, the corpus itself, or nothing).
  * **③ Scoring Model** -- which model produces the signal (none for BM25, the
    frozen base model for embedding / perplexity, ...).

Splitting those three apart needs a common intermediate that differs wildly in
cost and shape across methods, so they stay fused behind one primitive:

    score(train_dataset, val_dataset=None) -> (scores, features)

  * ``scores`` is an ``(N,)`` array following the convention **higher == more
    valuable**; scorers that prefer low values negate first, and unscorable
    examples sink to ``-inf``. This is exactly what a ④ :mod:`policy` consumes.
  * ``features`` is an optional ``(N, d)`` array of per-example vectors that
    interaction-aware policies (e.g. ``diversity``) need; score-only scorers
    return ``None``.

A scorer is *blind to how scores become a subset* (the ④ Policy axis) and to
*when* it runs (the ⑤ Timing axis). Each scorer lives in ``scorer/<name>.py`` and
defines a class named ``Scorer``; the matching offline ``alg/<name>.py`` selector
is then just "score, then hand to the policy".
"""

from abc import ABC, abstractmethod


class BaseScorer(ABC):
    """Maps a dataset to per-example value scores (and optional features)."""

    #: Whether this scorer requires a model + tokenizer (③). Model-free scorers
    #: (e.g. BM25) leave this False; forward-pass scorers set it True so the
    #: constructor can fail fast with a clear message.
    needs_model = False

    def __init__(self, cfg, model=None, tokenizer=None):
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer
        if self.needs_model and (model is None or tokenizer is None):
            raise ValueError(
                f"{type(self).__name__} needs a model and tokenizer."
            )

    @abstractmethod
    def score(self, train_dataset, val_dataset=None):
        """Score every example in ``train_dataset``.

        Returns ``(scores, features)`` where ``scores`` is an ``(N,)`` array
        (higher == more valuable; unscorable -> ``-inf``) and ``features`` is an
        optional ``(N, d)`` array for interaction-aware policies (else ``None``).
        """
