"""Base interface shared by all data-selection algorithms."""

from abc import ABC, abstractmethod


class BaseSelector(ABC):
    """Selects a subset of the training data.

    Subclasses (in ``alg/<method>.py``) implement :meth:`select` directly, so
    methods that don't fit a per-example scoring model (clustering, dedup,
    coreset, ...) are first-class. Score-based methods can use the
    :meth:`topk_by_score` helper to avoid re-implementing budget handling.
    """

    def __init__(self, cfg, model=None, tokenizer=None):
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer

    @abstractmethod
    def select(self, train_dataset, val_dataset=None):
        """Return a list of selected example indices into ``train_dataset``."""

    # ---- helpers for subclasses -------------------------------------------

    def _budget_to_k(self, n):
        """Resolve ``cfg.selection.budget`` into an absolute count for ``n`` examples."""
        budget = self.cfg.selection.budget
        k = int(round(budget * n)) if budget <= 1 else int(budget)
        return max(1, min(k, n))

    def topk_by_score(self, scores):
        """Pick the highest-scoring examples, honoring the configured budget.

        Convenience for score-based selectors; returns sorted indices.
        """
        k = self._budget_to_k(len(scores))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return sorted(ranked[:k])
