"""Selection policies: turn per-example scores into per-example weights.

This is the **④ Selection Policy** axis, factored out as an orthogonal module. A
policy is *blind to what the score means* (axes ①②③); it only decides how a
vector of per-example scores -- and, for interaction-aware policies, features --
becomes a weighting of the examples.

The single primitive is :meth:`weights`: ``scores -> wᵢ >= 0``.

  * **Hard selection** (``hard``) emits a ``{0, 1}`` top-k mask -- keep or drop.
  * **Coverage** (``diversity``) emits a ``{0, 1}`` mask too, but greedily
    chooses a maximally-spread subset from per-example features rather than the
    plain top-k.

The subset is the positively-weighted indices (:meth:`select`), which the
generic trainer then fine-tunes on.

``cfg.selection.policy`` chooses one; ``policy/<name>.py`` defines a ``Policy``.
Score convention: **higher == more valuable** (selectors that prefer low values
negate first; unscorable examples sink to ``-inf``).
"""

from abc import ABC, abstractmethod

import numpy as np


class BasePolicy(ABC):
    """Maps per-example scores (and optional features) to per-example weights."""

    #: Whether :meth:`weights` requires per-example ``features``. Score-only
    #: policies (hard top-k) leave this False; coverage / submodular policies
    #: that reason about sample interactions (``diversity``) set it True.
    needs_features = False

    def __init__(self, cfg):
        self.cfg = cfg

    @abstractmethod
    def weights(self, scores, features=None):
        """Return non-negative per-sample weights, one per entry of ``scores``.

        Policies emit a ``{0, 1}`` keep/drop mask. ``features`` is an optional
        ``(N, d)`` array some policies (e.g. ``diversity``) need.
        """

    def select(self, scores, features=None):
        """The selected subset: the positively-weighted indices, sorted.

        Derived from :meth:`weights`, so every policy yields a subset for free.
        """
        w = np.asarray(self.weights(scores, features=features), dtype=np.float64)
        return sorted(int(i) for i in np.flatnonzero(w > 0))

    def _budget_to_k(self, n):
        """Resolve ``cfg.selection.budget`` into an absolute count for ``n`` items.

        A budget ``<= 1`` is read as a fraction of ``n``; anything larger is an
        absolute count. Clamped to ``[1, n]``.
        """
        budget = self.cfg.selection.budget
        k = int(round(budget * n)) if budget <= 1 else int(budget)
        return max(1, min(k, n))
