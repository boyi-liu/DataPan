"""Selection policies: turn per-example scores into per-example weights.

This is the **④ Selection Policy** axis, factored out as an orthogonal module. A
policy is *blind to what the score means* (axes ①②③); it only decides how a
vector of per-example scores -- and, for interaction-aware policies, features --
becomes a weighting of the examples.

The single primitive is :meth:`weights`: ``scores -> wᵢ >= 0``.

  * **Hard selection** emits a ``{0, 1}`` mask -- keep or drop.
  * **Soft reweighting** emits continuous weights -- a per-sample learning-rate
    multiplier, nothing dropped.

Selection is therefore the *binary special case* of reweighting, and both live
behind one method. The primitive is also **scope-agnostic**: ``scores`` may cover
the whole dataset (offline) or a single minibatch (online) -- the ⑤ Timing axis
decides the scope and what happens to the weights, not the policy:

  * **Offline** (``select``): weigh the whole set once; the subset is the
    positively-weighted indices. The generic Trainer can't consume soft weights,
    so offline collapses to the binary case.
  * **Online** (in the training loop): weigh the current batch each step and
    apply the weights to the per-sample loss -- reweighting *or* in-batch
    selection, depending only on the policy.

``cfg.selection.policy`` chooses one; ``policy/<name>.py`` defines a ``Policy``.
Score convention: **higher == more valuable** (selectors that prefer low values
negate first; unscorable examples sink to ``-inf``).
"""

from abc import ABC, abstractmethod

import numpy as np


class BasePolicy(ABC):
    """Maps per-example scores (and optional features) to per-example weights."""

    #: Whether :meth:`weights` requires per-example ``features``. Score-only
    #: policies (hard filtering, soft reweighting) leave this False; coverage /
    #: submodular policies that reason about sample interactions set it True.
    needs_features = False

    def __init__(self, cfg):
        self.cfg = cfg

    @abstractmethod
    def weights(self, scores, features=None):
        """Return non-negative per-sample weights, one per entry of ``scores``.

        Hard policies emit a ``{0, 1}`` numpy mask; soft policies emit continuous
        weights. ``features`` is an optional ``(N, d)`` array some policies need.
        Works whether ``scores`` is a whole dataset (offline) or a batch (online).
        """

    def select(self, scores, features=None):
        """Offline subset: the positively-weighted indices, sorted.

        Derived from :meth:`weights`, so every policy yields a subset for free --
        for soft policies that simply means "everything with weight > 0".
        """
        w = np.asarray(self.weights(scores, features=features), dtype=np.float64)
        return sorted(int(i) for i in np.flatnonzero(w > 0))

    def _budget_to_k(self, n):
        """Resolve ``cfg.selection.budget`` into an absolute count for ``n`` items.

        A budget ``<= 1`` is read as a fraction of ``n`` (of the dataset offline,
        of the batch online); anything larger is an absolute count. Clamped to
        ``[1, n]``.
        """
        budget = self.cfg.selection.budget
        k = int(round(budget * n)) if budget <= 1 else int(budget)
        return max(1, min(k, n))
