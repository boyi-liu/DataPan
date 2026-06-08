"""Random selection baseline.

A fully-working selector that ignores the model entirely. Useful as a sanity
check / lower bound while implementing smarter methods like LESS.
Enable with ``--method random``.
"""

import numpy as np

from selector.base import BaseSelector


class RandomSelector(BaseSelector):
    def select(self, train_dataset, val_dataset=None):
        rng = np.random.default_rng(self.cfg.seed)
        k = self._budget_to_k(len(train_dataset))
        return sorted(rng.choice(len(train_dataset), size=k, replace=False).tolist())


SELECTOR = RandomSelector
