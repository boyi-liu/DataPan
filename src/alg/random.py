"""Random selection baseline.

A fully-working selector that ignores the model entirely. Useful as a sanity
check / lower bound while implementing smarter methods like LESS.
Enable with ``--method random``.
"""

import numpy as np

from alg.base import BaseSelector
from policy.hard import Policy  # only used for its budget -> k helper; policy is moot here


class Selector(BaseSelector):
    def __init__(self, cfg, model=None, tokenizer=None):
        super().__init__(cfg, model, tokenizer)
        self.policy = Policy(cfg)

    def select(self, train_dataset, val_dataset=None):
        rng = np.random.default_rng(self.cfg.seed)
        k = self.policy._budget_to_k(len(train_dataset))
        return sorted(rng.choice(len(train_dataset), size=k, replace=False).tolist())
