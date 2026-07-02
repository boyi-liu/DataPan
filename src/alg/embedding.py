"""Embedding-similarity baseline -- offline top-k over representation proximity.

A thin selector: the scoring (cosine similarity of mean-pooled hidden states to a
reference centroid, axes ①②③) lives in :mod:`scorer.embedding`; here it is just
"score, then apply the ④ policy" offline. The scorer also exposes per-example
embeddings as features, so ``--policy diversity`` works out of the box. See
``scorer/embedding.py`` for the method and its knobs. Enable with
``--method embedding``.
"""

from alg.base import BaseSelector
from policy.hard import Policy  # ④ policy loaded directly (get_policy is for `default`)
from scorer.embedding import Scorer, add_args  # add_args re-exported for utils.options


class Selector(BaseSelector):
    def __init__(self, cfg, model=None, tokenizer=None):
        super().__init__(cfg, model, tokenizer)
        self.scorer = Scorer(cfg, model, tokenizer)
        self.policy = Policy(cfg)

    def select(self, train_dataset, val_dataset=None):
        scores, features = self.scorer.score(train_dataset, val_dataset)
        return self.apply_policy(scores, features=features)
